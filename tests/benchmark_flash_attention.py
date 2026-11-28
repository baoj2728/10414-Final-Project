
import sys
from pathlib import Path

# Import Needle from this repository.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np
import pandas as pd
import torch
import needle as ndl

from needle.backend_ndarray import ndarray_backend_cuda as dev


# ============================================================
# CONFIGURATION
# ============================================================

B, H, D = 1, 2, 64
SEQUENCE_LENGTHS = [128, 256, 512, 1024, 2048]

WARMUP = 10
REPEATS = 50

ATOL = 1e-4
RTOL = 1e-4

rng = np.random.default_rng(42)

OUTPUT_DIR = ROOT / "benchmarks"


# ============================================================
# CUDA ARRAY HELPERS
# ============================================================

def make_inputs(N):
    return [
        rng.standard_normal((B, H, N, D)).astype(np.float32)
        for _ in range(4)
    ]


def to_needle(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    arr = dev.Array(x.size)
    dev.from_numpy(x.reshape(-1), arr)
    return arr


def from_needle(arr, shape):
    strides = []
    stride = 1

    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim

    strides = tuple(reversed(strides))

    return np.array(
        dev.to_numpy(arr, shape, strides, 0),
        copy=True,
    )


def allocate_outputs(N):
    return tuple(
        dev.Array(B * H * N * D)
        for _ in range(3)
    )


# ============================================================
# PYTORCH REFERENCE
# ============================================================

def torch_reference(inputs, causal):
    q_np, k_np, v_np, do_np = inputs

    q, k, v = [
        torch.tensor(
            x,
            device="cuda",
            requires_grad=True,
        )
        for x in (q_np, k_np, v_np)
    ]

    dO = torch.tensor(do_np, device="cuda")

    scores = (q @ k.transpose(-2, -1)) / np.sqrt(D)

    if causal:
        N = q.shape[-2]

        mask = torch.ones(
            (N, N),
            device="cuda",
            dtype=torch.bool,
        ).triu(1)

        scores = scores.masked_fill(
            mask, float("-inf")
        )

    lse = torch.logsumexp(scores, dim=-1)
    p = torch.softmax(scores, dim=-1)
    o = p @ v

    o.backward(dO)

    return {
        "O": o.detach().cpu().numpy(),
        "LSE": lse.detach().cpu().numpy(),
        "dQ": q.grad.detach().cpu().numpy(),
        "dK": k.grad.detach().cpu().numpy(),
        "dV": v.grad.detach().cpu().numpy(),
    }


# ============================================================
# KERNELS
# ============================================================

forward_kernels = {
    "Baseline": dev.flash_attention_forward_baseline,
    "Optimized": dev.flash_attention_forward,
}

backward_kernels = {
    "Baseline": dev.flash_attention_backward_baseline,
    "Optimized": dev.flash_attention_backward,
}


# ============================================================
# CORRECTNESS
# ============================================================

def test_correctness(N, causal):
    inputs = make_inputs(N)
    reference = torch_reference(inputs, causal)

    q, k, v, dO = [
        to_needle(x) for x in inputs
    ]

    shape = (B, H, N, D)
    rows = []

    for name in forward_kernels:
        O = dev.Array(B * H * N * D)
        LSE = dev.Array(B * H * N)

        dQ, dK, dV = allocate_outputs(N)

        forward_kernels[name](
            q, k, v, O, LSE,
            B, H, N, D, causal,
        )

        backward_kernels[name](
            q, k, v, O, LSE, dO,
            dQ, dK, dV,
            B, H, N, D, causal,
        )

        torch.cuda.synchronize()

        outputs = {
            "O": from_needle(O, shape),
            "LSE": from_needle(LSE, (B, H, N)),
            "dQ": from_needle(dQ, shape),
            "dK": from_needle(dK, shape),
            "dV": from_needle(dV, shape),
        }

        row = {
            "N": N,
            "Causal": causal,
            "Implementation": name,
        }

        for key, actual in outputs.items():
            expected = reference[key]

            row[f"{key}_Max_Error"] = float(
                np.max(np.abs(actual - expected))
            )

            row[f"{key}_Pass"] = bool(
                np.allclose(
                    actual,
                    expected,
                    atol=ATOL,
                    rtol=RTOL,
                )
            )

        row["Pass"] = all(
            row[f"{key}_Pass"] for key in outputs
        )

        rows.append(row)

        print(
            f"N={N:4d} | Causal={causal} | "
            f"{name:10s} | "
            f"{'PASS' if row['Pass'] else 'FAIL'}"
        )

    return rows


# ============================================================
# TIMING
# ============================================================

def time_cuda(fn, warmup=WARMUP, repeats=REPEATS):
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []

    for _ in range(repeats):
        start.record()
        fn()
        end.record()
        end.synchronize()

        times.append(start.elapsed_time(end))

    return float(np.median(times))


def benchmark_final(N, causal):
    q, k, v, dO = [
        to_needle(x) for x in make_inputs(N)
    ]

    O = dev.Array(B * H * N * D)
    LSE = dev.Array(B * H * N)

    measurements = {}

    for name, fn in forward_kernels.items():

        def run_forward(fn=fn):
            fn(
                q, k, v, O, LSE,
                B, H, N, D, causal,
            )

        measurements[f"Forward_{name}"] = time_cuda(
            run_forward
        )

    # Prepare common inputs for backward benchmarks.
    dev.flash_attention_forward(
        q, k, v, O, LSE,
        B, H, N, D, causal,
    )

    torch.cuda.synchronize()

    for name, fn in backward_kernels.items():
        dQ, dK, dV = allocate_outputs(N)

        def run_backward(fn=fn):
            fn(
                q, k, v, O, LSE, dO,
                dQ, dK, dV,
                B, H, N, D, causal,
            )

        measurements[f"Backward_{name}"] = time_cuda(
            run_backward
        )

    measurements["Forward_Speedup"] = (
        measurements["Forward_Baseline"]
        / measurements["Forward_Optimized"]
    )

    measurements["Backward_Speedup"] = (
        measurements["Backward_Baseline"]
        / measurements["Backward_Optimized"]
    )

    return measurements


# ============================================================
# MAIN
# ============================================================

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("GPU:", torch.cuda.get_device_name(0))
    print(f"B={B}, H={H}, D={D}")
    print(f"Warmup={WARMUP}, Repeats={REPEATS}")

    correctness_rows = []

    print("\n=== FLASHATTENTION CORRECTNESS ===")

    for causal in [False, True]:
        for N in SEQUENCE_LENGTHS:
            correctness_rows.extend(
                test_correctness(N, causal)
            )

    correctness_df = pd.DataFrame(correctness_rows)

    correctness_path = (
        OUTPUT_DIR / "flash_attention_final_correctness.csv"
    )

    correctness_df.to_csv(
        correctness_path, index=False
    )

    if not correctness_df["Pass"].all():
        raise RuntimeError(
            f"Correctness test failed. See {correctness_path}"
        )

    print("\n=== FINAL FLASHATTENTION BENCHMARK ===")

    rows = []

    for causal in [False, True]:
        print(f"\nCausal={causal}")

        for N in SEQUENCE_LENGTHS:
            measurements = benchmark_final(N, causal)

            rows.append({
                "N": N,
                "Causal": causal,
                **measurements,
            })

            print(
                f"N={N:4d} | "
                f"Forward: "
                f"{measurements['Forward_Baseline']:.4f} -> "
                f"{measurements['Forward_Optimized']:.4f} ms "
                f"({measurements['Forward_Speedup']:.2f}x) | "
                f"Backward: "
                f"{measurements['Backward_Baseline']:.4f} -> "
                f"{measurements['Backward_Optimized']:.4f} ms "
                f"({measurements['Backward_Speedup']:.2f}x)"
            )

    benchmark_df = pd.DataFrame(rows)

    print("\n=== FINAL BENCHMARK TABLE ===")
    print(benchmark_df.round(4).to_string(index=False))

    benchmark_path = (
        OUTPUT_DIR / "flash_attention_final_benchmark.csv"
    )

    benchmark_df.to_csv(benchmark_path, index=False)

    print("\nSaved:")
    print(benchmark_path)
    print(correctness_path)


if __name__ == "__main__":
    main()
