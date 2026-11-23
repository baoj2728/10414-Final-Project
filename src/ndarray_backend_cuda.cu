#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>
#include <limits>
#include <cmath>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }

  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides


__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   *
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of a array
   *   offset: offset of a array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if (gid >= size) return;

  size_t index = offset;
  size_t remaining = gid;
  for (int i = shape.size - 1; i >= 0; i--) {
    size_t coord = remaining % shape.data[i];
    remaining /= shape.data[i];
    index += coord * strides.data[i];
  }

  out[gid] = a[index];
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the
   * relevant CUDA kernel. In this case, we illustrate how you should set this up (i.e., we give
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to
   * execute the underlying function.
   *
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}

__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                                   CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if (gid >= size) return;

  size_t index = offset;
  size_t remaining = gid;
  for (int i = shape.size - 1; i >= 0; i--) {
    size_t coord = remaining % shape.data[i];
    remaining /= shape.data[i];
    index += coord * strides.data[i];
  }

  out[index] = a[gid];
  /// END SOLUTION
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(a.size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                              VecToCuda(strides), offset);
  /// END SOLUTION
}

__global__ void ScalarSetitemKernel(size_t size, scalar_t val, scalar_t* out, CudaVec shape,
                                    CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if (gid >= size) return;

  size_t index = offset;
  size_t remaining = gid;
  for (int i = shape.size - 1; i >= 0; i--) {
    size_t coord = remaining % shape.data[i];
    remaining /= shape.data[i];
    index += coord * strides.data[i];
  }

  out[index] = val;
  /// END SOLUTION
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(size, val, out->ptr, VecToCuda(shape),
                                               VecToCuda(strides), offset);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////


__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a',
  // and store the result in array 'out'.
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseMulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * b[gid];
}

void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);

  EwiseMulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMulKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * val;
}

void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);

  ScalarMulKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}


__global__ void EwiseDivKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] / b[gid];
}

void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);

  EwiseDivKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarDivKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] / val;
}

void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);

  ScalarDivKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}


__global__ void ScalarPowerKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = std::pow(a[gid], val);
}

void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);

  ScalarPowerKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}


__global__ void EwiseMaximumKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = max(a[gid], b[gid]);
}

void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);

  EwiseMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMaximumKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = max(a[gid], val);
}

void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);

  ScalarMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}


__global__ void EwiseEqKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] == b[gid] ? 1.0f : 0.0f;
}

void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);

  EwiseEqKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarEqKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] == val ? 1.0f : 0.0f;
}

void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);

  ScalarEqKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}


__global__ void EwiseGeKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] >= b[gid] ? 1.0f : 0.0f;
}

void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);

  EwiseGeKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarGeKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] >= val ? 1.0f : 0.0f;
}

void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);

  ScalarGeKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}


__global__ void EwiseLogKernel(const scalar_t* a, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = std::log(a[gid]);
}

void EwiseLog(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);

  EwiseLogKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}


__global__ void EwiseExpKernel(const scalar_t* a, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = std::exp(a[gid]);
}

void EwiseExp(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);

  EwiseExpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}


__global__ void EwiseTanhKernel(const scalar_t* a, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = std::tanh(a[gid]);
}

void EwiseTanh(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);

  EwiseTanhKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}


/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */


////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////
constexpr size_t L = TILE;
constexpr size_t S = 16;
constexpr size_t V = 1;

__global__ void MatmulKernal(const scalar_t* a, const scalar_t* b, scalar_t* out,
                             uint32_t M, uint32_t N, uint32_t P) {
  __shared__ scalar_t sA[L][S];
  __shared__ scalar_t sB[S][L];
  float c[V][V] = {};
  float ar[V], br[V];

  size_t tx = threadIdx.x;
  size_t ty = threadIdx.y;

  size_t blockRow = blockIdx.y * L;
  size_t blockCol = blockIdx.x * L;

  for (size_t k = 0; k < N; k += S) {
    for (size_t idx = ty * blockDim.x + tx; idx < L * S; idx += blockDim.x * blockDim.y) {
      size_t row = idx / S;
      size_t col = idx % S;

      if (blockRow + row < M && k + col < N) sA[row][col] = a[(blockRow + row) * N + k + col];
      else sA[row][col] = 0;
    }

    for (size_t idx = ty * blockDim.x + tx; idx < L * S; idx += blockDim.x * blockDim.y) {
      size_t row = idx / L;
      size_t col = idx % L;

      if (k + row < N && blockCol + col < P) sB[row][col] = b[(k + row) * P + blockCol + col];
      else sB[row][col] = 0;

    }

    __syncthreads();

    for (size_t idx = 0; idx < S; idx++) {
      for (int i = 0; i < V; i++) {
        ar[i] = sA[threadIdx.y * V + i][idx];
      }
      for (int i = 0; i < V; i++) {
        br[i] = sB[idx][threadIdx.x * V + i];
      }

      for (int y = 0; y < V; y++) {
        for (int x = 0; x < V; x++) {
          c[y][x] += ar[y] * br[x];
        }
      }
    }
    __syncthreads();
  }

  size_t colBase = blockIdx.x * L + threadIdx.x * V;
  size_t rowBase = blockIdx.y * L + threadIdx.y * V;

  for (int y = 0; y < V; y++) {
    for (int x = 0; x < V; x++) {
      size_t row = rowBase + y;
      size_t col = colBase + x;

      if (row < M && col < P) {
        out[row * P + col] = c[y][x];
      }
    }
  }
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling,
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   *
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN SOLUTION
  dim3 threadsPerBlock(L / V, L / V);
  dim3 numBlocks((P + L - 1) / L,
                 (M + L - 1) / L);

  MatmulKernal<<<numBlocks, threadsPerBlock>>>(a.ptr, b.ptr, out->ptr, M, N, P);

  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////
__global__ void ReduceMaxKernal(const scalar_t* a, scalar_t* out, size_t size, size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    size_t start = gid * reduce_size;
    size_t end = start + reduce_size;

    scalar_t result = a[start];
    for (size_t i = start; i < end; i++) {
      result = max(result, a[i]);
    }
    out[gid] = result;
  }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);

  ReduceMaxKernal<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END SOLUTION
}


__global__ void ReduceSumKernal(const scalar_t* a, scalar_t* out, size_t size, size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    size_t start = gid * reduce_size;
    size_t end = start + reduce_size;

    scalar_t result = 0.0f;
    for (size_t i = start; i < end; i++) {
      result += a[i];
    }
    out[gid] = result;
  }
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you
   * can perform each reduction in a single CUDA thread.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);

  ReduceSumKernal<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END SOLUTION
}


constexpr int BR = 32;
constexpr int BC = 32;
constexpr int D_CONST = 64; // support generic size D in future
constexpr int NUM_THREADS = 128;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;


__global__ void FlashAttentionForwardKernal(const scalar_t* q, const scalar_t* k, const scalar_t* v, 
                                            scalar_t* out, scalar_t* lse, 
                                            size_t B, size_t H, size_t N, size_t D, bool causal) {
  constexpr int ROWS_PER_WARP = BR / NUM_WARPS;
  constexpr int FEATURES_PER_ROW = (D_CONST + WARP_SIZE - 1) / WARP_SIZE;
  constexpr int FEATURES_PER_THREAD = ROWS_PER_WARP * FEATURES_PER_ROW;
  size_t tileIdx = blockIdx.x;
  size_t BatchHeadIdx = blockIdx.y;
  size_t BatchIdx = BatchHeadIdx / H;
  size_t HeadIdx = BatchHeadIdx % H;
  size_t tid = threadIdx.x;
  size_t warp_id = tid / WARP_SIZE;
  size_t lane_id = tid % WARP_SIZE;
  size_t base = BatchHeadIdx * N * D_CONST;
  size_t Tr = (N + BR - 1) / BR;
  size_t Tc = (N + BC - 1) / BC;
  size_t q_start = tileIdx * BR;

  __shared__ scalar_t K_shared[BC][D_CONST];
  __shared__ scalar_t V_shared[BC][D_CONST];
  
  float q_reg[ROWS_PER_WARP][FEATURES_PER_ROW];
  float m[ROWS_PER_WARP];
  float l[ROWS_PER_WARP];
  float acc[ROWS_PER_WARP][FEATURES_PER_ROW];

  for (int r = 0; r < ROWS_PER_WARP; r++) {
    size_t query_row = q_start + warp_id * ROWS_PER_WARP + r;

    for (int f = 0; f < FEATURES_PER_ROW; f++) {
        int feature = lane_id + f * WARP_SIZE;

        if (query_row < N && feature < D_CONST) {
            q_reg[r][f] = q[base + query_row * D_CONST + feature];
        } else {
            q_reg[r][f] = 0.0f;
        }
    }
  }

  for (int r = 0; r < ROWS_PER_WARP; r++) {
    m[r] = -INFINITY;
    l[r] = 0.0f;

    for (int f = 0; f < FEATURES_PER_ROW; f++) {
        acc[r][f] = 0.0f;
    }
  }



  constexpr int ELEMENTS_PER_TILE = BC * D_CONST;
  constexpr int LOAD_ITERATIONS = (ELEMENTS_PER_TILE + NUM_THREADS - 1) / NUM_THREADS;

  const float scale = rsqrtf((float)D_CONST);

  for (size_t Ti = 0; Ti < Tc; Ti++) {

    // Init the tilied K, V matrices
    size_t iterations = (D_CONST * BC + NUM_THREADS - 1) / NUM_THREADS;
    for (size_t i = 0; i < iterations; i += 1) {
      size_t idx = tid + NUM_THREADS * i;
      
      if (idx < D_CONST * BC) {
        size_t row = idx / D_CONST;
        size_t col = idx % D_CONST;
        size_t key_row = Ti * BC + row;

        if (key_row < N) {
          K_shared[row][col] = k[base + key_row * D_CONST + col];
          V_shared[row][col] = v[base + key_row * D_CONST + col];
        } else {
          K_shared[row][col] = 0.0f;
          V_shared[row][col] = 0.0f;
        }
      }
    }
    __syncthreads();

    
    // Compute attention scores and online softmax
    for (int r = 0; r < ROWS_PER_WARP; r++) {
      size_t query_row = q_start + warp_id * ROWS_PER_WARP + r;

      if (query_row >= N) continue;

      for (int j = 0; j < BC; j++) {
        size_t key_row = Ti * BC + j;

        if (key_row >= N) continue;
        if (causal && key_row > query_row) continue;

        float partial = 0.0f;

        for (int f = 0; f < FEATURES_PER_ROW; f++) {
          int feature = lane_id + f * WARP_SIZE;

          if (feature < D_CONST) {
            partial += q_reg[r][f] * K_shared[j][feature];
          }
        }

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
          partial += __shfl_down_sync(0xffffffff, partial, offset);
        }

        float score = __shfl_sync(0xffffffff, partial, 0) * scale;

        float new_m = fmaxf(m[r], score);
        float alpha = expf(m[r] - new_m);
        float beta = expf(score - new_m);

        l[r] = alpha * l[r] + beta;
        
        for (int f = 0; f < FEATURES_PER_ROW; f++) {
          int feature = lane_id + f * WARP_SIZE;

          if (feature < D_CONST) {
              acc[r][f] = alpha * acc[r][f] + beta * V_shared[j][feature];
          }
        }

        m[r] = new_m;
      }
    }
    __syncthreads();
  }

  for (size_t r = 0; r < ROWS_PER_WARP; r++) {
    size_t query_row = q_start + warp_id * ROWS_PER_WARP + r;

    if (query_row >= N) continue;

    for (size_t f = 0; f < FEATURES_PER_ROW; f++) {
      int feature = lane_id + f * WARP_SIZE;
      if (feature < D_CONST) {
        out[base + query_row * D_CONST + feature] = acc[r][f] / l[r];
      }
    }

    if (lane_id == 0) lse[BatchHeadIdx * N + query_row] = m[r] + logf(l[r]);
  }


  
}

void FlashAttentionForward_Baseline(const CudaArray& q, const CudaArray& k, const CudaArray& v, CudaArray* out,
                           CudaArray* lse, uint32_t B, uint32_t H, uint32_t N, uint32_t D, bool causal) {
  dim3 grid((N + BR - 1) / BR, B * H);
  dim3 block(NUM_THREADS);

  FlashAttentionForwardKernal<<<grid, block>>>(q.ptr, k.ptr, v.ptr, out->ptr, lse->ptr,
                                               B, H, N, D, causal);
}

__global__ void InitDeltaMatrixKernel(const scalar_t* dO, const scalar_t* O, scalar_t* Dout,
                                  size_t B, size_t H, size_t N, size_t D) {
  constexpr int ROWS_PER_WARP = BC / NUM_WARPS;
  constexpr int FEATURES_PER_ROW = (D_CONST + WARP_SIZE - 1) / WARP_SIZE;
  constexpr int FEATURES_PER_THREAD = ROWS_PER_WARP * FEATURES_PER_ROW;
  size_t tileIdx = blockIdx.x;
  size_t BatchHeadIdx = blockIdx.y;
  size_t BatchIdx = BatchHeadIdx / H;
  size_t HeadIdx = BatchHeadIdx % H;
  size_t tid = threadIdx.x;
  size_t warp_id = tid / WARP_SIZE;
  size_t lane_id = tid % WARP_SIZE;
  size_t base = BatchHeadIdx * N * D_CONST;
  size_t d_start = tileIdx * BC;
                                    
  for (size_t r = 0; r < ROWS_PER_WARP; r++) {
    size_t d_row = d_start + warp_id * ROWS_PER_WARP + r;

    if (d_row >= N) continue;
    float sum = 0.0f;

    for (size_t f = 0; f < FEATURES_PER_ROW; f++) {
      size_t d_feature = f * WARP_SIZE + lane_id;

      if (d_feature < D_CONST) {
        size_t idx = base + d_row * D_CONST + d_feature;
        sum += O[idx] * dO[idx];
      }
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane_id == 0) Dout[BatchHeadIdx * N + d_row] = sum;
  }
}

__global__ void FlashAttentionBackwardKernal(const scalar_t* q, const scalar_t* k, const scalar_t* v, const scalar_t* o,
                                             const scalar_t* lse, const scalar_t* dO, scalar_t* dq, scalar_t* dk, scalar_t* dv,
                                             const scalar_t* delta, size_t B, size_t H, size_t N, size_t D, bool causal) {
  constexpr int ROWS_PER_WARP = BC / NUM_WARPS;
  constexpr int FEATURES_PER_ROW = (D_CONST + WARP_SIZE - 1) / WARP_SIZE;
  constexpr int FEATURES_PER_THREAD = ROWS_PER_WARP * FEATURES_PER_ROW;
  size_t tileIdx = blockIdx.x;
  size_t BatchHeadIdx = blockIdx.y;
  size_t BatchIdx = BatchHeadIdx / H;
  size_t HeadIdx = BatchHeadIdx % H;
  size_t tid = threadIdx.x;
  size_t warp_id = tid / WARP_SIZE;
  size_t lane_id = tid % WARP_SIZE;
  size_t base = BatchHeadIdx * N * D_CONST;
  size_t Tr = (N + BR - 1) / BR;
  size_t Tc = (N + BC - 1) / BC;
  size_t kv_start = tileIdx * BC;

  const float scale = rsqrtf((float)D_CONST);

  scalar_t Kj[ROWS_PER_WARP][FEATURES_PER_ROW];
  scalar_t Vj[ROWS_PER_WARP][FEATURES_PER_ROW];
  scalar_t dKj[ROWS_PER_WARP][FEATURES_PER_ROW];
  scalar_t dVj[ROWS_PER_WARP][FEATURES_PER_ROW];

  for (size_t r = 0; r < ROWS_PER_WARP; r++) {
    size_t kv_row = kv_start + warp_id * ROWS_PER_WARP + r;

    for (size_t f = 0; f < FEATURES_PER_ROW; f++) {
      size_t kv_feature = f * WARP_SIZE + lane_id;
      if (kv_feature < D_CONST && kv_row < N) {
        Kj[r][f] = k[base + kv_row * D_CONST + kv_feature];
        Vj[r][f] = v[base + kv_row * D_CONST + kv_feature];
      } else {
        Kj[r][f] = 0.0f;
        Vj[r][f] = 0.0f;
      }
      dKj[r][f] = 0.0f;
      dVj[r][f] = 0.0f;
      
    }
  }

  __shared__ scalar_t Qi[BR][D_CONST];
  __shared__ scalar_t dOi[BR][D_CONST];
  // __shared__ scalar_t dQi[BR][D_CONST];
  __shared__ scalar_t Li[BR];
  __shared__ scalar_t Di[BR];

  size_t BR_ROWS_PER_WARP = BR / NUM_WARPS;
  for (size_t i = 0; i < Tr; i++) {
    for (size_t r = 0; r < BR_ROWS_PER_WARP; r++) {
      size_t shared_row = warp_id * BR_ROWS_PER_WARP + r;
      size_t global_row = i * BR + warp_id * BR_ROWS_PER_WARP + r;

      if (lane_id == 0) {
        if (global_row < N) {
          Li[shared_row] = lse[BatchHeadIdx * N + global_row];
          Di[shared_row] = delta[BatchHeadIdx * N + global_row];
        } else {
          Li[shared_row] = 0.0f;
          Di[shared_row] = 0.0f;
        }
      }

      for (size_t f = 0; f < FEATURES_PER_ROW; f++) {
        size_t feature = f * WARP_SIZE + lane_id;

        if (global_row < N && feature < D_CONST) {
          Qi[shared_row][feature] = q[base + global_row * D_CONST + feature];
          dOi[shared_row][feature] = dO[base + global_row * D_CONST + feature];
          // dQi[shared_row][feature] = dq[base + global_row * D_CONST + feature];
        } else {
          Qi[shared_row][feature] = 0.0f;
          dOi[shared_row][feature] = 0.0f;
          // dQi[shared_row][feature] = 0.0f;
        }
      }
    }
    __syncthreads();

    for (int r = 0; r < ROWS_PER_WARP; r++) {
      size_t kv_row = kv_start + warp_id * ROWS_PER_WARP + r;

      if (kv_row >= N) continue;

      for (int qr = 0; qr < BR; qr++) {
        size_t query_row = i * BR + qr;

        if (query_row >= N) continue;
        if (causal && kv_row > query_row) continue;

        float score_partial = 0.0f;
        float dp_partial = 0.0f;

        for (size_t f = 0; f < FEATURES_PER_ROW; f++) {
          size_t feature = f * WARP_SIZE + lane_id;

          if (feature >= D_CONST) break;
          score_partial += Qi[qr][feature] * Kj[r][f];
          dp_partial += dOi[qr][feature] * Vj[r][f];
        }

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
          score_partial += __shfl_down_sync(0xffffffff, score_partial, offset);
          dp_partial += __shfl_down_sync(0xffffffff, dp_partial, offset);
        }

        float score = __shfl_sync(0xffffffff, score_partial, 0) * scale;
        float dp = __shfl_sync(0xffffffff, dp_partial, 0);

        float p = expf(score - Li[qr]);
        float ds = p * (dp - Di[qr]);

        for (int f = 0; f < FEATURES_PER_ROW; f++) {
          int feature = f * WARP_SIZE + lane_id;

          if (feature < D_CONST) {
            float q_val = Qi[qr][feature];
            float do_val = dOi[qr][feature];

            dKj[r][f] += ds * q_val * scale;
            dVj[r][f] += p * do_val;

            size_t dq_idx = base + query_row * D_CONST + feature;

            atomicAdd(&dq[dq_idx], ds * Kj[r][f] * scale);
          }
        }
      }
    }

    __syncthreads();
  }

  for (int r = 0; r < ROWS_PER_WARP; r++) {
    size_t kv_row = kv_start + warp_id * ROWS_PER_WARP + r;

    if (kv_row >= N) continue;

    for (int f = 0; f < FEATURES_PER_ROW; f++) {
      int feature = f * WARP_SIZE + lane_id;

      if (feature < D_CONST) {
        size_t idx = base + kv_row * D_CONST + feature;

        dk[idx] = dKj[r][f];
        dv[idx] = dVj[r][f];
      }
    }
  }
}


void FlashAttentionBackward_Baseline(const CudaArray& q, const CudaArray& k, const CudaArray& v, const CudaArray& o,
                            const CudaArray& lse, const CudaArray& dO, CudaArray* dq, CudaArray* dk, CudaArray* dv,
                            uint32_t B, uint32_t H, uint32_t N, uint32_t D, bool causal) {
  dim3 grid((N + BC - 1) / BC, B * H);
  dim3 block(NUM_THREADS);
  
  CudaArray delta((size_t)B * H * N);
  InitDeltaMatrixKernel<<<grid, block>>>(dO.ptr, o.ptr, delta.ptr, B, H, N, D);
  
  cudaMemset(dq->ptr, 0, (size_t)B * H * N * D_CONST * sizeof(scalar_t));
  FlashAttentionBackwardKernal<<<grid, block>>>(q.ptr, k.ptr, v.ptr, o.ptr, lse.ptr,
                                                dO.ptr, dq->ptr, dk->ptr, dv->ptr,
                                                delta.ptr, B, H, N, D, causal);
  cudaDeviceSynchronize();
}

__global__ void FlashAttentionForwardKernel_MatMul(
    const scalar_t* q, const scalar_t* k, const scalar_t* v,
    scalar_t* out, scalar_t* lse,
    size_t B, size_t H, size_t N, size_t D, bool causal) {

  // Number of queries assigned to each warp
  constexpr int ROWS_PER_WARP = BR / NUM_WARPS;

  // Divide each warp into sections for threads to process
  constexpr int LANES_PER_ROW = 8;
  constexpr int LANES_PER_COL = WARP_SIZE / LANES_PER_ROW;
  constexpr int ROWS_PER_THREAD =
      (ROWS_PER_WARP + LANES_PER_COL - 1) / LANES_PER_COL;
  constexpr int COLS_PER_THREAD =
      (BC + LANES_PER_ROW - 1) / LANES_PER_ROW;
  
  // Number of output feature columns assigned to each thread
  constexpr int FEATURES_PER_THREAD =
      (D_CONST + LANES_PER_ROW - 1) / LANES_PER_ROW;

  // Identify the query tile, batch/head, warp, and lane
  const size_t tileIdx = blockIdx.x;
  const size_t BatchHeadIdx = blockIdx.y;
  const size_t tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  // Offset of the current batch/head in Q/K/V
  const size_t base = BatchHeadIdx * N * D;
  /// Number of K/V tiles
  const size_t Tc = (N + BC - 1) / BC;
  // First query for this tile
  const size_t q_start = tileIdx * BR;

  // Map lanes to row and column positions
  const int row_group = lane_id / LANES_PER_ROW;
  const int col_group = lane_id % LANES_PER_ROW;

  // Starting row and column of this thread's section in S = QK^T
  const int row_base = row_group * ROWS_PER_THREAD;
  const int col_base = col_group * COLS_PER_THREAD;

  // Shared-memory tiles for Q, transposed K, and V
  __shared__ scalar_t Q_shared[BR][D_CONST];
  __shared__ scalar_t K_shared[D_CONST][BC];
  __shared__ scalar_t V_shared[BC][D_CONST];

  // Per-thread running softmax maximum and denominator
  float m[ROWS_PER_THREAD];
  float l[ROWS_PER_THREAD];

  // Per-thread output accumulator for O = PV
  float acc[ROWS_PER_THREAD][FEATURES_PER_THREAD];
  
  // Per-thread attention probabilities for the current K/V tile
  float p_reg[ROWS_PER_THREAD][COLS_PER_THREAD];

  // Load Q into shared memory
  for (size_t idx = tid;
       idx < (size_t)BR * D_CONST;
       idx += NUM_THREADS) {

    const int r = idx / D_CONST;
    const int d = idx % D_CONST;

    const size_t query_row = q_start + r;

    if (query_row < N && d < D) {
      Q_shared[r][d] = q[base + query_row * D + d];
    } else {
      Q_shared[r][d] = scalar_t(0);
    }
  }
  __syncthreads();

  // Initialize online softmax state and output accumulator
  for (int i = 0; i < ROWS_PER_THREAD; i++) {
    m[i] = -INFINITY;
    l[i] = 0.0f;

    for (int f = 0; f < FEATURES_PER_THREAD; f++) {
      acc[i][f] = 0.0f;
    }
  }

  // Number of elements in one K tile
  constexpr int ELEMENTS_PER_TILE = BC * D_CONST;

  // Number of cooperative loading iterations per thread
  constexpr int LOAD_ITERATIONS =
      (ELEMENTS_PER_TILE + NUM_THREADS - 1) / NUM_THREADS;

  // Scale attention scores by 1/sqrt(D)
  const float scale = rsqrtf((float)D);

  // Number of K/V tiles to attend to under causal attention
  const size_t causal_tile_limit =
      (q_start + BR + BC - 1) / BC;
  size_t tiles_to_process;
  if (causal && causal_tile_limit < Tc) {
      tiles_to_process = causal_tile_limit;
  } else {
      tiles_to_process = Tc;
  }

  // Mask including all 32 lanes for warp shuffle operations
  constexpr unsigned FULL_MASK = 0xffffffffu;

  // Load K transpose and V tiles into shared memory
  for (size_t Ti = 0; Ti < tiles_to_process; Ti++) {
    for (size_t i = 0; i < LOAD_ITERATIONS; i++) {
      const size_t idx = tid + NUM_THREADS * i;

      if (idx < ELEMENTS_PER_TILE) {
        const int row = idx / D_CONST;
        const int col = idx % D_CONST;

        const size_t key_row = Ti * BC + row;

        if (key_row < N && col < D) {
          K_shared[col][row] =
              k[base + key_row * D + col];

          V_shared[row][col] =
              v[base + key_row * D + col];
        } else {
          K_shared[col][row] = scalar_t(0);
          V_shared[row][col] = scalar_t(0);
        }
      }
    }
    __syncthreads();

    // Compute this thread's seciton of S = Q @ K^T
    float scores[ROWS_PER_THREAD][COLS_PER_THREAD] = {};

    for (int d = 0; d < D_CONST; d++) {
      float qr[ROWS_PER_THREAD];
      float kr[COLS_PER_THREAD];

      for (int i = 0; i < ROWS_PER_THREAD; i++) {
        const int q_local = warp_id * ROWS_PER_WARP + row_base + i;
        qr[i] = Q_shared[q_local][d];
      }

      for (int j = 0; j < COLS_PER_THREAD; j++) {
        const int c = col_base + j;
        kr[j] = K_shared[d][c];
      }

      for (int i = 0; i < ROWS_PER_THREAD; i++) {
        for (int j = 0; j < COLS_PER_THREAD; j++) {
          scores[i][j] = fmaf(qr[i], kr[j], scores[i][j]);
        }
      }
    }

    // Scale scores and apply masks
    for (int i = 0; i < ROWS_PER_THREAD; i++) {
      const int q_local = warp_id * ROWS_PER_WARP + row_base + i;
      const size_t query_row = q_start + q_local;

      for (int j = 0; j < COLS_PER_THREAD; j++) {
        const int c = col_base + j;
        const size_t key_row = Ti * BC + c;

        bool valid = q_local < BR && query_row < N &&
          c < BC && key_row < N;
        if (causal) {
          valid = valid && (key_row <= query_row);
        }

        if (valid) {
          scores[i][j] *= scale;
        } else {
          scores[i][j] = -INFINITY;
        }
      }
    }

    // Blockwise online softmax
    for (int i = 0; i < ROWS_PER_THREAD; i++) {
      const int q_local = warp_id * ROWS_PER_WARP + row_base + i;

      // Maximum score for this query
      float local_max = -INFINITY;
      for (int j = 0; j < COLS_PER_THREAD; j++) {
        local_max = fmaxf(local_max, scores[i][j]);
      }

      // Maximum across the eight lanes owning this query
      for (int offset = LANES_PER_ROW / 2; offset > 0; offset /= 2) {
        const float other = __shfl_xor_sync(FULL_MASK, local_max, offset);
        local_max = fmaxf(local_max, other);
      }

      // Update the running maximum
      const float old_max = m[i];
      const float new_max = fmaxf(old_max, local_max);
      float alpha;
      if (old_max == -INFINITY) {
        alpha = 0.0f;
      } else {
        alpha = expf(old_max - new_max);
      }

      // Sum unnormalized attention weights
      float local_sum = 0.0f;
      for (int j = 0; j < COLS_PER_THREAD; j++) {
        float p;
        if (scores[i][j] == -INFINITY) {
          p = 0.0f;
        } else {
          p = expf(scores[i][j] - new_max);
        }
        p_reg[i][j] = p;
        local_sum += p;
      }

      // Sum attention weights across the query's lanes
      for (int offset = LANES_PER_ROW / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_xor_sync(FULL_MASK, local_sum, offset);
      }

      // Update online softmax state based on the new maximum
      l[i] = alpha * l[i] + local_sum;
      m[i] = new_max;
      for (int f = 0; f < FEATURES_PER_THREAD; f++) {
        acc[i][f] *= alpha;
      }
    }

    // Compute P_tile @ V_tile with the unnormalized attention weights
    for (int owner_lane = 0; owner_lane < LANES_PER_ROW; owner_lane++) {
      const int source_lane = row_group * LANES_PER_ROW + owner_lane;

      for (int j = 0; j < COLS_PER_THREAD; j++) {
        const int c = owner_lane * COLS_PER_THREAD + j;

        for (int i = 0; i < ROWS_PER_THREAD; i++) {
          // Broadcast the attention weight from the lane that owns it
          const float p = __shfl_sync(FULL_MASK, p_reg[i][j], source_lane);
          const int q_local = warp_id * ROWS_PER_WARP + row_base + i;
          const size_t query_row = q_start + q_local;

          if (q_local >= BR || query_row >= N)
            continue;

          // Accumulate weighted V values for this thread's output features
          for (int f = 0; f < FEATURES_PER_THREAD; f++) {
            const int feature = col_group * FEATURES_PER_THREAD + f;
            if (feature < D) {
              acc[i][f] = fmaf(p, (float)V_shared[c][feature], acc[i][f]);
            }
          }
        }
      }
    }
    __syncthreads();
  }

  // Write final attention outputs and log-sum-exp
  for (int i = 0; i < ROWS_PER_THREAD; i++) {
    const int q_local = warp_id * ROWS_PER_WARP + row_base + i;
    const size_t query_row = q_start + q_local;

    if (q_local >= BR || query_row >= N) {
      continue;
    }

    for (int f = 0; f < FEATURES_PER_THREAD; f++) {
      const int feature = col_group * FEATURES_PER_THREAD + f;

      // Normalize only after all K/V tiles have been accumulated over
      if (feature < D) {
        if (l[i] > 0.0f) {
          out[base + query_row * D + feature] = (scalar_t)(acc[i][f] / l[i]);
        } else {
          out[base + query_row * D + feature] = scalar_t(0);
        }
      }
    }

    // One lane writes LSE per query row
    if (col_group == 0) {
      if (l[i] > 0.0f) {
        lse[BatchHeadIdx * N + query_row] = m[i] + logf(l[i]);
      } else {
        lse[BatchHeadIdx * N + query_row] = -INFINITY;
      }
    }
  }
}

void FlashAttentionForward_MatMul(
    const CudaArray& q, const CudaArray& k, const CudaArray& v,
    CudaArray* out, CudaArray* lse,
    uint32_t B, uint32_t H, uint32_t N, uint32_t D, bool causal) {

  dim3 grid((N + BR - 1) / BR, B * H);
  dim3 block(NUM_THREADS);

  FlashAttentionForwardKernel_MatMul<<<grid, block>>>(
      q.ptr, k.ptr, v.ptr, out->ptr, lse->ptr,
      B, H, N, D, causal);
}

template <int BR, int BC, int TM, int TN, int NUM_THREADS, bool USE_FMA>
__global__ void FlashAttentionBackwardKernel_MatMul(
    const scalar_t* q, const scalar_t* k, const scalar_t* v,
    const scalar_t* o, const scalar_t* lse, const scalar_t* dO,
    scalar_t* dq, scalar_t* dk, scalar_t* dv,
    const scalar_t* delta,
    size_t B, size_t H, size_t N, size_t D, bool causal) {

  // Each thread computes a 4x4 matrix tile

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  static_assert(NUM_THREADS % WARP_SIZE == 0);
  static_assert(BR % NUM_WARPS == 0);
  static_assert(BC % NUM_WARPS == 0);
  constexpr int GEMM_THREADS = (BR / TM) * (BC / TN);

  // Ensure the matrix dimensions are compatible with tiling
  static_assert(BR % TM == 0, "BR must be divisible by TM");
  static_assert(BC % TN == 0, "BC must be divisible by TN");
  static_assert(BC % TM == 0, "BC must be divisible by TM");
  static_assert(D_CONST % TN == 0, "D_CONST must be divisible by TN");

  // Ensure enough threads exist to compute all S and dP tiles
  static_assert(GEMM_THREADS <= NUM_THREADS,
                "Not enough threads for S and dP tiles");

  constexpr int GRAD_COL_TILES = D_CONST / TN;
  constexpr int GRAD_ROW_TILES = BC / TM;
  constexpr int GRAD_TILES = GRAD_ROW_TILES * GRAD_COL_TILES;

  constexpr int DQ_ROW_TILES = BR / TM;
  constexpr int DQ_TILES = DQ_ROW_TILES * GRAD_COL_TILES;

  // Maximum number of dK/dV tiles assigned to each thread
  constexpr int GRAD_OWNED =
      (GRAD_TILES + NUM_THREADS - 1) / NUM_THREADS;

  constexpr int ROWS_PER_WARP = BC / NUM_WARPS;
  constexpr int FEATURES_PER_ROW = (D_CONST + WARP_SIZE - 1) / WARP_SIZE;
  constexpr int FEATURES_PER_THREAD = ROWS_PER_WARP * FEATURES_PER_ROW;
  size_t tileIdx = blockIdx.x;
  size_t BatchHeadIdx = blockIdx.y;
  size_t BatchIdx = BatchHeadIdx / H;
  size_t HeadIdx = BatchHeadIdx % H;
  size_t tid = threadIdx.x;
  size_t warp_id = tid / WARP_SIZE;
  size_t lane_id = tid % WARP_SIZE;
  size_t base = BatchHeadIdx * N * D_CONST;
  size_t Tr = (N + BR - 1) / BR;
  size_t Tc = (N + BC - 1) / BC;
  size_t kv_start = tileIdx * BC;

  const float scale = rsqrtf((float)D_CONST);

  // Load K and V tiles into shared memory
  __shared__ scalar_t Kj[BC][D_CONST];
  __shared__ scalar_t Vj[BC][D_CONST];
  
  // Each thread accumulates its assigned dK and dV tiles
  // Accumulators persist across all query tiles
  float dKj[GRAD_OWNED][TM][TN] = {};
  float dVj[GRAD_OWNED][TM][TN] = {};
  
  for (size_t r = 0; r < ROWS_PER_WARP; r++) {
    size_t kv_row = kv_start + warp_id * ROWS_PER_WARP + r;
    size_t shared_row = warp_id * ROWS_PER_WARP + r;

    for (size_t f = 0; f < FEATURES_PER_ROW; f++) {
      size_t kv_feature = f * WARP_SIZE + lane_id;

      if (kv_feature < D_CONST && kv_row < N) {
        Kj[shared_row][kv_feature] =
            k[base + kv_row * D_CONST + kv_feature];
        Vj[shared_row][kv_feature] =
            v[base + kv_row * D_CONST + kv_feature];
      } else if (kv_feature < D_CONST) {
        Kj[shared_row][kv_feature] = 0.0f;
        Vj[shared_row][kv_feature] = 0.0f;
      }

    }
  }
  __syncthreads();

  __shared__ scalar_t Qi[BR][D_CONST];
  __shared__ scalar_t dOi[BR][D_CONST];
  __shared__ scalar_t Li[BR];
  __shared__ scalar_t Di[BR];

  __shared__ float Pi[BR][BC];
  __shared__ float dSi[BR][BC];

  size_t BR_ROWS_PER_WARP = BR / NUM_WARPS;

  size_t first_q_tile; 
  if (causal) {
    first_q_tile = kv_start / BR;
  } else { 
    first_q_tile = 0;
  }

  // Load Li, Di, Qi, dOi
  for (size_t i = first_q_tile; i < Tr; i++) {
    for (size_t r = 0; r < BR_ROWS_PER_WARP; r++) {
      size_t shared_row = warp_id * BR_ROWS_PER_WARP + r;
      size_t global_row = i * BR + warp_id * BR_ROWS_PER_WARP + r;

      if (lane_id == 0) {
        if (global_row < N) {
          Li[shared_row] = lse[BatchHeadIdx * N + global_row];
          Di[shared_row] = delta[BatchHeadIdx * N + global_row];
        } else {
          Li[shared_row] = 0.0f;
          Di[shared_row] = 0.0f;
        }
      }

      for (size_t f = 0; f < FEATURES_PER_ROW; f++) {
        size_t feature = f * WARP_SIZE + lane_id;

        if (global_row < N && feature < D_CONST) {
          Qi[shared_row][feature] = q[base + global_row * D_CONST + feature];
          dOi[shared_row][feature] = dO[base + global_row * D_CONST + feature];
        } else {
          Qi[shared_row][feature] = 0.0f;
          dOi[shared_row][feature] = 0.0f;
        }
      }
    }
    __syncthreads();

    // Compute S = Q @ K^T and dP = dO @ V^T
    if (tid < GEMM_THREADS) {

      // Starting row and column of the output tile this thread owns
      int tile_row = tid / (BC / TN);
      int tile_col = tid % (BC / TN);
      int row_base = tile_row * TM;
      int col_base = tile_col * TN;

      // Accumulate attention scores and dP in registers
      float scores[TM][TN] = {};
      float dP[TM][TN] = {};

      for (int f = 0; f < D_CONST; f++) {
        // Load query and key values
        float qr[TM], dOr[TM];
        float kr[TN], vr[TN];
        for (int r = 0; r < TM; r++) {
          qr[r] = Qi[row_base + r][f];
          dOr[r] = dOi[row_base + r][f];
        }
        for (int c = 0; c < TN; c++) {
          kr[c] = Kj[col_base + c][f];
          vr[c] = Vj[col_base + c][f];
        }

        // Update S and dP
        for (int r = 0; r < TM; r++) {
          for (int c = 0; c < TN; c++) {
            scores[r][c] = USE_FMA ? fmaf(qr[r], kr[c], scores[r][c]) : scores[r][c] + qr[r] * kr[c];
            dP[r][c] = USE_FMA ? fmaf(dOr[r], vr[c], dP[r][c]) : dP[r][c] + dOr[r] * vr[c];
          }
        }
      }

      // Convert S and dP into P and dS
      for (int r = 0; r < TM; r++) {
        int qr = row_base + r;
        size_t query_row = i * BR + qr;

        for (int c = 0; c < TN; c++) {
          int j = col_base + c;
          size_t kv_row = kv_start + j;

          bool valid = query_row < N && kv_row < N;
          bool causal_diag = causal && (i * BR < kv_start + BC - 1);
          if (causal_diag) {
            valid = valid && (kv_row <= query_row);
          }

          float P = 0.0f;
          float dS = 0.0f;

          if (valid) {
            float score = scores[r][c] * scale;
            P = expf(score - Li[qr]);

            dS = P * (dP[r][c] - Di[qr]);
          }

          // Store results for the gradient matrix multiplications
          Pi[qr][j] = P;
          dSi[qr][j] = dS;
        }
      }
    }
    __syncthreads();

    // dK = dS^T @ Q and dV = P^T @ dO
    for (int tile = tid; tile < GRAD_TILES; tile += blockDim.x) {
      int tile_row = tile / GRAD_COL_TILES;
      int tile_col = tile % GRAD_COL_TILES;
      int row_base = tile_row * TM;
      int col_base = tile_col * TN;

      float dk_tile[TM][TN] = {};
      float dv_tile[TM][TN] = {};

      for (int qr = 0; qr < BR; qr++) {
        float dS[TM], P[TM];
        float qv[TN], dOv[TN];

        for (int r = 0; r < TM; r++) {
          dS[r] = dSi[qr][row_base + r];
          P[r] = Pi[qr][row_base + r];
        }

        for (int c = 0; c < TN; c++) {
          qv[c] = Qi[qr][col_base + c];
          dOv[c] = dOi[qr][col_base + c];
        }

        // Update dK and dV
        for (int r = 0; r < TM; r++) {
          for (int c = 0; c < TN; c++) {
            dk_tile[r][c] = USE_FMA ? fmaf(dS[r], qv[c], dk_tile[r][c]) : dk_tile[r][c] + dS[r] * qv[c];
            dv_tile[r][c] = USE_FMA ? fmaf(P[r], dOv[c], dv_tile[r][c]) : dv_tile[r][c] + P[r] * dOv[c];
          }
        }
      }

      int ind = tile / blockDim.x;

      // Accumulate gradients across all query tiles
      for (int r = 0; r < TM; r++) {
        for (int c = 0; c < TN; c++) {
          dKj[ind][r][c] += dk_tile[r][c] * scale;
          dVj[ind][r][c] += dv_tile[r][c];
        }
      }
    }

    // dQ = dS @ K
    for (int tile = tid; tile < DQ_TILES; tile += blockDim.x) {
      int tile_row = tile / GRAD_COL_TILES;
      int tile_col = tile % GRAD_COL_TILES;
      int row_base = tile_row * TM;
      int col_base = tile_col * TN;

      float dq_tile[TM][TN] = {};

      for (int j = 0; j < BC; j++) {
        float dS[TM];
        float kv[TN];

        for (int r = 0; r < TM; r++) {
          dS[r] = dSi[row_base + r][j];
        }

        for (int c = 0; c < TN; c++) {
          kv[c] = Kj[j][col_base + c];
        }

        // Update dQ
        for (int r = 0; r < TM; r++) {
          for (int c = 0; c < TN; c++) {
            dq_tile[r][c] = USE_FMA ? fmaf(dS[r], kv[c], dq_tile[r][c]) : dq_tile[r][c] + dS[r] * kv[c];
          }
        }
      }

      // Accumulate dQ across K/V tiles
      for (int r = 0; r < TM; r++) {
        size_t query_row = i * BR + row_base + r;

        if (query_row >= N) continue;

        if (causal && query_row < kv_start) continue;

        for (int c = 0; c < TN; c++) {
          int feature = col_base + c;
          size_t idx = base + query_row * D_CONST + feature;

          atomicAdd(&dq[idx], dq_tile[r][c] * scale);
        }
      }
    }
    __syncthreads();
  }

  // Store dK and dV
  for (int tile = tid; tile < GRAD_TILES; tile += blockDim.x) {
    int tile_row = tile / GRAD_COL_TILES;
    int tile_col = tile % GRAD_COL_TILES;
    int row_base = tile_row * TM;
    int col_base = tile_col * TN;

    // Index of this tile in the thread's accumulators
    int ind = tile / blockDim.x;
      
    // Write the accumulated dK and dV tiles
    for (int r = 0; r < TM; r++) {
      size_t kv_row = kv_start + row_base + r;

      if (kv_row >= N) continue;

      for (int c = 0; c < TN; c++) {
        int feature = col_base + c;
        size_t idx = base + kv_row * D_CONST + feature;

        dk[idx] = dKj[ind][r][c];
        dv[idx] = dVj[ind][r][c];
      }
    }
  }
}



template <int BR, int BC, int TM, int TN, int THREADS, bool USE_FMA>
void LaunchFlashAttentionBackward_MatMul(
    const CudaArray& q, const CudaArray& k, const CudaArray& v,
    const CudaArray& o, const CudaArray& lse, const CudaArray& dO,
    CudaArray* dq, CudaArray* dk, CudaArray* dv,
    const CudaArray& delta,
    uint32_t B, uint32_t H, uint32_t N, uint32_t D,
    bool causal) {

  dim3 grid((N + BC - 1) / BC, B * H);
  FlashAttentionBackwardKernel_MatMul<
      BR, BC, TM, TN, THREADS, USE_FMA
  ><<<grid, THREADS>>>(
      q.ptr, k.ptr, v.ptr, o.ptr, lse.ptr,
      dO.ptr, dq->ptr, dk->ptr, dv->ptr,
      delta.ptr, B, H, N, D, causal);
}

void FlashAttentionBackward_MatMul(
    const CudaArray& q, const CudaArray& k, const CudaArray& v,
    const CudaArray& o, const CudaArray& lse, const CudaArray& dO,
    CudaArray* dq, CudaArray* dk, CudaArray* dv,
    uint32_t B, uint32_t H, uint32_t N, uint32_t D,
    bool causal,
    int br, int bc, int tm, int tn,
    int num_threads, bool use_fma) {

  if (D != D_CONST) {
    throw std::runtime_error("FA2 tiled backward requires D=64");
  }

  // Delta remains on the original configuration.
  dim3 delta_grid((N + BC - 1) / BC, B * H);
  dim3 delta_block(NUM_THREADS);
  CudaArray delta((size_t)B * H * N);

  InitDeltaMatrixKernel<<<delta_grid, delta_block>>>(
      dO.ptr, o.ptr, delta.ptr, B, H, N, D);

  cudaMemset(dq->ptr, 0,
             (size_t)B * H * N * D_CONST * sizeof(scalar_t));

  #define LAUNCH(R, C, M, T, THREADS)                          \
    if (use_fma) {                                             \
      LaunchFlashAttentionBackward_MatMul<                     \
          R, C, M, T, THREADS, true>(                          \
          q, k, v, o, lse, dO, dq, dk, dv, delta,              \
          B, H, N, D, causal);                                 \
    } else {                                                   \
      LaunchFlashAttentionBackward_MatMul<                     \
          R, C, M, T, THREADS, false>(                         \
          q, k, v, o, lse, dO, dq, dk, dv, delta,              \
          B, H, N, D, causal);                                 \
    }

  if (br == 32 && bc == 32 && tm == 4 && tn == 4 && num_threads == 128) {
    LAUNCH(32, 32, 4, 4, 128);
  } else if (br == 16 && bc == 32 && tm == 4 && tn == 4 && num_threads == 128) {
    LAUNCH(16, 32, 4, 4, 128);
  } else if (br == 32 && bc == 32 && tm == 2 && tn == 4 && num_threads == 128) {
    LAUNCH(32, 32, 2, 4, 128);
  } else if (br == 32 && bc == 32 && tm == 4 && tn == 2 && num_threads == 128) {
    LAUNCH(32, 32, 4, 2, 128);
  } else if (br == 32 && bc == 32 && tm == 8 && tn == 4 && num_threads == 128) {
    LAUNCH(32, 32, 8, 4, 128);
  } else if (br == 32 && bc == 32 && tm == 4 && tn == 8 && num_threads == 128) {
    LAUNCH(32, 32, 4, 8, 128);
  } else if (br == 32 && bc == 32 && tm == 4 && tn == 4 && num_threads == 64) {
    LAUNCH(32, 32, 4, 4, 64);
  } else if (br == 32 && bc == 32 && tm == 4 && tn == 4 && num_threads == 256) {
    LAUNCH(32, 32, 4, 4, 256);
  } else {
    throw std::runtime_error("Unsupported backward configuration");
  }

  #undef LAUNCH
}
}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);


  // Optimized implementations
  m.def("flash_attention_forward",
        FlashAttentionForward_MatMul);
  m.def(
      "flash_attention_backward",
      FlashAttentionBackward_MatMul,
      py::arg("q"), py::arg("k"), py::arg("v"),
      py::arg("o"), py::arg("lse"), py::arg("dO"),
      py::arg("dq"), py::arg("dk"), py::arg("dv"),
      py::arg("B"), py::arg("H"), py::arg("N"), py::arg("D"),
      py::arg("causal"),
      py::arg("br") = 32,
      py::arg("bc") = 32,
      py::arg("tm") = 4,
      py::arg("tn") = 4,
      py::arg("num_threads") = 128,
      py::arg("use_fma") = false);

  // Original baseline implementations
  m.def("flash_attention_forward_baseline",
        FlashAttentionForward_Baseline);
  m.def("flash_attention_backward_baseline",
        FlashAttentionBackward_Baseline);

}
