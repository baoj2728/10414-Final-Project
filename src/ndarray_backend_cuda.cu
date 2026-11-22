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

void FlashAttentionForward(const CudaArray& q, const CudaArray& k, const CudaArray& v, CudaArray* out,
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


void FlashAttentionBackward(const CudaArray& q, const CudaArray& k, const CudaArray& v, const CudaArray& o,
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

  m.def("flash_attention_forward", FlashAttentionForward);
  m.def("flash_attention_backward", FlashAttentionBackward);
}
