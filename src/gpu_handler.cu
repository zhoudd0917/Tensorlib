#include <cmath>
#include <tensorlib/gpu_handler.cuh>
#include <tensorlib/utils.hpp>

GPUHandler& GPUHandler::getInstance() {
  static GPUHandler instance;
  return instance;
}

void GPUHandler::add(const float* x, const float* y, float* z, size_t size) {
  cublasHandle_t handle = getInstance().getHandle();
  float alpha = 1.0;
  checkCudaErrors(
      cudaMemcpy(z, y, size * sizeof(float), cudaMemcpyDeviceToDevice));
  checkCublasErrors(cublasSaxpy(handle, size, &alpha, x, 1, z, 1));
}

void GPUHandler::sub(const float* x, const float* y, float* z, size_t size) {
  cublasHandle_t handle = getInstance().getHandle();
  float alpha = -1.0f;

  checkCudaErrors(
      cudaMemcpy(z, x, size * sizeof(float), cudaMemcpyDeviceToDevice));
  checkCublasErrors(cublasSaxpy(handle, size, &alpha, y, 1, z, 1));
}

void GPUHandler::multiply(const float* x, const float* y, float* z,
                          size_t size) {
  cublasHandle_t handle = getInstance().getHandle();
  cublasSideMode_t mode = CUBLAS_SIDE_LEFT;

  int m = size, n = 1, lda = size, incx = 1, ldc = size;

  // Perform element-wise multiplication: z = diag(y) * x
  checkCublasErrors(cublasSdgmm(handle, mode, m, n, x, lda, y, incx, z, ldc));
}

__global__ void elementWiseDivision(const float* x, const float* y, float* z,
                                    size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    z[idx] = x[idx] / y[idx];  // Perform element-wise division
  }
}

void GPUHandler::divide(const float* x, const float* y, float* z, size_t size) {
  // Each thread processes one element; set the CUDA grid size
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  // Launch the CUDA kernel to perform element-wise division
  elementWiseDivision<<<gridSize, blockSize>>>(x, y, z, size);

  // Check for CUDA errors
  checkCudaErrors(cudaDeviceSynchronize());
}

// helper method for SubBackward
void GPUHandler::axpy(const float* x, float* y, float alpha, size_t size) {
  cublasHandle_t handle = getInstance().getHandle();
  checkCublasErrors(cublasSaxpy(handle, size, &alpha, x, 1, y, 1));
}

// matrix multiplication
void GPUHandler::matmul(const float* X, const float* Y, float* Z, size_t B,
                        size_t M, size_t K, size_t N, bool transX,
                        bool transY) {
  cublasHandle_t handle = getInstance().getHandle();

  auto trans_x = transX ? CUBLAS_OP_T : CUBLAS_OP_N,
       trans_y = transY ? CUBLAS_OP_T : CUBLAS_OP_N;

  // Perform matrix multiplication for each batch
  size_t stride_X = M * K, stride_Y = K * N, stride_Z = M * N;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Call cuBLAS gemm function for batch computation
  checkCublasErrors(cublasSgemmStridedBatched(
      handle, trans_y, trans_x, N, M, K, &alpha, Y, (transY ? K : N), stride_Y,
      X, (transX ? M : K), stride_X, &beta, Z, N, stride_Z, B));
}

void GPUHandler::transpose(const float* input, float* output, size_t B,
                           size_t M, size_t N) {
  cublasHandle_t handle = getInstance().getHandle();
  const float alpha = 1.0f;
  const float beta = 0.0f;

  for (size_t b = 0; b < B; ++b) {
    checkCublasErrors(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N,
                                  &alpha, input + b * M * N, N, &beta,
                                  input + b * M * N, M, output + b * M * N, M));
  }
}

__global__ void elementWiseLog(const float* input, float* output, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = logf(input[idx]);  // Compute natural logarithm (base e)
  }
}

void GPUHandler::log(const float* input, float* output, size_t size) {
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  elementWiseLog<<<gridSize, blockSize>>>(input, output, size);

  checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void selectIdx(float* X, float* Z, size_t size, size_t idx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    Z[i] = X[i + idx * size];
  }
}

void GPUHandler::select_idx(float* X, float* Z, std::vector<size_t> x_shape,
                            size_t idx) {
  size_t size = calculate_size(x_shape) / x_shape[0];

  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;
  selectIdx<<<gridSize, blockSize>>>(X, Z, size, idx);

  checkCudaErrors(cudaDeviceSynchronize());
}

__device__ size_t d_calculate_index_after_drop_axis(size_t index, size_t axis,
                                                    const size_t* shape,
                                                    size_t nDims) {
  size_t newIndex = 0;
  size_t stride = 1;

  for (int i = nDims; i > 0; i--) {
    if (i - 1 != axis) {
      size_t dimSize = shape[i - 1];
      size_t currentDimIndex = index % dimSize;
      newIndex += currentDimIndex * stride;
      stride *= dimSize;
    }
    index /= shape[i - 1];
  }
  return newIndex;
}

__global__ void sumAlongAxisKernel(float* X, float* Z, const size_t* shape,
                                   size_t axis, size_t input_size,
                                   size_t nDims) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < input_size) {
    size_t output_idx =
        d_calculate_index_after_drop_axis(i, axis, shape, nDims);
    atomicAdd(&Z[output_idx], X[i]);
  }
}

void GPUHandler::sum(float* X, float* Z, std::vector<size_t> x_shape,
                     size_t axis) {
  size_t input_size = calculate_size(x_shape);
  size_t output_size = input_size / x_shape[axis];

  size_t* d_x_shape;
  cudaMalloc(&d_x_shape, x_shape.size() * sizeof(size_t));
  cudaMemcpy(d_x_shape, x_shape.data(), x_shape.size() * sizeof(size_t),
             cudaMemcpyHostToDevice);

  int blockSize = 256;
  int numBlocks = (input_size + blockSize - 1) / blockSize;
  cudaMemset(Z, 0, output_size * sizeof(float));
  sumAlongAxisKernel<<<numBlocks, blockSize>>>(X, Z, d_x_shape, axis,
                                               input_size, output_size);

  cudaFree(d_x_shape);
}

__device__ size_t d_calculate_index_after_add_axis(size_t index, size_t axis,
                                                   const size_t* shape,
                                                   size_t nDims) {
  size_t new_index = 0;
  size_t old_stride = 1, new_stride = 1;

  for (int i = nDims; i > 0; i--) {
    size_t dim_size = shape[i - 1];
    if (i - 1 != axis) {
      size_t c_i = index % dim_size;
      new_index += c_i * new_stride;
      new_stride *= dim_size;
      old_stride *= dim_size;
      index /= dim_size;
    } else {
      new_stride *= dim_size;
    }
  }
  return new_index;
}

__global__ void addAxisKernel(float* x_grad, const float* output_grad,
                              const size_t* x_shape, const size_t* x_stride,
                              size_t axis, size_t axis_size, float factor,
                              size_t size, size_t nDims) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size) {
    size_t input_idx =
        d_calculate_index_after_add_axis(i, axis, x_shape, nDims);

    for (size_t j = 0; j < axis_size; j++) {
      x_grad[input_idx] += output_grad[i] * factor;
      input_idx += x_stride[axis];
    }
  }
}

void GPUHandler::add_axis(float* x_grad, const float* output_grad,
                          std::vector<size_t> x_shape,
                          std::vector<size_t> x_stride, size_t axis,
                          size_t axis_size, float factor) {
  size_t size = calculate_size(x_shape);
  size_t nDims = x_shape.size();

  size_t *d_x_shape, *d_x_stride;
  cudaMalloc(&d_x_shape, nDims * sizeof(size_t));
  cudaMalloc(&d_x_stride, nDims * sizeof(size_t));

  cudaMemcpy(d_x_shape, x_shape.data(), nDims * sizeof(size_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_x_stride, x_stride.data(), nDims * sizeof(size_t),
             cudaMemcpyHostToDevice);

  cudaMemset(x_grad, 0, size * sizeof(float));

  int blockSize = 256;
  int numBlocks = (size + blockSize - 1) / blockSize;
  addAxisKernel<<<numBlocks, blockSize>>>(x_grad, output_grad, d_x_shape,
                                          d_x_stride, axis, axis_size, factor,
                                          size, nDims);

  cudaFree(d_x_shape);
  cudaFree(d_x_stride);
}

// exp
__global__ void elementWiseExp(const float* input, float* output, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = expf(input[idx]);
  }
}

void GPUHandler::exp(const float* input, float* output, size_t size) {
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  elementWiseExp<<<gridSize, blockSize>>>(input, output, size);

  checkCudaErrors(cudaDeviceSynchronize());
}

// sin
__global__ void elementWiseSin(const float* input, float* output, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = sinf(input[idx]);  // Compute sine
  }
}

void GPUHandler::sin(const float* input, float* output, size_t size) {
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  elementWiseSin<<<gridSize, blockSize>>>(input, output, size);

  checkCudaErrors(cudaDeviceSynchronize());
}

// cos
__global__ void elementWiseCos(const float* input, float* output, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = cosf(input[idx]);  // Compute cosine
  }
}

void GPUHandler::cos(const float* input, float* output, size_t size) {
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  elementWiseCos<<<gridSize, blockSize>>>(input, output, size);

  checkCudaErrors(cudaDeviceSynchronize());
}

// relu
__global__ void elementWiseReLU(const float* input, float* output,
                                size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = fmaxf(0.0f, input[idx]);  // Compute ReLU
  }
}

void GPUHandler::relu(const float* input, float* output, size_t size) {
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  elementWiseReLU<<<gridSize, blockSize>>>(input, output, size);

  checkCudaErrors(cudaDeviceSynchronize());
}

void GPUHandler::reshape(const float* input, float* output, size_t size) {
  checkCudaErrors(cudaMemcpy(output, input, size * sizeof(float),
                             cudaMemcpyDeviceToDevice));

  // Synchronize to ensure the operation is complete
  checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void logBackwardKernel(const float* output_grad, const float* x_data,
                                  float* x_grad, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    x_grad[idx] += output_grad[idx] / x_data[idx];  // Gradient of log(x)
  }
}

void GPUHandler::logBackward(const float* output_grad, const float* x_data,
                             float* x_grad, size_t size) {
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  logBackwardKernel<<<gridSize, blockSize>>>(output_grad, x_data, x_grad, size);

  checkCudaErrors(cudaDeviceSynchronize());
}

// exp back
__global__ void expBackwardKernel(const float* output_grad, const float* x_data, float* x_grad, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x_grad[idx] += output_grad[idx] * expf(x_data[idx]);  // Gradient computation
    }
}

void GPUHandler::expBackward(const float* output_grad, const float* x_data, float* x_grad, size_t size) {
    int blockSize = 256;  // Number of threads per block
    int gridSize = (size + blockSize - 1) / blockSize;  // Number of blocks

    // Launch the kernel
    expBackwardKernel<<<gridSize, blockSize>>>(output_grad, x_data, x_grad, size);

    // Synchronize and check for CUDA errors
    checkCudaErrors(cudaDeviceSynchronize());
}

// sin back
__global__ void sinBackwardKernel(const float* output_grad, const float* x_data, float* x_grad, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x_grad[idx] += output_grad[idx] * cosf(x_data[idx]);  // Gradient computation
    }
}

void GPUHandler::sinBackward(const float* output_grad, const float* x_data, float* x_grad, size_t size) {
    int blockSize = 256;  // Number of threads per block
    int gridSize = (size + blockSize - 1) / blockSize;  // Number of blocks

    // Launch the kernel
    sinBackwardKernel<<<gridSize, blockSize>>>(output_grad, x_data, x_grad, size);

    // Synchronize and check for CUDA errors
    checkCudaErrors(cudaDeviceSynchronize());
}

// cos back
__global__ void cosBackwardKernel(const float* output_grad, const float* x_data, float* x_grad, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x_grad[idx] -= output_grad[idx] * sinf(x_data[idx]);  // Gradient computation
    }
}

void GPUHandler::cosBackward(const float* output_grad, const float* x_data, float* x_grad, size_t size) {
    int blockSize = 256;  // Number of threads per block
    int gridSize = (size + blockSize - 1) / blockSize;  // Number of blocks

    // Launch the kernel
    cosBackwardKernel<<<gridSize, blockSize>>>(output_grad, x_data, x_grad, size);

    // Synchronize and check for CUDA errors
    checkCudaErrors(cudaDeviceSynchronize());
}

// mean
__global__ void meanKernel(const float* input, float* output, size_t axis_size, size_t inner_dim, size_t outer_dim) {
    int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;  // Outer loop index
    if (outer_idx >= outer_dim) return;

    float sum = 0.0f;
    for (size_t i = 0; i < axis_size; ++i) {
        sum += input[outer_idx * axis_size + i];
    }
    output[outer_idx] = sum / axis_size;  // Compute the mean
}

void GPUHandler::mean(const float* input, float* output, const std::vector<size_t>& shape, size_t axis) {
    size_t axis_size = shape[axis];
    size_t inner_dim = 1;
    size_t outer_dim = 1;

    // Compute dimensions
    for (size_t i = 0; i < axis; ++i) outer_dim *= shape[i];
    for (size_t i = axis + 1; i < shape.size(); ++i) inner_dim *= shape[i];

    int blockSize = 256;  // Threads per block
    int gridSize = (outer_dim + blockSize - 1) / blockSize;

    // Launch kernel
    meanKernel<<<gridSize, blockSize>>>(input, output, axis_size, inner_dim, outer_dim);

    // Synchronize and check for errors
    checkCudaErrors(cudaDeviceSynchronize());
}

// max
__global__ void maxKernelWithIndices(const float* input, float* output, size_t* indices,
                                     size_t axis_size, size_t inner_dim, size_t outer_dim) {
    int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;  // Outer loop index
    if (outer_idx >= outer_dim) return;

    float max_val = -FLT_MAX;  // Initialize to the smallest possible value
    size_t max_idx = 0;

    for (size_t i = 0; i < axis_size; ++i) {
        float val = input[outer_idx * axis_size + i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    output[outer_idx] = max_val;      // Store the maximum value
    indices[outer_idx] = max_idx;    // Store the index of the maximum value
}

size_t* GPUHandler::max(const float* input, float* output, const std::vector<size_t>& shape, size_t axis) {
    size_t axis_size = shape[axis];
    size_t inner_dim = 1;
    size_t outer_dim = 1;

    // Compute dimensions
    for (size_t i = 0; i < axis; ++i) outer_dim *= shape[i];
    for (size_t i = axis + 1; i < shape.size(); ++i) inner_dim *= shape[i];

    int blockSize = 256;  // Threads per block
    int gridSize = (outer_dim + blockSize - 1) / blockSize;

    // Allocate GPU memory for indices
    size_t* d_indices;
    checkCudaErrors(cudaMalloc(&d_indices, outer_dim * sizeof(size_t)));

    // Launch the kernel
    maxKernelWithIndices<<<gridSize, blockSize>>>(input, output, d_indices, axis_size, inner_dim, outer_dim);

    // Synchronize and check for errors
    checkCudaErrors(cudaDeviceSynchronize());

    // Allocate host memory for indices and copy them back to host
    size_t* h_indices = new size_t[outer_dim];
    checkCudaErrors(cudaMemcpy(h_indices, d_indices, outer_dim * sizeof(size_t), cudaMemcpyDeviceToHost));

    // Free GPU memory
    checkCudaErrors(cudaFree(d_indices));

    return h_indices;  // Return the indices to the caller
}


// min
__global__ void minKernelWithIndices(const float* input, float* output, size_t* indices,
                                     size_t axis_size, size_t inner_dim, size_t outer_dim) {
    int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;  // Outer loop index
    if (outer_idx >= outer_dim) return;

    float min_val = FLT_MAX;  // Initialize to the largest possible value
    size_t min_idx = 0;

    for (size_t i = 0; i < axis_size; ++i) {
        float val = input[outer_idx * axis_size + i];
        if (val < min_val) {
            min_val = val;
            min_idx = i;
        }
    }

    output[outer_idx] = min_val;      // Store the minimum value
    indices[outer_idx] = min_idx;    // Store the index of the minimum value
}

size_t* GPUHandler::min(const float* input, float* output, const std::vector<size_t>& shape, size_t axis) {
    size_t axis_size = shape[axis];
    size_t inner_dim = 1;
    size_t outer_dim = 1;

    // Compute dimensions
    for (size_t i = 0; i < axis; ++i) outer_dim *= shape[i];
    for (size_t i = axis + 1; i < shape.size(); ++i) inner_dim *= shape[i];

    int blockSize = 256;  // Threads per block
    int gridSize = (outer_dim + blockSize - 1) / blockSize;

    // Allocate GPU memory for indices
    size_t* d_indices;
    checkCudaErrors(cudaMalloc(&d_indices, outer_dim * sizeof(size_t)));

    // Launch the kernel
    minKernelWithIndices<<<gridSize, blockSize>>>(input, output, d_indices, axis_size, inner_dim, outer_dim);

    // Synchronize and check for errors
    checkCudaErrors(cudaDeviceSynchronize());

    // Allocate host memory for indices and copy them back to host
    size_t* h_indices = new size_t[outer_dim];
    checkCudaErrors(cudaMemcpy(h_indices, d_indices, outer_dim * sizeof(size_t), cudaMemcpyDeviceToHost));

    // Free GPU memory
    checkCudaErrors(cudaFree(d_indices));

    return h_indices;  // Return the indices to the caller
}

