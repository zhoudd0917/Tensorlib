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