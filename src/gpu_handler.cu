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

void GPUHandler::negate(const float* x, float* y, size_t size) {
  cublasHandle_t handle = getInstance().getHandle();
  float alpha = -1.0f;

  checkCudaErrors(
      cudaMemcpy(y, x, size * sizeof(float), cudaMemcpyDeviceToDevice));
  checkCublasErrors(cublasSscal(handle, size, &alpha, y, 1));
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

__global__ void sum_along_axis(const float* X, float* Z, size_t input_size,
                               size_t output_size, size_t axis_stride,
                               size_t axis_size, float factor = 1.0) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= input_size) return;

  size_t output_idx =
      ((idx / axis_size / axis_stride) * axis_stride + idx % axis_stride);

  atomicAdd(&Z[output_idx], X[idx] * factor);
}

void GPUHandler::sum(float* X, float* Z, std::vector<size_t> x_shape,
                     size_t axis) {
  size_t input_size = calculate_size(x_shape);
  size_t output_size = input_size / x_shape[axis];
  std::vector<size_t> strides = calculate_strides(x_shape);
  size_t axis_stride = strides[axis];
  size_t axis_size = x_shape[axis];

  int threads = 256, blocks = (input_size + threads - 1) / threads;

  sum_along_axis<<<blocks, threads>>>(X, Z, input_size, output_size,
                                      axis_stride, axis_size);
  checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void sumAll(const float* X, float* Z, size_t size,
                       float factor = 1.0) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    atomicAdd(&Z[0], X[idx] * factor);
  }
}

void GPUHandler::sum(float* X, float* Z, size_t size) {
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  sumAll<<<gridSize, blockSize>>>(X, Z, size);

  checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void add_axis_kernel(const float* output_grad, float* x_grad,
                                size_t input_size, size_t output_size,
                                size_t axis_stride, size_t axis_size,
                                float factor) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < input_size) {
    size_t output_idx =
        ((i / axis_size / axis_stride) * axis_stride + i % axis_stride);

    x_grad[i] += output_grad[output_idx] * factor;
  }
}

void GPUHandler::mean(float* X, float* Z, std::vector<size_t> x_shape,
                      size_t axis) {
  size_t input_size = calculate_size(x_shape);
  size_t output_size = input_size / x_shape[axis];
  std::vector<size_t> strides = calculate_strides(x_shape);
  size_t axis_stride = strides[axis];
  size_t axis_size = x_shape[axis];

  int threads = 256, blocks = (input_size + threads - 1) / threads;

  sum_along_axis<<<blocks, threads>>>(X, Z, input_size, output_size,
                                      axis_stride, axis_size, 1.0 / axis_size);

  checkCudaErrors(cudaDeviceSynchronize());
}

void GPUHandler::mean(float* X, float* Z, size_t size) {
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  sumAll<<<gridSize, blockSize>>>(X, Z, size, 1.0 / size);

  checkCudaErrors(cudaDeviceSynchronize());
}

void GPUHandler::add_axis(float* x_grad, const float* output_grad,
                          std::vector<size_t> x_shape,
                          std::vector<size_t> x_stride, size_t axis,
                          size_t axis_size, float factor) {
  size_t input_size = calculate_size(x_shape);
  size_t output_size = input_size / axis_size;
  size_t axis_stride = x_stride[axis];

  int threads = 256, blocks = (output_size + threads - 1) / threads;
  add_axis_kernel<<<blocks, threads>>>(output_grad, x_grad, input_size,
                                       output_size, axis_stride, axis_size,
                                       factor);
}

__global__ void set_all_kernel(float* x, const float* val, float factor,
                               size_t size) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size) {
    x[i] = factor * val[0];
  }
}

void GPUHandler::set_all(float* x, const float* val, float factor,
                         size_t size) {
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  set_all_kernel<<<gridSize, blockSize>>>(x, val, factor, size);

  checkCudaErrors(cudaDeviceSynchronize());
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

// expMul
__global__ void expMulKernel(const float* output_data, float* x_grad,
                             const float* output_grad, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    x_grad[idx] += output_grad[idx] * output_data[idx];
  }
}

// x_grad += output_grad * exp(x_data)
void GPUHandler::expMul(const float* output_data, float* x_grad,
                        const float* output_grad, size_t size) {
  cublasHandle_t handle = getInstance().getHandle();

  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;
  expMulKernel<<<gridSize, blockSize>>>(output_data, x_grad, output_grad, size);

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

// sin backward
__global__ void sinBackwardKernel(const float* output_grad, const float* x_data,
                                  float* x_grad, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    x_grad[idx] += output_grad[idx] * cosf(x_data[idx]);  // Gradient of sin(x)
  }
}

void GPUHandler::sinBackward(const float* output_grad, const float* x_data,
                             float* x_grad, size_t size) {
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  sinBackwardKernel<<<gridSize, blockSize>>>(output_grad, x_data, x_grad, size);

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

__global__ void cosBackwardKernel(const float* output_grad, const float* x_data,
                                  float* x_grad, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    x_grad[idx] -= output_grad[idx] * sinf(x_data[idx]);  // Gradient of cos(x)
  }
}

void GPUHandler::cosBackward(const float* output_grad, const float* x_data,
                             float* x_grad, size_t size) {
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  cosBackwardKernel<<<gridSize, blockSize>>>(output_grad, x_data, x_grad, size);

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

// relu backward
__global__ void reluBackwardKernel(const float* output_grad,
                                   const float* x_data, float* x_grad,
                                   size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    x_grad[idx] += output_grad[idx] * (x_data[idx] > 0 ? 1 : 0);
  }
}

void GPUHandler::reluBackward(const float* output_grad, const float* x_data,
                              float* x_grad, size_t size) {
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  reluBackwardKernel<<<gridSize, blockSize>>>(output_grad, x_data, x_grad,
                                              size);

  checkCudaErrors(cudaDeviceSynchronize());
}

// sigmoid
__global__ void elementWiseSigmoid(const float* input, float* output,
                                   size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = 1.0f / (1.0f + expf(-input[idx]));  // Compute sigmoid
  }
}

void GPUHandler::sigmoid(const float* input, float* output, size_t size) {
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  elementWiseSigmoid<<<gridSize, blockSize>>>(input, output, size);

  checkCudaErrors(cudaDeviceSynchronize());
}

// sigmoid backward
__global__ void sigmoidBackwardKernel(const float* output_grad,
                                      const float* output, float* x_grad,
                                      size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    x_grad[idx] += output_grad[idx] * output[idx] * (1 - output[idx]);
  }
}

void GPUHandler::sigmoidBackward(const float* output_grad, const float* output,
                                 float* x_grad, size_t size) {
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  sigmoidBackwardKernel<<<gridSize, blockSize>>>(output_grad, output, x_grad,
                                                 size);

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

__global__ void broadcast_kernel(const float* X, float* Z,
                                 const size_t* x_shape, const size_t* z_shape,
                                 const size_t* x_strides,
                                 const size_t* z_strides, size_t total_elements,
                                 size_t x_dims, size_t z_dims) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total_elements) return;

  size_t x_index = 0, z_index = i;

  for (size_t dim = 0; dim < z_dims; ++dim) {
    size_t z_coord = z_index / z_strides[dim];
    z_index %= z_strides[dim];

    size_t x_coord = (x_shape[dim] == 1) ? 0 : z_coord;
    x_index += x_coord * x_strides[dim];
  }

  Z[i] = X[x_index];
}

void GPUHandler::broadcast(const float* X, float* Z,
                           const std::vector<size_t>& x_shape,
                           const std::vector<size_t>& z_shape) {
  size_t x_dims = x_shape.size();
  size_t z_dims = z_shape.size();
  size_t total_elements = calculate_size(z_shape);

  std::vector<size_t> x_strides = calculate_strides(x_shape),
                      z_strides = calculate_strides(z_shape);

  size_t *d_x_shape, *d_z_shape, *d_x_strides, *d_z_strides;
  cudaMalloc(&d_x_shape, x_dims * sizeof(size_t));
  cudaMalloc(&d_z_shape, z_dims * sizeof(size_t));
  cudaMalloc(&d_x_strides, x_dims * sizeof(size_t));
  cudaMalloc(&d_z_strides, z_dims * sizeof(size_t));

  cudaMemcpy(d_x_shape, x_shape.data(), x_dims * sizeof(size_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_z_shape, z_shape.data(), z_dims * sizeof(size_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_x_strides, x_strides.data(), x_dims * sizeof(size_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_z_strides, z_strides.data(), z_dims * sizeof(size_t),
             cudaMemcpyHostToDevice);

  size_t block_size = 256;
  size_t grid_size = (total_elements + block_size - 1) / block_size;
  broadcast_kernel<<<grid_size, block_size>>>(X, Z, d_x_shape, d_z_shape,
                                              d_x_strides, d_z_strides,
                                              total_elements, x_dims, z_dims);

  cudaDeviceSynchronize();

  cudaFree(d_x_shape);
  cudaFree(d_z_shape);
  cudaFree(d_x_strides);
  cudaFree(d_z_strides);
}

__global__ void broadcastBackwardKernel(const float* output_grad, float* x_grad,
                                        const size_t* x_shape,
                                        const size_t* z_stride,
                                        const size_t* x_stride, size_t z_size,
                                        size_t ndim) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= z_size) return;

  size_t x_index = 0, z_index = i;

  for (size_t dim = 0; dim < ndim; ++dim) {
    size_t z_coord = z_index / z_stride[dim];
    z_index %= z_stride[dim];

    size_t x_coord = (x_shape[dim] == 1) ? 0 : z_coord;
    x_index += x_coord * x_stride[dim];
  }

  atomicAdd(&x_grad[x_index], output_grad[i]);
}

void GPUHandler::broadcastBackward(const float* output_grad, float* x_grad,
                                   const std::vector<size_t>& x_shape,
                                   const std::vector<size_t>& z_stride,
                                   const std::vector<size_t>& x_stride,
                                   size_t z_size) {
  size_t ndim = x_shape.size();

  size_t *d_x_shape, *d_z_stride, *d_x_stride;
  cudaMalloc(&d_x_shape, ndim * sizeof(size_t));
  cudaMalloc(&d_z_stride, ndim * sizeof(size_t));
  cudaMalloc(&d_x_stride, ndim * sizeof(size_t));

  cudaMemcpy(d_x_shape, x_shape.data(), ndim * sizeof(size_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_z_stride, z_stride.data(), ndim * sizeof(size_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_x_stride, x_stride.data(), ndim * sizeof(size_t),
             cudaMemcpyHostToDevice);

  size_t block_size = 256;
  size_t grid_size = (z_size + block_size - 1) / block_size;

  broadcastBackwardKernel<<<grid_size, block_size>>>(
      output_grad, x_grad, d_x_shape, d_z_stride, d_x_stride, z_size, ndim);

  cudaDeviceSynchronize();

  cudaFree(d_x_shape);
  cudaFree(d_z_stride);
  cudaFree(d_x_stride);
}

__global__ void max_kernel(const float* X, float* Z, size_t* idx_list,
                           const size_t* x_shape, const size_t* x_stride,
                           size_t input_size, size_t output_size, size_t ndim,
                           size_t axis) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= output_size) return;

  Z[i] = -INFINITY;
  idx_list[i] = 0;

  size_t start_idx = (i / x_stride[axis]) * x_stride[axis] * x_shape[axis] +
                     (i % x_stride[axis]);

  for (size_t j = 0; j < x_shape[axis]; ++j) {
    size_t idx = start_idx + j * x_stride[axis];
    if (X[idx] > Z[i]) {
      Z[i] = X[idx];
      idx_list[i] = idx;
    }
  }
}

size_t* GPUHandler::max(const float* X, float* Z, std::vector<size_t> x_shape,
                        size_t axis) {
  size_t input_size = calculate_size(x_shape);
  size_t output_size = input_size / x_shape[axis];
  size_t ndim = x_shape.size();

  size_t* d_x_shape;
  size_t* d_x_stride;

  size_t* idx_list;
  cudaMalloc(&idx_list, output_size * sizeof(size_t));

  cudaMalloc(&d_x_shape, ndim * sizeof(size_t));
  cudaMalloc(&d_x_stride, ndim * sizeof(size_t));

  cudaMemcpy(d_x_shape, x_shape.data(), ndim * sizeof(size_t),
             cudaMemcpyHostToDevice);

  std::vector<size_t> x_stride = calculate_strides(x_shape);
  cudaMemcpy(d_x_stride, x_stride.data(), ndim * sizeof(size_t),
             cudaMemcpyHostToDevice);

  size_t block_size = 256;
  size_t grid_size = (output_size + block_size - 1) / block_size;
  max_kernel<<<grid_size, block_size>>>(X, Z, idx_list, d_x_shape, d_x_stride,
                                        input_size, output_size, ndim, axis);

  cudaFree(d_x_shape);
  cudaFree(d_x_stride);

  return idx_list;
}

size_t* GPUHandler::max(const float* X, float* Z, size_t size) {
  return max(X, Z, std::vector<size_t>{size}, 0);
}

__global__ void min_kernel(const float* X, float* Z, size_t* idx_list,
                           const size_t* x_shape, const size_t* x_stride,
                           size_t input_size, size_t output_size, size_t ndim,
                           size_t axis) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= output_size) return;

  Z[i] = INFINITY;
  idx_list[i] = 0;

  size_t start_idx = (i / x_stride[axis]) * x_stride[axis] * x_shape[axis] +
                     (i % x_stride[axis]);

  for (size_t j = 0; j < x_shape[axis]; ++j) {
    size_t idx = start_idx + j * x_stride[axis];
    if (X[idx] < Z[i]) {
      Z[i] = X[idx];
      idx_list[i] = idx;
    }
  }
}

size_t* GPUHandler::min(const float* X, float* Z, std::vector<size_t> x_shape,
                        size_t axis) {
  size_t input_size = calculate_size(x_shape);
  size_t output_size = input_size / x_shape[axis];
  size_t ndim = x_shape.size();

  size_t* d_x_shape;
  size_t* d_x_stride;

  size_t* idx_list;
  cudaMalloc(&idx_list, output_size * sizeof(size_t));

  cudaMalloc(&d_x_shape, ndim * sizeof(size_t));
  cudaMalloc(&d_x_stride, ndim * sizeof(size_t));

  cudaMemcpy(d_x_shape, x_shape.data(), ndim * sizeof(size_t),
             cudaMemcpyHostToDevice);

  std::vector<size_t> x_stride = calculate_strides(x_shape);
  cudaMemcpy(d_x_stride, x_stride.data(), ndim * sizeof(size_t),
             cudaMemcpyHostToDevice);

  size_t block_size = 256;
  size_t grid_size = (output_size + block_size - 1) / block_size;
  min_kernel<<<grid_size, block_size>>>(X, Z, idx_list, d_x_shape, d_x_stride,
                                        input_size, output_size, ndim, axis);

  cudaFree(d_x_shape);
  cudaFree(d_x_stride);

  return idx_list;
}

size_t* GPUHandler::min(const float* X, float* Z, size_t size) {
  return min(X, Z, std::vector<size_t>{size}, 0);
}

__global__ void update_grad_kernel(size_t* index_list, float* x_grad,
                                   float* output_grad, size_t size) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;

  atomicAdd(&x_grad[index_list[i]], output_grad[i]);
}

void GPUHandler::update_grad_selector(size_t* index_list, float* x_grad,
                                      float* output_grad, size_t size) {
  size_t block_size = 256;
  size_t grid_size = (size + block_size - 1) / block_size;

  update_grad_kernel<<<grid_size, block_size>>>(index_list, x_grad, output_grad,
                                                size);

  cudaDeviceSynchronize();
}

__global__ void softmax_kernel(const float* X, float* Z, size_t axis_size,
                               size_t axis_stride, size_t size_squashed) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size_squashed) {
    size_t offset =
        (idx / axis_stride) * axis_stride * axis_size + (idx % axis_stride);

    float max_val = -INFINITY;
    for (size_t j = 0; j < axis_size; j++) {
      size_t current_idx = offset + j * axis_stride;
      max_val = fmaxf(max_val, X[current_idx]);
    }

    float sum = 0.0f;
    for (size_t j = 0; j < axis_size; j++) {
      size_t current_idx = offset + j * axis_stride;
      sum += expf(X[current_idx] - max_val);
    }

    // Apply the softmax function
    for (size_t j = 0; j < axis_size; j++) {
      size_t current_idx = offset + j * axis_stride;
      Z[current_idx] = expf(X[current_idx] - max_val) / sum;
    }
  }
}

void GPUHandler::softmax(const float* X, float* Z, std::vector<size_t> x_shape,
                         size_t axis) {
  std::vector<size_t> strides = calculate_strides(x_shape);
  size_t ndim = x_shape.size();
  size_t axis_size = x_shape[axis];
  size_t size = calculate_size(x_shape);
  size_t size_squashed = size / axis_size;

  size_t block_size = 256;
  size_t grid_size = (size_squashed + block_size - 1) / block_size;

  softmax_kernel<<<grid_size, block_size>>>(X, Z, axis_size, strides[axis],
                                            size_squashed);

  cudaDeviceSynchronize();
}

__global__ void softmax_backward_kernel(float* x_grad, const float* output_grad,
                                        const float* z_data, size_t axis_size,
                                        size_t size_squashed,
                                        size_t axis_stride) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size_squashed) {
    size_t offset =
        (idx / axis_stride) * axis_stride * axis_size + idx % axis_stride;

    for (size_t j = 0; j < axis_size; j++) {
      for (size_t i = 0; i < axis_size; i++) {
        // Compute the gradient update: ds_i/dx_j = s_i * (delta_ij - s_j)
        x_grad[offset + j * axis_stride] +=
            output_grad[offset + i * axis_stride] *
            z_data[offset + i * axis_stride] *
            ((i == j ? 1 : 0) - z_data[offset + j * axis_stride]);
      }
    }
  }
}

void GPUHandler::softmax_backward(float* x_grad, const float* output_grad,
                                  const float* z_data, size_t axis_size,
                                  size_t size_squashed, size_t axis_stride) {
  size_t block_size = 256;
  size_t grid_size = (size_squashed + block_size - 1) / block_size;
  std::cout << "axis_size: " << axis_size << " size squashed: " << size_squashed
            << " axis_stride: " << axis_stride << std::endl;

  softmax_backward_kernel<<<grid_size, block_size>>>(
      x_grad, output_grad, z_data, axis_size, size_squashed, axis_stride);

  cudaDeviceSynchronize();
}