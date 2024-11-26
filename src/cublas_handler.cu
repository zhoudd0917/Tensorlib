#include <tensorlib/cublas_handler.cuh>

CublasHandler& CublasHandler::getInstance() {
  static CublasHandler instance;
  return instance;
}

void CublasHandler::add(const float* x, const float* y, float* z, size_t size) {
  cublasHandle_t handle = getInstance().getHandle();
  float alpha = 1.0;
  checkCudaErrors(
      cudaMemcpy(z, y, size * sizeof(float), cudaMemcpyDeviceToDevice));
  checkCublasErrors(cublasSaxpy(handle, size, &alpha, x, 1, z, 1));
}

void CublasHandler::sub(const float* x, const float* y, float* z, size_t size) {
  cublasHandle_t handle = getInstance().getHandle();
  float alpha = -1.0f;

  checkCudaErrors(
      cudaMemcpy(z, x, size * sizeof(float), cudaMemcpyDeviceToDevice));
  checkCublasErrors(cublasSaxpy(handle, size, &alpha, y, 1, z, 1));
}

void CublasHandler::multiply(const float* x, const float* y, float* z,
                             size_t size) {
  cublasHandle_t handle = getInstance().getHandle();
  cublasSideMode_t mode = CUBLAS_SIDE_LEFT;

  int m = size;    // Number of rows
  int n = 1;       // Single column (vector)
  int lda = size;  // Leading dimension of x (stride between rows)
  int incx = 1;    // Stride of y (scalar/vector elements)
  int ldc = size;  // Leading dimension of output z

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

void CublasHandler::divide(const float* x, const float* y, float* z,
                           size_t size) {
  // Each thread processes one element; set the CUDA grid size
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  // Launch the CUDA kernel to perform element-wise division
  elementWiseDivision<<<gridSize, blockSize>>>(x, y, z, size);

  // Check for CUDA errors
  checkCudaErrors(cudaDeviceSynchronize());
}

// helper method for SubBackward
void CublasHandler::axpy(const float* x, float* y, float alpha, size_t size) {
  cublasHandle_t handle = getInstance().getHandle();
  checkCublasErrors(cublasSaxpy(handle, size, &alpha, x, 1, y, 1));
}

// matrix multiplication
void CublasHandler::matmul(const float* X, const float* Y, float* Z, size_t B,
                           size_t M, size_t K, size_t N) {
  cublasHandle_t handle = getInstance().getHandle();

  // Perform matrix multiplication for each batch
  for (size_t b = 0; b < B; ++b) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Call cuBLAS gemm function for batch computation
    checkCublasErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                  &alpha, Y + b * N * K, N, X + b * M * K, K,
                                  &beta, Z + b * M * N, N));
  }
}
