#ifndef CUBLAS_HANDLER_HPP
#define CUBLAS_HANDLER_HPP

#include <cublas_v2.h>

#include <stdexcept>

#define checkCublasErrors(call)                                         \
  {                                                                     \
    cublasStatus_t err;                                                 \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS) {                      \
      throw std::runtime_error("cuBLAS error: " + std::to_string(err)); \
    }                                                                   \
  }

#define checkCudaErrors(call)                                         \
  {                                                                   \
    cudaError_t err;                                                  \
    if ((err = (call)) != cudaSuccess) {                              \
      throw std::runtime_error("CUDA error: " +                       \
                               std::string(cudaGetErrorString(err))); \
    }                                                                 \
  }

class Tensor;

class GPUHandler {
 public:
  static GPUHandler& getInstance();

  static void add(const float* x, const float* y, float* z, size_t size);
  static void sub(const float* x, const float* y, float* z, size_t size);
  static void multiply(const float* x, const float* y, float* z, size_t size);
  static void divide(const float* x, const float* y, float* z, size_t size);
  static void axpy(const float* x, float* y, float alpha, size_t size);
  static void matmul(const float* X, const float* Y, float* Z, size_t B,
                     size_t M, size_t K, size_t N, bool transX = false,
                     bool transY = false);

  cublasHandle_t getHandle() { return handle; }

  GPUHandler(const GPUHandler&) = delete;
  GPUHandler& operator=(const GPUHandler&) = delete;

 private:
  cublasHandle_t handle;

  GPUHandler() {
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("Failed to create cuBLAS handle");
    }
  }

  ~GPUHandler() { cublasDestroy(handle); }
};
#endif