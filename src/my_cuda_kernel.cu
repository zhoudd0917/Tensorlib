#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <iostream>

// Error handling for cuBLAS
#define CHECK_CUBLAS_ERROR(err)                        \
  if (err != CUBLAS_STATUS_SUCCESS) {                  \
    std::cerr << "cuBLAS error: " << err << std::endl; \
    return;                                            \
  }

void scale_vector_with_cublas(float* x, int n, float alpha) {
  // copy to device
  float* d_x;
  cudaMalloc(&d_x, n * sizeof(float));
  cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasStatus_t status = cublasCreate(&handle);
  CHECK_CUBLAS_ERROR(status);

  status = cublasSscal(handle, n, &alpha, d_x, 1);
  CHECK_CUBLAS_ERROR(status);

  cublasDestroy(handle);

  // copy back to host
  cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
}