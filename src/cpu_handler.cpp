#include "cpu_handler.hpp"

#include <cblas.h>
#include <omp.h>

#include <cmath>
#include <iostream>

// add two arrays X and Y of size size
void CPUHandler::add(float* X, float* Y, float* Z, size_t size) {
  cblas_scopy(size, X, 1, Z, 1);
  cblas_saxpy(size, 1.0f, Y, 1, Z, 1);
}

// subtract two arrays X and Y of size size
void CPUHandler::sub(float* X, float* Y, float* Z, size_t size) {
  cblas_scopy(size, X, 1, Z, 1);
  cblas_saxpy(size, -1.0f, Y, 1, Z, 1);
}

// multiply two arrays X and Y of size size
void CPUHandler::mul(float* X, float* Y, float* Z, size_t size) {
#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    Z[i] = X[i] * Y[i];
  }
}

// divide two arrays X and Y of size size
void CPUHandler::div(float* X, float* Y, float* Z, size_t size) {
#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    Z[i] = X[i] / Y[i];
  }
}

// transpose a matrix of shape (B, M, N) to (B, N, M)
void CPUHandler::transpose(float* X, float* Y, size_t B, size_t M, size_t N) {
  for (size_t b = 0 ; b < B; ++b){
    cblas_somatcopy(CblasRowMajor, CblasTrans, M, N, 1.0f, &X[b * M * N], N, &Y[b * N * M], M);
  }
}

// Matrix multiplication of X and Y with shape (B, M, K) and (B, K, N)
void CPUHandler::matmul(float* X, float* Y, float* Z, size_t B, size_t M,
                        size_t K, size_t N) {
#pragma omp parallel for
  for (size_t b = 0; b < B; ++b) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f,
                &X[b * M * K], K, &Y[b * K * N], N, 0.0f, &Z[b * M * N], N);
  }
}

// element-wise log of X
void CPUHandler::log(float* X, float* Y, size_t size) {
#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    Y[i] = std::log(X[i]);
  }
}

// element-wise exp of X

void CPUHandler::exp(float* X, float* Y, size_t size) {
#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    Y[i] = std::exp(X[i]);
  }
}

// element-wise sin of X
void CPUHandler::sin(float* X, float* Y, size_t size) {
#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    Y[i] = std::sin(X[i]);
  }
}

// element-wise cos of X
void CPUHandler::cos(float* X, float* Y, size_t size) {
#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    Y[i] = std::cos(X[i]);
  }
}

// element-wise relu of X
void CPUHandler::relu(float* X, float* Y, size_t size) {
#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    Y[i] = std::max(0.0f, X[i]);
  }
}

// select a row from a tensor X of shape x_shape and store it in Z
void CPUHandler::select_idx(float* X, float* Z, std::vector<size_t> x_shape,
                            size_t idx) {
  size_t offset = idx;
  for (size_t i = 1; i < x_shape.size(); ++i) {
    offset *= x_shape[i];
  }

  size_t size = 1;
  for (size_t i = 1; i < x_shape.size(); ++i) {
    size *= x_shape[i];
  }

#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    Z[i] = X[offset + i];
  }
}

// sum a tensor X along an axis and store it in Z
void CPUHandler::sum(float* X, float* Z, std::vector<size_t> x_shape,
                     size_t axis) {
  size_t input_size = 1, output_size = 1;
  for (size_t i = 0; i < x_shape.size(); ++i) {
    input_size *= x_shape[i];
    if (i != axis) {
      output_size *= x_shape[i];
    }
  }

#pragma omp parallel for
  for (size_t i = 0; i < output_size; ++i) {
    Z[i] = 0.0f;
  }

  std::vector<size_t> strides(x_shape.size(), 1);
  for (size_t i = x_shape.size() - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * x_shape[i];
  }

#pragma omp parallel for
  for (size_t i = 0; i < input_size; ++i) {
    size_t idx = i;
    size_t output_idx = 0;
    for (size_t j = 0; j < x_shape.size(); ++j) {
      if (j < axis) {
        output_idx += (idx / strides[j]) * strides[j] / x_shape[axis];
      } else if (j > axis) {
        output_idx += (idx / strides[j]) * strides[j];
      }
      idx %= strides[j];
    }
#pragma omp atomic
    Z[output_idx] += X[i];
  }
}