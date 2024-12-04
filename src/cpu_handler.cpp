#include <cblas.h>
#include <omp.h>

#include <cmath>
#include <tensorlib/cpu_handler.hpp>
#include <tensorlib/utils.hpp>

// add two arrays X and Y of size size
// Z = X + Y
void CPUHandler::add(float* X, float* Y, float* Z, size_t size) {
#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    Z[i] = X[i] + Y[i];
  }
}

// subtract two arrays X and Y of size size
// Z = X - Y
void CPUHandler::sub(float* X, float* Y, float* Z, size_t size) {
#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    Z[i] = X[i] - Y[i];
  }
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

// negate an array X of size size
void CPUHandler::negate(float* X, float* Z, size_t size) {
#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    Z[i] = -X[i];
  }
}

// transpose a matrix of shape (B, M, N) to (B, N, M)
void CPUHandler::transpose(float* X, float* Y, size_t B, size_t M, size_t N) {
  for (size_t b = 0; b < B; ++b) {
    cblas_somatcopy(CblasRowMajor, CblasTrans, M, N, 1.0f, &X[b * M * N], N,
                    &Y[b * N * M], M);
  }
}

// Matrix multiplication of X and Y with shape (B, M, K) and (B, K, N)
void CPUHandler::matmul(float* X, float* Y, float* Z, size_t B, size_t M,
                        size_t K, size_t N, bool transpose_X,
                        bool transpose_Y) {
  auto trans_x = transpose_X ? CblasTrans : CblasNoTrans,
       trans_y = transpose_Y ? CblasTrans : CblasNoTrans;

#pragma omp parallel for
  for (size_t b = 0; b < B; ++b) {
    cblas_sgemm(CblasRowMajor, trans_x, trans_y, M, N, K, 1.0f, X + b * M * K,
                (transpose_X ? M : K), Y + b * K * N, (transpose_Y ? K : N),
                0.0f, Z + b * M * N, N);
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

// element-wise sigmoid of X
void CPUHandler::sigmoid(float* X, float* Y, size_t size) {
#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    Y[i] = 1.0f / (1.0f + std::exp(-X[i]));
  }
}

// select a row from a tensor X of shape x_shape and
// store it in Z
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

void CPUHandler::broadcast(float* X, float* Z,
                           const std::vector<size_t>& x_shape,
                           const std::vector<size_t>& z_shape) {
  size_t x_dims = x_shape.size();
  size_t z_dims = z_shape.size();

  std::vector<size_t> x_strides = calculate_strides(x_shape);
  std::vector<size_t> z_strides = calculate_strides(z_shape);
  size_t total_elements = calculate_size(z_shape);

// Perform broadcasting
#pragma omp parallel for
  for (size_t i = 0; i < total_elements; ++i) {
    size_t x_index = 0, z_index = i;

    for (size_t dim = 0; dim < z_dims; ++dim) {
      size_t z_coord = z_index / z_strides[dim];
      z_index %= z_strides[dim];

      size_t x_coord = (x_shape[dim] == 1) ? 0 : z_coord;
      x_index += x_coord * x_strides[dim];
    }

    Z[i] = X[x_index];
  }
}

// sum a tensor X along an axis and store it in Z
void CPUHandler::sum(float* X, float* Z, std::vector<size_t> x_shape,
                     size_t axis) {
  size_t input_size = calculate_size(x_shape),
         output_size = input_size / x_shape[axis];

#pragma omp parallel for
  for (size_t i = 0; i < output_size; ++i) {
    Z[i] = 0.0f;
  }

#pragma omp parallel for
  for (size_t i = 0; i < input_size; ++i) {
    size_t output_idx = calculate_index_after_drop_axis(i, axis, x_shape);
#pragma omp atomic
    Z[output_idx] += X[i];
  }
}

// sum all elements of a tensor X
void CPUHandler::sum(float* X, float* Z, size_t size) {
  float sum = 0.0f;
#pragma omp parallel for reduction(+ : sum)
  for (size_t i = 0; i < size; ++i) {
    sum += X[i];
  }
  *Z = sum;
}

// mean a tensor X along an axis and store it in Z
void CPUHandler::mean(float* X, float* Z, std::vector<size_t> x_shape,
                      size_t axis) {
  size_t input_size = calculate_size(x_shape),
         output_size = input_size / x_shape[axis];

  float factor = 1.0f / x_shape[axis];

#pragma omp parallel for
  for (size_t i = 0; i < output_size; ++i) {
    Z[i] = 0.0f;
  }
#pragma omp parallel for
  for (size_t i = 0; i < input_size; ++i) {
    size_t output_idx = calculate_index_after_drop_axis(i, axis, x_shape);
#pragma omp atomic
    Z[output_idx] += X[i] * factor;
  }
}

// mean all elements of a tensor X
void CPUHandler::mean(float* X, float* Z, size_t size) {
  float sum = 0.0f;

#pragma omp parallel for reduction(+ : sum)
  for (size_t i = 0; i < size; ++i) {
    sum += X[i];
  }
  *Z = sum / size;
}

// max a tensor X along an axis and store it in Z, returns the argmax array
size_t* CPUHandler::max(float* X, float* Z, std::vector<size_t> x_shape,
                        size_t axis) {
  size_t input_size = calculate_size(x_shape),
         output_size = input_size / x_shape[axis];

  size_t* idx_list = new size_t[output_size];

#pragma omp parallel for
  for (size_t i = 0; i < output_size; ++i) {
    Z[i] = -INFINITY;
  }

  for (size_t i = 0; i < input_size; ++i) {
    size_t output_idx = calculate_index_after_drop_axis(i, axis, x_shape);
    if (X[i] > Z[output_idx]) {
      Z[output_idx] = X[i];
      idx_list[output_idx] = i;
    }
  }
  return idx_list;
}

// max all elements of a tensor X
size_t* CPUHandler::max(float* X, float* Z, size_t size) {
  float max_val = -INFINITY;
  size_t max_idx = 0;

  for (size_t i = 0; i < size; ++i) {
    if (X[i] > max_val) {
      if (X[i] > max_val) {
        max_val = X[i];
        max_idx = i;
      }
    }
  }

  *Z = max_val;
  size_t* idx_list = new size_t[1];
  idx_list[0] = max_idx;
  return idx_list;
}

// min a tensor X along an axis and store it in Z, retunns the argmin array
size_t* CPUHandler::min(float* X, float* Z, std::vector<size_t> x_shape,
                        size_t axis) {
  size_t input_size = calculate_size(x_shape),
         output_size = input_size / x_shape[axis];

  size_t* idx_list = new size_t[output_size];

#pragma omp parallel for
  for (size_t i = 0; i < output_size; ++i) {
    Z[i] = INFINITY;
  }

  for (size_t i = 0; i < input_size; ++i) {
    size_t output_idx = calculate_index_after_drop_axis(i, axis, x_shape);
    if (X[i] < Z[output_idx]) {
      Z[output_idx] = X[i];
      idx_list[output_idx] = i;
    }
  }
  return idx_list;
}

// min all elements of a tensor X
size_t* CPUHandler::min(float* X, float* Z, size_t size) {
  float min_val = INFINITY;
  size_t min_idx = 0;

  for (size_t i = 0; i < size; ++i) {
    if (X[i] < min_val) {
      if (X[i] < min_val) {
        min_val = X[i];
        min_idx = i;
      }
    }
  }

  *Z = min_val;
  size_t* idx_list = new size_t[1];
  idx_list[0] = min_idx;
  return idx_list;
}

// dot product of two arrays X and Y of size size
void CPUHandler::dot(float* X, float* Y, float* Z, size_t size) {
  *Z = cblas_sdot(size, X, 1, Y, 1);
}

// argmax a tensor X along an axis and store it in Z
void CPUHandler::argmax(float* X, float* Z, std::vector<size_t> x_shape,
                        size_t axis) {
  size_t input_size = calculate_size(x_shape),
         output_size = input_size / x_shape[axis];

  std::vector<size_t> strides = calculate_strides(x_shape);

  std::vector<float> max_values(output_size, -INFINITY);

  for (size_t i = 0; i < input_size; ++i) {
    size_t output_idx = calculate_index_after_drop_axis(i, axis, x_shape);
    if (X[i] > max_values[output_idx]) {
      size_t max_idx = i / strides[axis] % x_shape[axis];
      max_values[output_idx] = X[i];
      Z[output_idx] = max_idx;
    }
  }
}

// argmin a tensor X along an axis and store it in Z
void CPUHandler::argmin(float* X, float* Z, std::vector<size_t> x_shape,
                        size_t axis) {
  size_t input_size = calculate_size(x_shape),
         output_size = input_size / x_shape[axis];

  std::vector<size_t> strides = calculate_strides(x_shape);

  std::vector<float> min_values(output_size, INFINITY);

  for (size_t i = 0; i < input_size; ++i) {
    size_t output_idx = calculate_index_after_drop_axis(i, axis, x_shape);
    if (X[i] < min_values[output_idx]) {
      size_t min_idx = i / strides[axis] % x_shape[axis];
      min_values[output_idx] = X[i];
      Z[output_idx] = min_idx;
    }
  }
}

// softmax of a tensor X along an axis and store it in Z
void CPUHandler::softmax(float* X, float* Z, std::vector<size_t> x_shape,
                         size_t axis) {
  std::vector<size_t> strides = calculate_strides(x_shape);
  size_t size = calculate_size(x_shape);

#pragma omp parallel for
  for (size_t i = 0; i < size / x_shape[axis]; i++) {
    size_t offset = calculate_index_after_add_axis(i, axis, x_shape);

    // for numerical stability
    float max_val = -INFINITY;
    for (size_t j = 0; j < x_shape[axis]; j++) {
      max_val = std::max(max_val, X[offset + j * strides[axis]]);
    }

    float sum = 0.0f;
    for (size_t j = 0; j < x_shape[axis]; j++) {
      sum += std::exp(X[offset + j * strides[axis]] - max_val);
    }

    for (size_t j = 0; j < x_shape[axis]; j++) {
      Z[offset + j * strides[axis]] =
          std::exp(X[offset + j * strides[axis]] - max_val) / sum;
    }
  }
}

// softmax cross entropy loss between X and Y
void CPUHandler::cross_entropy(float* X, float* Y, float* Z,
                               std::vector<size_t> x_shape) {
  size_t batch_size = x_shape[0], num_classes = x_shape[1];

#pragma omp parallel for
  for (size_t i = 0; i < batch_size; ++i) {
    float *x_batch = X + i * num_classes, *y_batch = Y + i * num_classes,
          *z_batch = Z + i;

    float max_val = -INFINITY;
    for (size_t j = 0; j < num_classes; ++j) {
      max_val = std::max(max_val, x_batch[j]);
    }

    float sum = 0.0f;
    for (size_t j = 0; j < num_classes; ++j) {
      sum += std::exp(x_batch[j] - max_val);
    }
    float log_sum = max_val + std::log(sum);

    float loss = 0.0f;
    for (size_t j = 0; j < num_classes; ++j) {
      loss -= y_batch[j] * (x_batch[j] - log_sum);
    }

    *z_batch = loss;
  }
}
