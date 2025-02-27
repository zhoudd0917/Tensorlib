#ifndef CPU_HANDLER_HPP
#define CPU_HANDLER_HPP

#include <algorithm>
#include <vector>

class Tensor;

class CPUHandler {
 public:
  static void add(float* X, float* Y, float* Z, size_t size);
  static void sub(float* X, float* Y, float* Z, size_t size);
  static void mul(float* X, float* Y, float* Z, size_t size);
  static void div(float* X, float* Y, float* Z, size_t size);
  static void negate(float* X, float* Z, size_t size);
  static void transpose(float* X, float* Y, size_t B, size_t M, size_t N);
  static void matmul(float* X, float* Y, float* Z, size_t B, size_t M, size_t K,
                     size_t N, bool transpose_X = false,
                     bool transpose_Y = false);
  static void log(float* X, float* Y, size_t size);
  static void exp(float* X, float* Y, size_t size);
  static void sin(float* X, float* Y, size_t size);
  static void cos(float* X, float* Y, size_t size);
  static void relu(float* X, float* Y, size_t size);
  static void sigmoid(float* X, float* Y, size_t size);
  static void select_idx(float* X, float* Z, std::vector<size_t> x_shape,
                         size_t idx);
  static void broadcast(float* X, float* Z, const std::vector<size_t>& x_shape,
                        const std::vector<size_t>& z_shape);
  static void sum(float* X, float* Z, std::vector<size_t> x_shape, size_t axis);
  static void sum(float* X, float* Z, size_t size);
  static void mean(float* X, float* Z, std::vector<size_t> x_shape,
                   size_t axis);
  static void mean(float* X, float* Z, size_t size);
  static size_t* max(float* X, float* Z, std::vector<size_t> x_shape,
                     size_t axis);
  static size_t* max(float* X, float* Z, size_t size);
  static size_t* min(float* X, float* Z, std::vector<size_t> x_shape,
                     size_t axis);
  static size_t* min(float* X, float* Z, size_t size);
  static void dot(float* X, float* Y, float* Z, size_t size);
  static void argmax(float* X, float* Z, std::vector<size_t> x_shape,
                     size_t axis);
  static void argmin(float* X, float* Z, std::vector<size_t> x_shape,
                     size_t axis);
  static void softmax(float* X, float* Z, std::vector<size_t> x_shape,
                      size_t axis);
  static void cross_entropy(float* X, float* Y, float* Z,
                            std::vector<size_t> x_shape);
};

#endif