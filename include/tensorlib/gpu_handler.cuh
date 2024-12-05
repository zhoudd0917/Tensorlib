#ifndef CUBLAS_HANDLER_HPP
#define CUBLAS_HANDLER_HPP

#include <cublas_v2.h>

#include <stdexcept>
#include <vector>

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
  static void negate(const float* x, float* y, size_t size);
  static void axpy(const float* x, float* y, float alpha, size_t size);
  static void matmul(const float* X, const float* Y, float* Z, size_t B,
                     size_t M, size_t K, size_t N, bool transX = false,
                     bool transY = false);
  static void transpose(const float* input, float* output, size_t B, size_t M,
                        size_t N);
  static void select_idx(float* X, float* Z, std::vector<size_t> x_shape,
                         size_t idx);
  static void log(const float* input, float* output, size_t size);
  static void logBackward(const float* output_grad, const float* x_data,
                          float* x_grad, size_t size);
  static void exp(const float* input, float* output, size_t size);
  static void expMul(const float* output_data, float* x_grad,
                     const float* output_grad, size_t size);
  static void sin(const float* input, float* output, size_t size);
  static void cos(const float* input, float* output, size_t size);
  // sin backward, x_grad[i] += output_grad[i] * cos(x_data[i])
  static void sinBackward(const float* output_grad, const float* x_data,
                          float* x_grad, size_t size);
  // cos backward, x_grad[i] += -output_grad[i] * sin(x_data[i])
  static void cosBackward(const float* output_grad, const float* x_data,
                          float* x_grad, size_t size);
  static void relu(const float* input, float* output, size_t size);
  // relu backward, x_grad[i] += output_grad[i] * (x_data[i] > 0 ? 1 : 0)
  static void reluBackward(const float* output_grad, const float* x_data,
                           float* x_grad, size_t size);
  // sigmoid
  static void sigmoid(const float* input, float* output, size_t size);
  // sigmoid backward, x_grad[i] += output_grad[i] * output[i] * (1 - output[i])
  static void sigmoidBackward(const float* output_grad, const float* output,
                              float* x_grad, size_t size);
  static void sum(float* input, float* output, std::vector<size_t> shape,
                  size_t axis);
  // sum all elements
  static void sum(float* input, float* output, size_t size);
  static void mean(float* input, float* output, std::vector<size_t> shape,
                   size_t axis);
  // mean all elements
  static void mean(float* input, float* output, size_t size);
  static void add_axis(float* x_grad, const float* output_grad,
                       std::vector<size_t> x_shape,
                       std::vector<size_t> x_stride, size_t axis,
                       size_t axis_size, float factor);
  static void set_all(float* x, const float* val, float factor, size_t size);
  static void reshape(const float* input, float* output, size_t size);
  static void broadcast(const float* input, float* output,
                        const std::vector<size_t>& input_shape,
                        const std::vector<size_t>& output_shape);
  static void broadcastBackward(const float* output_grad, float* x_grad,
                                const std::vector<size_t>& x_shape,
                                const std::vector<size_t>& z_stride,
                                const std::vector<size_t>& x_stride,
                                size_t z_size);
  // finds the maximum value along the specified axis, and stores the index of
  // the maximum value in index_list return
  static size_t* max(const float* input, float* output,
                     std::vector<size_t> shape, size_t axis);
  // find the maximum value in the tensor
  static size_t* max(const float* input, float* output, size_t size);
  // find the minimum value along the specified axis
  static size_t* min(const float* input, float* output,
                     std::vector<size_t> shape, size_t axis);
  // find the minimum value in the tensor
  static size_t* min(const float* input, float* output, size_t size);
  // softmax along the specified axis
  static void softmax(const float* input, float* output,
                      std::vector<size_t> shape, size_t axis);
  // softmax backward
  static void softmax_backward(float* x_grad, const float* output_grad,
                               const float* z_data, size_t axis_size,
                               size_t size_squashed, size_t axis_stride);
  // cross entropy loss
  static void cross_entropy(const float* x, const float* y, float* z,
                            std::vector<size_t> shape);
  // cross entropy loss backward for x
  static void cross_entropy_backward_x(const float* t_softmax, const float* y,
                                       float* x_grad, const float* output_grad,
                                       size_t batch_size, size_t num_classes);
  // cross entropy loss backward for y
  static void cross_entropy_backward_y(const float* t_softmax, float* y_grad,
                                       const float* output_grad,
                                       size_t batch_size, size_t num_classes);
  static void argmax(const float* input, float* output,
                     std::vector<size_t> shape, size_t axis);
  static void argmin(const float* input, float* output,
                     std::vector<size_t> shape, size_t axis);

  // update the gradient of the input tensor with the gradient of the output
  // x_grad[index_list[i]] += output_grad[i]
  static void update_grad_selector(size_t* index_list, float* x_grad,
                                   float* output_grad, size_t size);

  static float* allocate(size_t size);
  static float* allocate_and_copy(const float* host_data, size_t size);
  static float* allocate_and_zero(size_t size);
  static void deallocate(float* device_data);
  static void deallocate(size_t* device_data);
  static void copy_host_to_device(float* device_data, const float* host_data,
                                  size_t size);
  static void copy_device_to_host(float* host_data, const float* device_data,
                                  size_t size);
  static void copy_device_to_device(float* dst, const float* src, size_t size);
  static void zero(float* x, size_t size);

  cublasHandle_t getHandle() { return handle_; }
  cudaStream_t getStream() { return stream_; }

  GPUHandler(const GPUHandler&) = delete;
  GPUHandler& operator=(const GPUHandler&) = delete;

 private:
  cublasHandle_t handle_;
  cudaStream_t stream_;

  GPUHandler();
  ~GPUHandler();
};
#endif