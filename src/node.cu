#include <set>
#include <tensorlib/autograd.hpp>
#include <tensorlib/cpu_handler.hpp>
#include <tensorlib/gpu_handler.cuh>
#include <tensorlib/node.cuh>
#include <tensorlib/tensor.cuh>
#include <tensorlib/utils.hpp>

void Node::set_next_edges() {
  // to ensure no duplicate nodes
  std::set<std::shared_ptr<Node>> next_nodes;

  for (auto& input : inputs_) {
    if (!input->requires_grad()) continue;

    std::shared_ptr<Node> input_grad_fn = input->autograd_meta().grad_fn_;
    if (input_grad_fn && next_nodes.find(input_grad_fn) == next_nodes.end()) {
      Edge edge;
      edge.next = input_grad_fn;
      next_edges_.push_back(edge);
      next_nodes.insert(input_grad_fn);
    }
  }
}

AddBackward::AddBackward(variable output, variable x, variable y) {
  inputs_.push_back(x);
  inputs_.push_back(y);
  output_ = output;

  name_ = "AddBackward";

  set_next_edges();
}

void AddBackward::apply() {
  // The gradient of the sum is 1
  // check last_gradient has same size as inputs
  variable output_grad_tensor = output_.lock()->autograd_meta().grad_;
  float* output_grad = output_grad_tensor->data();

  variable x = inputs_[0], y = inputs_[1];

  if (x->requires_grad()) {
    variable x_grad_tensor = x->autograd_meta().grad_;

    check_tensor_shape(x_grad_tensor, output_grad_tensor);
    check_tensor_device(x_grad_tensor, output_grad_tensor);

    Device device = x->device();
    float* x_grad = x_grad_tensor->data();

    if (device == Device::CPU) {
      CPUHandler::add(output_grad, x_grad, x_grad, x_grad_tensor->size());
    } else if (device == Device::GPU) {
      GPUHandler::add(output_grad, x_grad, x_grad, x_grad_tensor->size());
    }
  }

  if (y->requires_grad()) {
    variable y_grad_tensor = y->autograd_meta().grad_;

    check_tensor_shape(y_grad_tensor, output_grad_tensor);
    check_tensor_device(y_grad_tensor, output_grad_tensor);

    Device device = y->device();
    float* y_grad = y_grad_tensor->data();

    if (device == Device::CPU) {
      CPUHandler::add(output_grad, y_grad, y_grad, y_grad_tensor->size());
    } else if (device == Device::GPU) {
      GPUHandler::add(output_grad, y_grad, y_grad, y_grad_tensor->size());
    }
  }
}

SubBackward::SubBackward(variable output, variable x, variable y) {
  inputs_.push_back(x);
  inputs_.push_back(y);
  output_ = output;

  name_ = "SubBackward";

  set_next_edges();
}

void SubBackward::apply() {
  // The gradient of the difference is 1
  // check last_gradient has same size as inputs
  variable output_grad_tensor = output_.lock()->autograd_meta().grad_;
  float* output_grad = output_grad_tensor->data();

  variable x = inputs_[0], y = inputs_[1];

  if (x->requires_grad()) {
    variable x_grad_tensor = x->autograd_meta().grad_;
    check_tensor_shape(x_grad_tensor, output_grad_tensor);

    float* x_grad = x_grad_tensor->data();
    Device device = x->device();

    if (device == Device::CPU) {
      CPUHandler::add(output_grad, x_grad, x_grad, x_grad_tensor->size());
    } else if (device == Device::GPU) {
      // ignoring strides for now
      GPUHandler::add(output_grad, x_grad, x_grad, x_grad_tensor->size());
    }
  }

  if (y->requires_grad()) {
    variable y_grad_tensor = y->autograd_meta().grad_;
    check_tensor_shape(y_grad_tensor, output_grad_tensor);

    float* y_grad = y_grad_tensor->data();

    Device device = y->device();

    if (device == Device::CPU) {
      CPUHandler::sub(y_grad, output_grad, y_grad, y_grad_tensor->size());
    } else if (device == Device::GPU) {
      // Gradient for y with subtraction
      GPUHandler::axpy(output_grad, y_grad, -1.f, y_grad_tensor->size());
    }
  }
}

MulBackward::MulBackward(variable output, variable x, variable y) {
  inputs_.push_back(x);
  inputs_.push_back(y);
  output_ = output;

  name_ = "MulBackward";

  set_next_edges();
}

void MulBackward::apply() {
  // The gradient of the product is the other input
  // check last_gradient has same size as inputs
  variable output_grad_tensor = output_.lock()->autograd_meta().grad_;
  float* output_grad = output_grad_tensor->data();

  variable x = inputs_[0], y = inputs_[1];

  if (x->requires_grad()) {
    variable x_grad_tensor = x->autograd_meta().grad_;

    check_tensor_shape(x_grad_tensor, output_grad_tensor);

    float* x_grad = x_grad_tensor->data();
    Device device = x->device();

    if (device == Device::CPU) {
      for (size_t i = 0; i < x_grad_tensor->size(); i++) {
        x_grad[i] += output_grad[i] * y->data()[i];
      }
    } else if (device == Device::GPU) {
      // GPU logic for x gradient: x_grad += output_grad * y
      float* y_data = y->data();
      GPUHandler::multiply(output_grad, y_data, x_grad, x_grad_tensor->size());
    }
  }

  if (y->requires_grad()) {
    variable y_grad_tensor = y->autograd_meta().grad_;
    check_tensor_shape(y_grad_tensor, output_grad_tensor);

    float* y_grad = y_grad_tensor->data();
    Device device = y->device();

    if (device == Device::CPU) {
      for (size_t i = 0; i < y_grad_tensor->size(); i++) {
        y_grad[i] += output_grad[i] * x->data()[i];
      }
    } else if (device == Device::GPU) {
      // GPU logic for y gradient: y_grad += output_grad * x
      float* x_data = x->data();
      GPUHandler::multiply(output_grad, x_data, y_grad, y_grad_tensor->size());
    }
  }
}

DivBackward::DivBackward(variable output, variable x, variable y) {
  inputs_.push_back(x);
  inputs_.push_back(y);
  output_ = output;

  name_ = "DivBackward";

  set_next_edges();
}

void DivBackward::apply() {
  // The gradient of the division is 1/y
  // check last_gradient has same size as inputs
  variable output_grad_tensor = output_.lock()->autograd_meta().grad_;
  float* output_grad = output_grad_tensor->data();

  variable x = inputs_[0], y = inputs_[1];

  if (x->requires_grad()) {
    variable x_grad_tensor = x->autograd_meta().grad_;
    check_tensor_shape(x_grad_tensor, output_grad_tensor);
    float* x_grad = x_grad_tensor->data();

    Device device = x->device();

    if (device == Device::CPU) {
      for (size_t i = 0; i < x_grad_tensor->size(); i++) {
        x_grad[i] += output_grad[i] / y->data()[i];
      }
    } else if (device == Device::GPU) {
      // Use the divide utility to compute x_grad += output_grad / y
      GPUHandler::divide(output_grad, y->data(), x_grad, x_grad_tensor->size());
    }
  }

  if (y->requires_grad()) {
    variable y_grad_tensor = y->autograd_meta().grad_;
    check_tensor_shape(y_grad_tensor, output_grad_tensor);

    float* y_grad = y_grad_tensor->data();
    Device device = y->device();

    if (device == Device::CPU) {
      for (size_t i = 0; i < y_grad_tensor->size(); i++) {
        y_grad[i] -=
            output_grad[i] * x->data()[i] / (y->data()[i] * y->data()[i]);
      }
    } else if (device == Device::GPU) {
      // Allocate a temporary buffer for intermediate computations
      float* temp_grad_x;
      checkCudaErrors(
          cudaMalloc(&temp_grad_x, y_grad_tensor->size() * sizeof(float)));

      // Compute output_grad * x and store in temp_grad_x
      GPUHandler::multiply(output_grad, x->data(), temp_grad_x,
                           y_grad_tensor->size());

      // Compute temp_grad_x / (y^2) and store in y_grad
      GPUHandler::divide(temp_grad_x, y->data(), y_grad, y_grad_tensor->size());

      // Multiply y_grad by -1 (axpy helper can be used)
      GPUHandler::axpy(y_grad, y_grad, -1.0, y_grad_tensor->size());

      // Free the temporary buffer
      checkCudaErrors(cudaFree(temp_grad_x));
    }
  }
}

NegateBackward::NegateBackward(variable output, variable x) {
  inputs_.push_back(x);
  output_ = output;

  name_ = "NegateBackward";

  set_next_edges();
}

void NegateBackward::apply() {
  // The gradient of the negation is -1
  // check last_gradient has same size as inputs
  variable output_grad_tensor = output_.lock()->autograd_meta().grad_;
  float* output_grad = output_grad_tensor->data();

  variable x = inputs_[0];

  if (x->requires_grad()) {
    variable x_grad_tensor = x->autograd_meta().grad_;
    check_tensor_shape(x_grad_tensor, output_grad_tensor);

    float* x_grad = x_grad_tensor->data();
    Device device = x->device();

    if (device == Device::CPU) {
      for (size_t i = 0; i < x_grad_tensor->size(); i++) {
        x_grad[i] -= output_grad[i];
      }
    } else if (device == Device::GPU) {
      GPUHandler::axpy(x_grad, output_grad, -1.0, x_grad_tensor->size());
    }
  }
}

MatmulBackward::MatmulBackward(variable output, variable x, variable y) {
  inputs_.push_back(x);
  inputs_.push_back(y);
  output_ = output;

  name_ = "MatmulBackward";

  set_next_edges();
}

void MatmulBackward::apply() {
  // gradient for X is output_grad @ Y^T, and for Y is X^T @ output_grad
  variable output_grad_tensor = output_.lock()->autograd_meta().grad_;
  float* output_grad = output_grad_tensor->data();

  variable x = inputs_[0], y = inputs_[1];

  size_t B, M, N, K;

  if (x->shape().size() == 2) {
    B = 1;
    M = x->shape()[0];
    N = y->shape()[1];
    K = x->shape()[1];
  } else {
    B = x->shape()[0];
    M = x->shape()[1];
    N = y->shape()[2];
    K = x->shape()[2];
  }

  if (x->requires_grad()) {
    variable x_grad_tensor = x->autograd_meta().grad_;

    Device device = x->device();
    float* x_grad = x_grad_tensor->data();

    if (device == Device::CPU) {
      CPUHandler::matmul(output_grad, y->data(), x_grad, B, M, N, K, false,
                         true);
    } else if (device == Device::GPU) {
      GPUHandler::matmul(output_grad, y->data(), x_grad, B, M, N, K, false,
                         true);
    }
  }

  if (y->requires_grad()) {
    variable y_grad_tensor = y->autograd_meta().grad_;

    Device device = y->device();
    float* y_grad = y_grad_tensor->data();

    if (device == Device::CPU) {
      CPUHandler::matmul(x->data(), output_grad, y_grad, B, K, M, N, true,
                         false);
    } else if (device == Device::GPU) {
      GPUHandler::matmul(x->data(), output_grad, y_grad, B, K, M, N, true,
                         false);
    }
  }
}

TransposeBackward::TransposeBackward(variable output, variable x) {
  inputs_.push_back(x);
  output_ = output;

  name_ = "TransposeBackward";

  set_next_edges();
}

void TransposeBackward::apply() {
  // The gradient of the transpose is the transpose itself
  // check last_gradient has same size as inputs
  variable output_grad_tensor = output_.lock()->autograd_meta().grad_;
  float* output_grad = output_grad_tensor->data();

  variable x = inputs_[0];

  if (x->requires_grad()) {
    variable x_grad_tensor = x->autograd_meta().grad_;

    // check tensor shapes match
    if (x_grad_tensor->shape().size() != x->shape().size() ||
        x_grad_tensor->shape().size() > 3 ||
        x_grad_tensor->shape().size() < 2) {
      std::cerr << "Gradient shape mismatch" << std::endl;
      throw std::runtime_error("Gradient shape mismatch");
    }
    if (x_grad_tensor->shape().size() == 3) {
      if (x_grad_tensor->shape()[0] != output_grad_tensor->shape()[0] ||
          x_grad_tensor->shape()[1] != output_grad_tensor->shape()[2] ||
          x_grad_tensor->shape()[2] != output_grad_tensor->shape()[1]) {
        std::cerr << "Gradient shape mismatch" << std::endl;
        throw std::runtime_error("Gradient shape mismatch");
      }
    } else {
      if (x_grad_tensor->shape()[0] != output_grad_tensor->shape()[1] ||
          x_grad_tensor->shape()[1] != output_grad_tensor->shape()[0]) {
        std::cerr << "Gradient shape mismatch" << std::endl;
        throw std::runtime_error("Gradient shape mismatch");
      }
    }

    Device device = x->device();
    float* x_grad = x_grad_tensor->data();

    size_t B, M, N;
    if (x->shape().size() == 2) {
      B = 1;
      M = x->shape()[0];
      N = x->shape()[1];
    } else {
      B = x->shape()[0];
      M = x->shape()[1];
      N = x->shape()[2];
    }

    if (device == Device::CPU) {
      CPUHandler::transpose(output_grad, x_grad, B, N, M);
    } else if (device == Device::GPU) {
      GPUHandler::transpose(output_grad, x_grad, B, N, M);
    }
  }
}

LogBackward::LogBackward(variable output, variable x) {
  inputs_.push_back(x);
  output_ = output;

  name_ = "LogBackward";

  set_next_edges();
}

void LogBackward::apply() {
  // The gradient of the log is 1/x
  // check last_gradient has same size as inputs
  variable output_grad_tensor = output_.lock()->autograd_meta().grad_;
  float* output_grad = output_grad_tensor->data();

  variable x = inputs_[0];

  if (x->requires_grad()) {
    variable x_grad_tensor = x->autograd_meta().grad_;
    check_tensor_shape(x_grad_tensor, output_grad_tensor);

    float* x_grad = x_grad_tensor->data();
    Device device = x->device();

    if (device == Device::CPU) {
      for (size_t i = 0; i < x_grad_tensor->size(); i++) {
        x_grad[i] += output_grad[i] / x->data()[i];
      }
    } else if (device == Device::GPU) {
      GPUHandler::logBackward(output_grad, x->data(), x_grad,
                              x_grad_tensor->size());
    }
  }
}

ExpBackward::ExpBackward(variable output, variable x) {
  inputs_.push_back(x);
  output_ = output;

  name_ = "ExpBackward";

  set_next_edges();
}

void ExpBackward::apply() {
  // The gradient of the exp is exp(x)
  // check last_gradient has same size as inputs
  variable output_grad_tensor = output_.lock()->autograd_meta().grad_;
  float* output_grad = output_grad_tensor->data();

  variable x = inputs_[0];

  if (x->requires_grad()) {
    variable x_grad_tensor = x->autograd_meta().grad_;
    check_tensor_shape(x_grad_tensor, output_grad_tensor);

    float* x_grad = x_grad_tensor->data();
    Device device = x->device();

    if (device == Device::CPU) {
      for (size_t i = 0; i < x_grad_tensor->size(); i++) {
        x_grad[i] += output_grad[i] * std::exp(x->data()[i]);
      }
    } else if (device == Device::GPU) {
      GPUHandler::expMul(output_.lock()->data(), x_grad, output_grad,
                         x_grad_tensor->size());
    }
  }
}

SinBackward::SinBackward(variable output, variable x) {
  inputs_.push_back(x);
  output_ = output;

  name_ = "SinBackward";

  set_next_edges();
}

void SinBackward::apply() {
  // The gradient of the sin is cos(x)
  // check last_gradient has same size as inputs
  variable output_grad_tensor = output_.lock()->autograd_meta().grad_;
  float* output_grad = output_grad_tensor->data();

  variable x = inputs_[0];

  if (x->requires_grad()) {
    variable x_grad_tensor = x->autograd_meta().grad_;
    check_tensor_shape(x_grad_tensor, output_grad_tensor);

    float* x_grad = x_grad_tensor->data();
    Device device = x->device();

    if (device == Device::CPU) {
      for (size_t i = 0; i < x_grad_tensor->size(); i++) {
        x_grad[i] += output_grad[i] * std::cos(x->data()[i]);
      }
    } else if (device == Device::GPU) {
      GPUHandler::sinBackward(output_grad, x->data(), x_grad,
                              x_grad_tensor->size());
    }
  }
}

CosBackward::CosBackward(variable output, variable x) {
  inputs_.push_back(x);
  output_ = output;

  name_ = "CosBackward";

  set_next_edges();
}

void CosBackward::apply() {
  // The gradient of the cos is -sin(x)
  // check last_gradient has same size as inputs
  variable output_grad_tensor = output_.lock()->autograd_meta().grad_;
  float* output_grad = output_grad_tensor->data();

  variable x = inputs_[0];

  if (x->requires_grad()) {
    variable x_grad_tensor = x->autograd_meta().grad_;
    check_tensor_shape(x_grad_tensor, output_grad_tensor);

    float* x_grad = x_grad_tensor->data();
    Device device = x->device();

    if (device == Device::CPU) {
      for (size_t i = 0; i < x_grad_tensor->size(); i++) {
        x_grad[i] -= output_grad[i] * std::sin(x->data()[i]);
      }
    } else if (device == Device::GPU) {
      GPUHandler::cosBackward(output_grad, x->data(), x_grad,
                              x_grad_tensor->size());
    }
  }
}

ReluBackward::ReluBackward(variable output, variable x) {
  inputs_.push_back(x);
  output_ = output;

  name_ = "ReluBackward";

  set_next_edges();
}

void ReluBackward::apply() {
  // The gradient of the ReLU is 1 if x > 0, 0 otherwise
  // check last_gradient has same size as inputs
  variable output_grad_tensor = output_.lock()->autograd_meta().grad_;
  float* output_grad = output_grad_tensor->data();

  variable x = inputs_[0];

  if (x->requires_grad()) {
    variable x_grad_tensor = x->autograd_meta().grad_;
    check_tensor_shape(x_grad_tensor, output_grad_tensor);

    float* x_grad = x_grad_tensor->data();
    Device device = x->device();

    if (device == Device::CPU) {
      for (size_t i = 0; i < x_grad_tensor->size(); i++) {
        x_grad[i] += output_grad[i] * (x->data()[i] > 0);
      }
    } else if (device == Device::GPU) {
      GPUHandler::reluBackward(output_grad, x->data(), x_grad,
                               x_grad_tensor->size());
    }
  }
}

SigmoidBackward::SigmoidBackward(variable output, variable x) {
  inputs_.push_back(x);
  output_ = output;

  name_ = "SigmoidBackward";

  set_next_edges();
}

void SigmoidBackward::apply() {
  // The gradient of the sigmoid is sigmoid(x) * (1 - sigmoid(x))
  // check last_gradient has same size as inputs
  variable output_grad_tensor = output_.lock()->autograd_meta().grad_;
  float* output_grad = output_grad_tensor->data();

  variable x = inputs_[0];

  if (x->requires_grad()) {
    variable x_grad_tensor = x->autograd_meta().grad_;
    check_tensor_shape(x_grad_tensor, output_grad_tensor);

    float* x_grad = x_grad_tensor->data();
    Device device = x->device();

    if (device == Device::CPU) {
      for (size_t i = 0; i < x_grad_tensor->size(); i++) {
        float sigmoid_x = output_.lock()->data()[i];
        x_grad[i] += output_grad[i] * sigmoid_x * (1 - sigmoid_x);
      }
    } else if (device == Device::GPU) {
      GPUHandler::sigmoidBackward(output_grad, output_.lock()->data(), x_grad,
                                  x_grad_tensor->size());
    }
  }
}

SelectBackward::SelectBackward(variable output, variable x, size_t index) {
  inputs_.push_back(x);
  output_ = output;
  index_ = index;

  name_ = "SelectBackward";

  set_next_edges();
}

void SelectBackward::apply() {
  variable output_grad_tensor = output_.lock()->autograd_meta().grad_;
  float* output_grad = output_grad_tensor->data();

  variable x = inputs_[0];

  if (x->requires_grad()) {
    variable x_grad_tensor = x->autograd_meta().grad_;
    float* x_grad = x_grad_tensor->data();

    Device device = x->device();

    if (device == Device::CPU) {
      const std::vector<size_t>& x_shape = x->shape();
      size_t offset = index_;
      for (size_t i = 1; i < x_shape.size(); ++i) {
        offset *= x_shape[i];
      }

      size_t size = 1;
      for (size_t i = 1; i < x_shape.size(); ++i) {
        size *= x_shape[i];
      }

      for (size_t i = 0; i < size; i++) {
        x_grad[offset + i] += output_grad[i];
      }
    } else if (device == Device::GPU) {
      std::runtime_error("Not implemented for GPU");
    }
  }
}

ReshapeBackward::ReshapeBackward(variable output, variable x) {
  inputs_.push_back(x);
  output_ = output;

  name_ = "ReshapeBackward";

  set_next_edges();
}

void ReshapeBackward::apply() {
  variable output_grad_tensor = output_.lock()->autograd_meta().grad_;
  float* output_grad = output_grad_tensor->data();

  variable x = inputs_[0];

  if (x->requires_grad()) {
    variable x_grad_tensor = x->autograd_meta().grad_;
    float* x_grad = x_grad_tensor->data();

    if (output_grad_tensor->size() != x_grad_tensor->size()) {
      throw std::runtime_error("Gradient shape mismatch");
    }

    Device device = x->device();

    if (device == Device::CPU) {
      // reshape is just a view, so we can directly copy the data
      for (size_t i = 0; i < x_grad_tensor->size(); i++) {
        x_grad[i] += output_grad[i];
      }
    } else if (device == Device::GPU) {
      // copy memory
      checkCudaErrors(cudaMemcpy(x_grad, output_grad,
                                 x_grad_tensor->size() * sizeof(float),
                                 cudaMemcpyDeviceToDevice));
    }
  }
}

BroadcastBackward::BroadcastBackward(variable output, variable x) {
  inputs_.push_back(x);
  output_ = output;

  name_ = "BroadcastBackward";

  set_next_edges();
}

void BroadcastBackward::apply() {
  variable output_grad_tensor = output_.lock()->autograd_meta().grad_;
  float* output_grad = output_grad_tensor->data();

  variable x = inputs_[0];

  if (x->requires_grad()) {
    variable x_grad_tensor = x->autograd_meta().grad_;
    float* x_grad = x_grad_tensor->data();

    if (output_grad_tensor->shape().size() != x_grad_tensor->shape().size()) {
      throw std::runtime_error("Gradient dim mismatch");
    }

    const std::vector<size_t>&x_shape = x_grad_tensor->shape(),
          &z_shape = output_grad_tensor->shape(),
          &x_stride = x_grad_tensor->stride(),
          &z_stride = output_grad_tensor->stride();

    Device device = x->device();

    if (device == Device::CPU) {
      size_t ndim = x_shape.size();
      size_t z_size = output_grad_tensor->size();

      for (size_t i = 0; i < z_size; i++) {
        size_t x_index = 0, z_index = i;

        for (size_t dim = 0; dim < ndim; ++dim) {
          size_t z_coord = z_index / z_stride[dim];
          z_index %= z_stride[dim];

          size_t x_coord = (x_shape[dim] == 1) ? 0 : z_coord;
          x_index += x_coord * x_stride[dim];
        }

        x_grad[x_index] += output_grad[i];
      }

    } else if (device == Device::GPU) {
      GPUHandler::broadcastBackward(output_grad, x_grad, x_shape, z_stride,
                                    x_stride, output_grad_tensor->size());
    }
  }
}

SumBackward::SumBackward(variable output, variable x, size_t axis,
                         float factor) {
  inputs_.push_back(x);
  output_ = output;
  axis_ = axis;
  factor_ = factor;

  name_ = "SumBackward";

  set_next_edges();
}

void SumBackward::apply() {
  variable output_grad_tensor = output_.lock()->autograd_meta().grad_;
  float* output_grad = output_grad_tensor->data();

  variable x = inputs_[0];

  if (x->requires_grad()) {
    variable x_grad_tensor = x->autograd_meta().grad_;
    float* x_grad = x_grad_tensor->data();

    const std::vector<size_t>& x_shape = x->shape();
    const std::vector<size_t>& x_stride = x->stride();

    size_t size = output_grad_tensor->size();
    size_t axis_size = x_shape[axis_];

    Device device = x->device();

    if (device == Device::CPU) {
      for (size_t i = 0; i < size; i++) {
        size_t input_idx = calculate_index_after_add_axis(i, axis_, x_shape);

        for (size_t j = 0; j < axis_size; j++) {
          x_grad[input_idx] += output_grad[i] * factor_;
          input_idx += x_stride[axis_];
        }
      }
    } else if (device == Device::GPU) {
      GPUHandler::add_axis(x_grad, output_grad, x_shape, x_stride, axis_,
                           axis_size, factor_);
    }
  }
}

SumAllBackward::SumAllBackward(variable output, variable x, float factor) {
  inputs_.push_back(x);
  output_ = output;
  factor_ = factor;

  name_ = "SumAllBackward";

  set_next_edges();
}

void SumAllBackward::apply() {
  variable output_grad_tensor = output_.lock()->autograd_meta().grad_;
  float* output_grad = output_grad_tensor->data();

  variable x = inputs_[0];

  if (x->requires_grad()) {
    variable x_grad_tensor = x->autograd_meta().grad_;
    float* x_grad = x_grad_tensor->data();

    size_t size = x_grad_tensor->size();

    Device device = x->device();

    if (device == Device::CPU) {
      for (size_t i = 0; i < size; i++) {
        x_grad[i] += output_grad[0] * factor_;
      }
    } else if (device == Device::GPU) {
      GPUHandler::set_all(x_grad, output_grad, factor_, size);
    }
  }
}

SelectorBackward::SelectorBackward(variable output, variable x, size_t axis,
                                   size_t* index_list) {
  inputs_.push_back(x);
  output_ = output;
  axis_ = axis;
  index_list_ = index_list;
  device_ = output->device();

  name_ = "SelectorBackward";

  set_next_edges();
}

SelectorBackward::~SelectorBackward() {
  if (device_ == Device::GPU) {
    cudaFree(index_list_);
  } else if (device_ == Device::CPU) {
    delete[] index_list_;
  }
}

void SelectorBackward::apply() {
  variable output_grad_tensor = output_.lock()->autograd_meta().grad_;
  float* output_grad = output_grad_tensor->data();

  variable x = inputs_[0];

  if (x->requires_grad()) {
    variable x_grad_tensor = x->autograd_meta().grad_;
    float* x_grad = x_grad_tensor->data();
    size_t size = output_grad_tensor->size();

    Device device = x->device();

    if (device == Device::CPU) {
#pragma omp parallel for
      for (size_t i = 0; i < size; i++) {
        size_t og_idx = index_list_[i];
        x_grad[og_idx] += output_grad[i];
      }
    } else if (device == Device::GPU) {
      GPUHandler::update_grad_selector(index_list_, x_grad, output_grad, size);
    }
  }
}

SelectAllBackward::SelectAllBackward(variable output, variable x,
                                     size_t* index) {
  inputs_.push_back(x);
  output_ = output;
  index_ = index;
  device_ = output->device();

  name_ = "SelectAllBackward";

  set_next_edges();
}

SelectAllBackward::~SelectAllBackward() {
  if (device_ == Device::GPU) {
    cudaFree(index_);
  } else if (device_ == Device::CPU) {
    delete[] index_;
  }
}

void SelectAllBackward::apply() {
  variable output_grad_tensor = output_.lock()->autograd_meta().grad_;
  float* output_grad = output_grad_tensor->data();

  variable x = inputs_[0];

  if (x->requires_grad()) {
    variable x_grad_tensor = x->autograd_meta().grad_;
    float* x_grad = x_grad_tensor->data();

    Device device = x->device();

    if (device == Device::CPU) {
      x_grad[*index_] += output_grad[0];
    } else if (device == Device::GPU) {
      GPUHandler::update_grad_selector(index_, x_grad, output_grad, 1);
    }
  }
}

SoftmaxBackward::SoftmaxBackward(variable output, variable x, size_t axis) {
  inputs_.push_back(x);
  output_ = output;
  axis_ = axis;

  name_ = "SoftmaxBackward";

  set_next_edges();
}

void SoftmaxBackward::apply() {
  variable output_grad_tensor = output_.lock()->autograd_meta().grad_;
  float* output_grad = output_grad_tensor->data();

  variable x = inputs_[0], z = output_.lock();

  Device device = x->device();

  if (x->requires_grad()) {
    variable x_grad_tensor = x->autograd_meta().grad_;
    float *x_grad = x_grad_tensor->data(), *z_data = z->data();

    const std::vector<size_t>& stride = x->stride();

    if (device == Device::CPU) {
      size_t size_squashed = x->size() / x->shape()[axis_];
      for (size_t k = 0; k < size_squashed; k++) {
        size_t offset = calculate_index_after_add_axis(k, axis_, x->shape());

#pragma omp parallel for
        for (size_t j = 0; j < x->shape()[axis_]; j++) {
          for (size_t i = 0; i < x->shape()[axis_]; i++) {
            // ds_i/dx_j = s_i * (delta_ij - s_j)
            x_grad[offset + j * stride[axis_]] +=
                output_grad[offset + i * stride[axis_]] *
                z_data[offset + i * stride[axis_]] *
                ((i == j ? 1 : 0) - z_data[offset + j * stride[axis_]]);
          }
        }
      }
    } else if (device == Device::GPU) {
      GPUHandler::softmax_backward(
          x_grad, output_grad, z_data, x->shape()[axis_],
          x->size() / x->shape()[axis_], stride[axis_]);
    }
  }
}

CrossEntropyBackward::CrossEntropyBackward(variable output, variable x,
                                           variable y) {
  inputs_.push_back(x);
  inputs_.push_back(y);
  output_ = output;

  name_ = "CrossEntropyBackward";

  set_next_edges();
}

void CrossEntropyBackward::apply() {
  variable output_grad_tensor = output_.lock()->autograd_meta().grad_;
  float* output_grad = output_grad_tensor->data();

  variable x = inputs_[0], y = inputs_[1];

  Device device = x->device();

  size_t batch_size = x->shape()[0], num_classes = x->shape()[1];
  float* t_softmax = nullptr;
  if (device == Device::CPU) {
    t_softmax = new float[batch_size * num_classes];
    CPUHandler::softmax(x->data(), t_softmax, x->shape(), 1);
  } else if (device == Device::GPU) {
    cudaMalloc(&t_softmax, batch_size * num_classes * sizeof(float));
    GPUHandler::softmax(x->data(), t_softmax, x->shape(), 1);
  }

  if (x->requires_grad()) {
    variable x_grad_tensor = x->autograd_meta().grad_;
    float* x_grad = x_grad_tensor->data();

    if (device == Device::CPU) {
      for (size_t b = 0; b < batch_size; b++) {
        float *x_grad_batch = x_grad + b * num_classes,
              *t_softmax_batch = t_softmax + b * num_classes,
              *y_batch = y->data() + b * num_classes;
        for (size_t i = 0; i < num_classes; i++) {
          float y_sum = 0.;
          for (size_t j = 0; j < num_classes; j++) {
            y_sum += y_batch[j];
          }
          x_grad_batch[i] +=
              output_grad[b] * (-y_batch[i] + t_softmax_batch[i] * y_sum);
        }
      }
    } else if (device == Device::GPU) {
      GPUHandler::cross_entropy_backward_x(
          t_softmax, y->data(), x_grad, output_grad, batch_size, num_classes);
    }
  }

  if (y->requires_grad()) {
    variable y_grad_tensor = y->autograd_meta().grad_;
    float* y_grad = y_grad_tensor->data();
    if (device == Device::CPU) {
      for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < num_classes; i++) {
          y_grad[b * num_classes + i] +=
              output_grad[b] * (-std::log(t_softmax[b * num_classes + i]));
        }
      }
    } else if (device == Device::GPU) {
      GPUHandler::cross_entropy_backward_y(t_softmax, y_grad, output_grad,
                                           batch_size, num_classes);
    }
  }

  if (device == Device::CPU) {
    delete[] t_softmax;
  } else if (device == Device::GPU) {
    cudaFree(t_softmax);
  }
}