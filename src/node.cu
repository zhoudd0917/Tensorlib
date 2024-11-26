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
  variable output_grad_tensor = output_->autograd_meta().grad_;
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
  variable output_grad_tensor = output_->autograd_meta().grad_;
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
      CPUHandler::sub(output_grad, y_grad, y_grad, y_grad_tensor->size());
    } else if (device == Device::GPU) {
      // Gradient for y with subtraction
      float alpha = -1.0f;
      GPUHandler::axpy(output_grad, y_grad, alpha, y_grad_tensor->size());
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
  variable output_grad_tensor = output_->autograd_meta().grad_;
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
  variable output_grad_tensor = output_->autograd_meta().grad_;
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

MatmulBackward::MatmulBackward(variable output, variable x, variable y) {
  inputs_.push_back(x);
  inputs_.push_back(y);
  output_ = output;

  name_ = "MatmulBackward";

  set_next_edges();
}

void MatmulBackward::apply() {
  // gradient for X is output_grad @ Y^T, and for Y is X^T @ output_grad
  variable output_grad_tensor = output_->autograd_meta().grad_;
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
  variable output_grad_tensor = output_->autograd_meta().grad_;
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
      std::runtime_error("Not implemented for GPU");
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
  variable output_grad_tensor = output_->autograd_meta().grad_;
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
      std::runtime_error("Not implemented for GPU");
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
  variable output_grad_tensor = output_->autograd_meta().grad_;
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
      std::runtime_error("Not implemented for GPU");
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
  variable output_grad_tensor = output_->autograd_meta().grad_;
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
      std::runtime_error("Not implemented for GPU");
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
  variable output_grad_tensor = output_->autograd_meta().grad_;
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
      std::runtime_error("Not implemented for GPU");
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
  variable output_grad_tensor = output_->autograd_meta().grad_;
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
      std::runtime_error("Not implemented for GPU");
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
  variable output_grad_tensor = output_->autograd_meta().grad_;
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
  variable output_grad_tensor = output_->autograd_meta().grad_;
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
      std::runtime_error("Not implemented for GPU");
    }
  }
}

SumBackward::SumBackward(variable output, variable x, size_t axis) {
  inputs_.push_back(x);
  output_ = output;
  axis_ = axis;

  name_ = "SumBackward";

  set_next_edges();
}

void SumBackward::apply() {
  variable output_grad_tensor = output_->autograd_meta().grad_;
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
        size_t output_idx = i, input_idx = 0;
        std::vector<size_t> strides(output_->shape().size(), 1);
        for (int j = output_->shape().size() - 2; j >= 0; --j) {
          strides[j] = strides[j + 1] * output_->shape()[j + 1];
        }

        for (int j = 0; j < x_shape.size(); ++j) {
          if (j == axis_) continue;
          size_t stride;
          if (j < axis_) {
            stride = strides[j];
          } else {
            stride = strides[j - 1];
          }
          size_t idx = output_idx / stride;

          input_idx += idx * x_stride[j];
          output_idx %= stride;
        }

        for (size_t j = 0; j < axis_size; j++) {
          x_grad[input_idx] += output_grad[i];
          input_idx += x_stride[axis_];
        }
      }
    } else if (device == Device::GPU) {
      std::runtime_error("Not implemented for GPU");
    }
  }
}