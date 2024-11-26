#include <tensorlib/autograd.hpp>
#include <tensorlib/cpu_handler.hpp>
#include <tensorlib/gpu_handler.cuh>
#include <tensorlib/node.cuh>
#include <tensorlib/operators.cuh>
#include <tensorlib/tensor.cuh>
#include <tensorlib/utils.hpp>

variable operator+(variable x, variable y) {
  check_tensor_device(x, y);
  if (x->shape().size() != y->shape().size()) {
    throw std::runtime_error("Incompatible shape");
  }
  size_t ndim = x->shape().size();

  bool broadcast_x = false, broadcast_y = false;
  std::vector<size_t> shape;
  for (int i = 0; i < ndim; ++i) {
    if (x->shape()[i] != y->shape()[i]) {
      if (x->shape()[i] == 1) {
        broadcast_x = true;
        shape.push_back(y->shape()[i]);
      } else if (y->shape()[i] == 1) {
        broadcast_y = true;
        shape.push_back(x->shape()[i]);
      } else {
        throw std::runtime_error("Incompatible shape");
      }
    } else {
      shape.push_back(x->shape()[i]);
    }
  }

  if (broadcast_x) {
    x = broadcast_to(x, shape);
  }
  if (broadcast_y) {
    y = broadcast_to(y, shape);
  }

  bool require_grad = x->requires_grad() || y->requires_grad();
  Device device = x->device();

  auto z = std::make_shared<Tensor>(x->shape(), device, require_grad);

  if (device == Device::CPU) {
    CPUHandler::add(x->data(), y->data(), z->data(), x->size());
  } else if (device == Device::GPU) {
    GPUHandler::add(x->data(), y->data(), z->data(), x->size());
  }

  if (require_grad) {
    z->autograd_meta().set_grad_fn(std::make_shared<AddBackward>(z, x, y));
  }

  return z;
}

variable operator+(variable x, float y) {
  Device device = x->device();
  // shape is 1 duplicated to the size of x
  std::shared_ptr<Tensor> y_tensor = std::make_shared<Tensor>(
      std::vector<float>{y}, std::vector<size_t>(x->shape().size(), 1), device,
      false);
  return x + y_tensor;
}

variable operator+(float x, variable y) { return y + x; }

variable operator-(variable x, variable y) {
  check_tensor_device(x, y);
  if (x->shape().size() != y->shape().size()) {
    throw std::runtime_error("Incompatible shape");
  }
  size_t ndim = x->shape().size();

  bool broadcast_x = false, broadcast_y = false;
  std::vector<size_t> shape;
  for (int i = 0; i < ndim; ++i) {
    if (x->shape()[i] != y->shape()[i]) {
      if (x->shape()[i] == 1) {
        broadcast_x = true;
        shape.push_back(y->shape()[i]);
      } else if (y->shape()[i] == 1) {
        broadcast_y = true;
        shape.push_back(x->shape()[i]);
      } else {
        throw std::runtime_error("Incompatible shape");
      }
    } else {
      shape.push_back(x->shape()[i]);
    }
  }

  if (broadcast_x) {
    x = broadcast_to(x, shape);
  }
  if (broadcast_y) {
    y = broadcast_to(y, shape);
  }

  std::cout << y->to_string() << std::endl;

  bool require_grad = x->requires_grad() || y->requires_grad();
  Device device = x->device();

  auto z = std::make_shared<Tensor>(x->shape(), device, require_grad);

  if (device == Device::CPU) {
    CPUHandler::sub(x->data(), y->data(), z->data(), x->size());
  } else if (device == Device::GPU) {
    GPUHandler::sub(x->data(), y->data(), z->data(), x->size());
  }

  if (require_grad) {
    z->autograd_meta().set_grad_fn(std::make_shared<SubBackward>(z, x, y));
  }

  return z;
}

variable operator-(variable x, float y) {
  Device device = x->device();
  std::shared_ptr<Tensor> y_tensor = std::make_shared<Tensor>(
      std::vector<float>{y}, std::vector<size_t>(x->shape().size(), 1), device,
      false);
  return x - y_tensor;
}

variable operator-(float x, variable y) {
  Device device = y->device();
  std::shared_ptr<Tensor> x_tensor = std::make_shared<Tensor>(
      std::vector<float>{x}, std::vector<size_t>(y->shape().size(), 1), device,
      false);
  return x_tensor - y;
}

variable operator*(variable x, variable y) {
  check_tensor_device(x, y);
  if (x->shape().size() != y->shape().size()) {
    throw std::runtime_error("Incompatible shape");
  }
  size_t ndim = x->shape().size();

  bool broadcast_x = false, broadcast_y = false;
  std::vector<size_t> shape;
  for (int i = 0; i < ndim; ++i) {
    if (x->shape()[i] != y->shape()[i]) {
      if (x->shape()[i] == 1) {
        broadcast_x = true;
        shape.push_back(y->shape()[i]);
      } else if (y->shape()[i] == 1) {
        broadcast_y = true;
        shape.push_back(x->shape()[i]);
      } else {
        throw std::runtime_error("Incompatible shape");
      }
    } else {
      shape.push_back(x->shape()[i]);
    }
  }

  if (broadcast_x) {
    x = broadcast_to(x, shape);
  }
  if (broadcast_y) {
    y = broadcast_to(y, shape);
  }

  bool require_grad = x->requires_grad() || y->requires_grad();
  Device device = x->device();

  auto z = std::make_shared<Tensor>(x->shape(), device, require_grad);

  if (device == Device::CPU) {
    CPUHandler::mul(x->data(), y->data(), z->data(), x->size());
  } else if (device == Device::GPU) {
    GPUHandler::multiply(x->data(), y->data(), z->data(), x->size());
  }

  if (require_grad) {
    z->autograd_meta().set_grad_fn(std::make_shared<MulBackward>(z, x, y));
  }

  return z;
}

variable operator*(variable x, float y) {
  Device device = x->device();
  std::shared_ptr<Tensor> y_tensor = std::make_shared<Tensor>(
      std::vector<float>{y}, std::vector<size_t>(x->shape().size(), 1), device,
      false);
  return x * y_tensor;
}

variable operator*(float x, variable y) { return y * x; }

variable operator/(variable x, variable y) {
  check_tensor_device(x, y);
  if (x->shape().size() != y->shape().size()) {
    throw std::runtime_error("Incompatible shape");
  }
  size_t ndim = x->shape().size();

  bool broadcast_x = false, broadcast_y = false;
  std::vector<size_t> shape;
  for (int i = 0; i < ndim; ++i) {
    if (x->shape()[i] != y->shape()[i]) {
      if (x->shape()[i] == 1) {
        broadcast_x = true;
        shape.push_back(y->shape()[i]);
      } else if (y->shape()[i] == 1) {
        broadcast_y = true;
        shape.push_back(x->shape()[i]);
      } else {
        throw std::runtime_error("Incompatible shape");
      }
    } else {
      shape.push_back(x->shape()[i]);
    }
  }

  if (broadcast_x) {
    x = broadcast_to(x, shape);
  }
  if (broadcast_y) {
    y = broadcast_to(y, shape);
  }

  std::cout << y->to_string() << std::endl;

  bool require_grad = x->requires_grad() || y->requires_grad();
  Device device = x->device();

  auto z = std::make_shared<Tensor>(x->shape(), device, require_grad);

  if (device == Device::CPU) {
    CPUHandler::div(x->data(), y->data(), z->data(), x->size());
  } else if (device == Device::GPU) {
    GPUHandler::divide(x->data(), y->data(), z->data(), x->size());
  }

  if (require_grad) {
    z->autograd_meta().set_grad_fn(std::make_shared<DivBackward>(z, x, y));
  }

  return z;
}

variable operator/(variable x, float y) {
  Device device = x->device();
  std::shared_ptr<Tensor> y_tensor = std::make_shared<Tensor>(
      std::vector<float>{y}, std::vector<size_t>(x->shape().size(), 1), device,
      false);
  return x / y_tensor;
}

variable operator/(float x, variable y) {
  Device device = y->device();
  std::shared_ptr<Tensor> x_tensor = std::make_shared<Tensor>(
      std::vector<float>{x}, std::vector<size_t>(y->shape().size(), 1), device,
      false);
  return x_tensor / y;
}

variable matmul(variable x, variable y) {
  check_tensor_device(x, y);

  // either 2d or 3d tensor
  if (x->shape().size() != y->shape().size() || x->shape().size() > 3 ||
      x->shape().size() < 2) {
    throw std::runtime_error("Incompatible shape");
  }

  // check if the inner dimensions match
  if (x->shape().size() == 2) {
    if (x->shape()[1] != y->shape()[0]) {
      throw std::runtime_error("Incompatible shape");
    }
  } else {
    // 3d tensor
    if (x->shape()[2] != y->shape()[1]) {
      throw std::runtime_error("Incompatible shape");
    }
  }

  bool require_grad = x->requires_grad() || y->requires_grad();
  Device device = x->device();

  size_t B, M, N, K;

  std::vector<size_t> shape;
  if (x->shape().size() == 2) {
    B = 1;
    M = x->shape()[0];
    N = y->shape()[1];
    K = x->shape()[1];
    shape = {M, N};
  } else {
    B = x->shape()[0];
    M = x->shape()[1];
    N = y->shape()[2];
    K = x->shape()[2];
    shape = {B, M, N};
  }

  auto z = std::make_shared<Tensor>(shape, device, require_grad);

  if (device == Device::CPU) {
    CPUHandler::matmul(x->data(), y->data(), z->data(), B, M, K, N);
  } else if (device == Device::GPU) {
    GPUHandler::matmul(x->data(), y->data(), z->data(), B, M, K, N);
  }

  if (require_grad) {
    z->autograd_meta().set_grad_fn(std::make_shared<MatmulBackward>(z, x, y));
  }

  return z;
}

variable transpose(variable x) {
  // check size
  if (x->shape().size() < 2 || x->shape().size() > 3) {
    throw std::runtime_error("Incompatible shape");
  }

  Device device = x->device();

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

  variable z;
  if (x->shape().size() == 2) {
    z = std::make_shared<Tensor>(std::vector<size_t>{N, M}, device,
                                 x->requires_grad());
  } else {
    z = std::make_shared<Tensor>(std::vector<size_t>{B, N, M}, device,
                                 x->requires_grad());
  }

  if (device == Device::CPU) {
    CPUHandler::transpose(x->data(), z->data(), B, M, N);
  } else if (device == Device::GPU) {
    GPUHandler::transpose(x->data(), z->data(), B, M, N);
  }

  if (x->requires_grad()) {
    z->autograd_meta().set_grad_fn(std::make_shared<TransposeBackward>(z, x));
  }

  return z;
}

variable log(variable x) {
  Device device = x->device();

  auto z = std::make_shared<Tensor>(x->shape(), device, x->requires_grad());

  if (device == Device::CPU) {
    CPUHandler::log(x->data(), z->data(), x->size());
  } else if (device == Device::GPU) {
    throw std::runtime_error("Not implemented for GPU");
  }

  if (x->requires_grad()) {
    z->autograd_meta().set_grad_fn(std::make_shared<LogBackward>(z, x));
  }

  return z;
}

variable exp(variable x) {
  Device device = x->device();

  auto z = std::make_shared<Tensor>(x->shape(), device, x->requires_grad());

  if (device == Device::CPU) {
    CPUHandler::exp(x->data(), z->data(), x->size());
  } else if (device == Device::GPU) {
    throw std::runtime_error("Not implemented for GPU");
  }

  if (x->requires_grad()) {
    z->autograd_meta().set_grad_fn(std::make_shared<ExpBackward>(z, x));
  }

  return z;
}

variable sin(variable x) {
  Device device = x->device();

  auto z = std::make_shared<Tensor>(x->shape(), device, x->requires_grad());

  if (device == Device::CPU) {
    CPUHandler::sin(x->data(), z->data(), x->size());
  } else if (device == Device::GPU) {
    throw std::runtime_error("Not implemented for GPU");
  }

  if (x->requires_grad()) {
    z->autograd_meta().set_grad_fn(std::make_shared<SinBackward>(z, x));
  }

  return z;
}

variable cos(variable x) {
  Device device = x->device();

  auto z = std::make_shared<Tensor>(x->shape(), device, x->requires_grad());

  if (device == Device::CPU) {
    CPUHandler::cos(x->data(), z->data(), x->size());
  } else if (device == Device::GPU) {
    throw std::runtime_error("Not implemented for GPU");
  }

  if (x->requires_grad()) {
    z->autograd_meta().set_grad_fn(std::make_shared<CosBackward>(z, x));
  }

  return z;
}

variable relu(variable x) {
  Device device = x->device();

  auto z = std::make_shared<Tensor>(x->shape(), device, x->requires_grad());

  if (device == Device::CPU) {
    CPUHandler::relu(x->data(), z->data(), x->size());
  } else if (device == Device::GPU) {
    throw std::runtime_error("Not implemented for GPU");
  }

  if (x->requires_grad()) {
    z->autograd_meta().set_grad_fn(std::make_shared<ReluBackward>(z, x));
  }

  return z;
}

variable select_idx(variable x, size_t index) {
  Device device = x->device();

  if (index >= x->shape()[0]) {
    throw std::runtime_error("Index out of bounds");
  }

  // collapse the first dimension
  std::vector<size_t> shape(x->shape().begin() + 1, x->shape().end());

  if (shape.empty()) {
    shape.push_back(1);
  }

  auto z = std::make_shared<Tensor>(shape, device, x->requires_grad());

  if (device == Device::CPU) {
    CPUHandler::select_idx(x->data(), z->data(), x->shape(), index);
  } else if (device == Device::GPU) {
    GPUHandler::select_idx(x->data(), z->data(), x->shape(), index);
  }

  if (x->requires_grad()) {
    z->autograd_meta().set_grad_fn(
        std::make_shared<SelectBackward>(z, x, index));
  }

  return z;
}

variable reshape(variable x, std::vector<size_t> shape) {
  Device device = x->device();

  // Check if the new shape is compatible with the old shape
  size_t size = 1;
  for (auto& s : shape) size *= s;
  if (size != x->size()) {
    throw std::runtime_error("Incompatible shape");
  }

  auto z = std::make_shared<Tensor>(shape, device, x->requires_grad());

  if (device == Device::CPU) {
    for (int i = 0; i < x->size(); ++i) {
      size_t x_index = convert_to_index(i, x), z_index = convert_to_index(i, z);

      z->data()[z_index] = x->data()[x_index];
    }
  } else if (device == Device::GPU) {
    throw std::runtime_error("Not implemented for GPU");
  }

  if (x->requires_grad()) {
    z->autograd_meta().set_grad_fn(std::make_shared<ReshapeBackward>(z, x));
  }

  return z;
}

variable flatten(variable x) { return reshape(x, {x->size()}); }

// Potentially broadcast expand to a larger shape
variable broadcast_to(variable x, std::vector<size_t> shape) {
  Device device = x->device();

  auto z = std::make_shared<Tensor>(shape, device, x->requires_grad());

  if (device == Device::CPU) {
    CPUHandler::broadcast(x->data(), z->data(), x->shape(), shape);
  } else if (device == Device::GPU) {
    throw std::runtime_error("Not implemented for GPU");
  }

  if (x->requires_grad()) {
    z->autograd_meta().set_grad_fn(std::make_shared<BroadcastBackward>(z, x));
  }

  return z;
}

variable sum(variable x, size_t axis) {
  Device device = x->device();

  if (axis >= x->shape().size()) {
    throw std::runtime_error("Axis out of bounds");
  }

  // new shape
  std::vector<size_t> shape = x->shape();
  shape.erase(shape.begin() + axis);

  auto z = std::make_shared<Tensor>(shape, device, x->requires_grad());

  if (device == Device::CPU) {
    CPUHandler::sum(x->data(), z->data(), x->shape(), axis);
  } else if (device == Device::GPU) {
    throw std::runtime_error("Not implemented for GPU");
  }

  if (x->requires_grad()) {
    z->autograd_meta().set_grad_fn(std::make_shared<SumBackward>(z, x, axis));
  }

  return z;
}

variable mean(variable x, size_t axis) {
  Device device = x->device();

  if (axis >= x->shape().size()) {
    throw std::runtime_error("Axis out of bounds");
  }

  // new shape
  size_t axis_size = x->shape()[axis];
  std::vector<size_t> shape = x->shape();
  shape.erase(shape.begin() + axis);

  auto z = std::make_shared<Tensor>(shape, device, x->requires_grad());

  if (device == Device::CPU) {
    CPUHandler::mean(x->data(), z->data(), x->shape(), axis);
  } else if (device == Device::GPU) {
    throw std::runtime_error("Not implemented for GPU");
  }

  if (x->requires_grad()) {
    z->autograd_meta().set_grad_fn(
        std::make_shared<SumBackward>(z, x, axis, 1. / axis_size));
  }

  return z;
}

variable max(variable x, size_t axis) {
  Device device = x->device();

  if (axis >= x->shape().size()) {
    throw std::runtime_error("Axis out of bounds");
  }

  // new shape
  std::vector<size_t> shape = x->shape();
  shape.erase(shape.begin() + axis);

  auto z = std::make_shared<Tensor>(shape, device, x->requires_grad());

  size_t* idx_list = nullptr;
  if (device == Device::CPU) {
    idx_list = CPUHandler::max(x->data(), z->data(), x->shape(), axis);
  } else if (device == Device::GPU) {
    throw std::runtime_error("Not implemented for GPU");
  }

  if (x->requires_grad()) {
    z->autograd_meta().set_grad_fn(
        std::make_shared<SelectorBackward>(z, x, axis, idx_list));
  }

  return z;
}

variable min(variable x, size_t axis) {
  Device device = x->device();

  if (axis >= x->shape().size()) {
    throw std::runtime_error("Axis out of bounds");
  }

  // new shape
  std::vector<size_t> shape = x->shape();
  shape.erase(shape.begin() + axis);

  auto z = std::make_shared<Tensor>(shape, device, x->requires_grad());

  size_t* idx_list = nullptr;
  if (device == Device::CPU) {
    idx_list = CPUHandler::min(x->data(), z->data(), x->shape(), axis);
  } else if (device == Device::GPU) {
    throw std::runtime_error("Not implemented for GPU");
  }

  if (x->requires_grad()) {
    z->autograd_meta().set_grad_fn(
        std::make_shared<SelectorBackward>(z, x, axis, idx_list));
  }

  return z;
}