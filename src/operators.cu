#include <tensorlib/autograd.hpp>
#include <tensorlib/cpu_handler.hpp>
#include <tensorlib/gpu_handler.cuh>
#include <tensorlib/node.cuh>
#include <tensorlib/operators.cuh>
#include <tensorlib/tensor.cuh>
#include <tensorlib/utils.hpp>

variable operator+(const variable& x, const variable& y) {
  check_tensor_shape(x, y);
  check_tensor_device(x, y);

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

variable operator-(const variable& x, const variable& y) {
  check_tensor_shape(x, y);
  check_tensor_device(x, y);

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

variable operator*(const variable& x, const variable& y) {
  check_tensor_shape(x, y);
  check_tensor_device(x, y);

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

variable operator/(const variable& x, const variable& y) {
  check_tensor_shape(x, y);
  check_tensor_device(x, y);

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

variable matmul(const variable& x, const variable& y) {
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

variable transpose(const variable& x) {
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
    throw std::runtime_error("Not implemented for GPU");
  }

  if (x->requires_grad()) {
    z->autograd_meta().set_grad_fn(std::make_shared<TransposeBackward>(z, x));
  }

  return z;
}

variable log(const variable& x) {
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

variable exp(const variable& x) {
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

variable sin(const variable& x) {
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

variable cos(const variable& x) {
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

variable relu(const variable& x) {
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

variable select_idx(const variable& x, size_t index) {
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
    throw std::runtime_error("Not implemented for GPU");
  }

  if (x->requires_grad()) {
    z->autograd_meta().set_grad_fn(
        std::make_shared<SelectBackward>(z, x, index));
  }

  return z;
}

variable reshape(const variable& x, std::vector<size_t> shape) {
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

variable flatten(const variable& x) { return reshape(x, {x->size()}); }

variable sum(const variable& x, size_t axis) {
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