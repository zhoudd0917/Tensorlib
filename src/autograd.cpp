#include "autograd.hpp"

#include "tensor.cuh"

AutogradMeta::AutogradMeta(Tensor* self) {
  grad_ = std::make_shared<Tensor>(self->shape(), self->device(), false);
}