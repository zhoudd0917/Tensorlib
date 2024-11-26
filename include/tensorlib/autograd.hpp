#ifndef AUTOGRAD_HPP
#define AUTOGRAD_HPP

#include <memory>
#include <tensorlib/types.hpp>

class Node;
class Tensor;

// Metadata for autograd
class AutogradMeta {
 public:
  AutogradMeta(Tensor* self);
  void set_grad_fn(std::shared_ptr<Node> grad_fn) { grad_fn_ = grad_fn; }

  variable grad_;
  std::shared_ptr<Node> grad_fn_;
};

#endif