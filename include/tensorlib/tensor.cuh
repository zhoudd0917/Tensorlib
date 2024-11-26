#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <memory>
#include <string>
#include <tensorlib/types.hpp>
#include <vector>

class AutogradMeta;

class Tensor {
 private:
  float* data_;                 // Raw data pointer
  Device device_;               // Device type (CPU or GPU)
  std::vector<size_t> shape_;   // Shape of the tensor
  std::vector<size_t> stride_;  // Stride of the tensor
  bool requires_grad_;          // Gradient tracking flag
  std::shared_ptr<AutogradMeta> autograd_meta_;

 public:
  // Constructors
  Tensor(std::vector<float> data, std::vector<size_t> shape = {},
         Device device = Device::CPU, bool requires_grad = false);
  Tensor(std::vector<size_t> shape, Device device = Device::CPU,
         bool requires_grad = false);

  // Destructor
  ~Tensor();

  // Accessors
  float* data() { return data_; }
  const std::vector<size_t>& shape() const { return shape_; }
  const std::vector<size_t>& stride() const { return stride_; }
  size_t size() const {
    size_t size = 1;
    for (auto& s : shape_) size *= s;
    return size;
  }
  Device device() const { return device_; }
  void to_device(Device device);

  // Autograd
  void set_requires_grad(bool requires_grad);
  bool requires_grad() const { return requires_grad_; }
  AutogradMeta& autograd_meta() const;
  std::shared_ptr<Tensor> grad() const;
  void set_grad(std::shared_ptr<Tensor> grad);
  void backward(std::shared_ptr<Tensor> grad);

  // Utility
  void zero_();
  std::string to_string() const;
};

#endif
