#ifndef TENSOR_FACTORY_HPP
#define TENSOR_FACTORY_HPP

#include <memory>
#include <tensorlib/tensor.cuh>
#include <tensorlib/types.hpp>
#include <vector>

class TensorFactory {
 public:
  // Factory methods

  // Create a tensor from data with optional shape (1d), device and gradient
  // flag
  static variable create(std::vector<float> data,
                         std::vector<size_t> shape = {},
                         Device device = Device::CPU,
                         bool requires_grad = false);
  // Create a scalar tensor from data with optional device and gradient flag
  static variable create(float data, Device device = Device::CPU,
                         bool requires_grad = false);
  // Create a zero tensor from shape with optional device and gradient flag
  static variable zeros(std::vector<size_t> shape, Device device = Device::CPU,
                        bool requires_grad = false);
  // Helper function to initialize random normal data
  static variable randn(std::vector<size_t> shape, float mean = 0.0,
                        float stddev = 1.0, Device device = Device::CPU,
                        bool requires_grad = false);
};

#endif