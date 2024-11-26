#include <random>
#include <tensorlib/tensor_factory.hpp>
#include <tensorlib/utils.hpp>

variable TensorFactory::create(std::vector<float> data,
                               std::vector<size_t> shape, Device device,
                               bool requires_grad) {
  return std::make_shared<Tensor>(data, shape, device, requires_grad);
}

variable TensorFactory::create(float value, Device device, bool requires_grad) {
  return std::make_shared<Tensor>(
      std::vector<float>{value}, std::vector<size_t>{1}, device, requires_grad);
}

variable TensorFactory::zeros(std::vector<size_t> shape, Device device,
                              bool requires_grad) {
  return std::make_shared<Tensor>(shape, device, requires_grad);
}

variable TensorFactory::randn(std::vector<size_t> shape, float mean_val,
                              float std_val, Device device,
                              bool requires_grad) {
  size_t size = calculate_size(shape);
  std::vector<float> data(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(mean_val, std_val);

  for (size_t i = 0; i < size; ++i) {
    data[i] = dist(gen);
  }

  return std::make_shared<Tensor>(data, shape, device, requires_grad);
}