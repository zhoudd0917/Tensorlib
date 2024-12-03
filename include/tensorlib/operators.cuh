#ifndef OPERATORS_HPP
#define OPERATORS_HPP

#include <tensorlib/types.hpp>

class Tensor;

variable operator+(variable x, variable y);
variable operator+(variable x, float y);
variable operator+(float x, variable y);
variable operator-(variable x, variable y);
variable operator-(variable x, float y);
variable operator-(float x, variable y);
variable operator*(variable x, variable y);
variable operator*(variable x, float y);
variable operator*(float x, variable y);
variable operator/(variable x, variable y);
variable operator/(variable x, float y);
variable operator/(float x, variable y);
variable operator-(variable x);
variable matmul(variable x, variable y);
variable transpose(variable x);
variable log(variable x);
variable exp(variable x);
variable sin(variable x);
variable cos(variable x);
variable relu(variable x);
variable sigmoid(variable x);
variable select_idx(variable x, size_t index);
variable reshape(variable x, std::vector<size_t> shape);
variable flatten(variable x);
variable broadcast_to(variable x, std::vector<size_t> shape);
variable sum(variable x, size_t axis, bool keepdims = false);
variable sum(variable x, bool keepdims = false);
variable mean(variable x, size_t axis, bool keepdims = false);
variable mean(variable x, bool keepdims = false);
variable max(variable x, size_t axis, bool keepdims = false);
variable max(variable x, bool keepdims = false);
variable min(variable x, size_t axis, bool keepdims = false);
variable min(variable x, bool keepdims = false);
variable argmax(variable x, size_t axis, bool keepdims = false);
variable argmin(variable x, size_t axis, bool keepdims = false);
variable softmax(variable x, size_t axis);
// applies softmax then cross entropy to the input,
// x should be of size (batch_size, num_classes), y should be of size
// (batch_size, num_classes)
variable cross_entropy(variable x, variable y);

#endif