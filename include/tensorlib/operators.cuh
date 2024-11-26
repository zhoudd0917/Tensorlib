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
variable select_idx(variable x, size_t index);
variable reshape(variable x, std::vector<size_t> shape);
variable flatten(variable x);
variable broadcast_to(variable x, std::vector<size_t> shape);
variable sum(variable x, size_t axis);
variable sum(variable x);
variable mean(variable x, size_t axis);
variable mean(variable x);
variable max(variable x, size_t axis);
variable max(variable x);
variable min(variable x, size_t axis);
variable min(variable x);

#endif