#ifndef UTILS_HPP
#define UTILS_HPP

#include <algorithm>
#include <iostream>
#include <memory>
#include <stack>
#include <tensorlib/types.hpp>
#include <unordered_set>
#include <vector>

// given an index in the flattened tensor, return the index in the tensor with
// given stride
size_t convert_to_index(size_t index, variable t);

// calculate size given shape
size_t calculate_size(const std::vector<size_t>& shape);

// check if two tensors have the same shape
void check_tensor_shape(const variable& x, const variable& y);
// check if two tensors are on the same device
void check_tensor_device(const variable& x, const variable& y);

node_list topological_sort(std::shared_ptr<Node> root);

#endif