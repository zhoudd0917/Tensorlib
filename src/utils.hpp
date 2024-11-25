#ifndef UTILS_HPP
#define UTILS_HPP

#include <algorithm>
#include <iostream>
#include <memory>
#include <stack>
#include <unordered_set>
#include <vector>

class Tensor;
class Node;
class Edge;

using variable = std::shared_ptr<Tensor>;
using variable_list = std::vector<variable>;
using node_list = std::vector<std::shared_ptr<Node>>;
using edge_list = std::vector<Edge>;

// given an index in the flattened tensor, return the index in the tensor with
// given stride
size_t convert_to_index(size_t index, variable t);

// check if two tensors have the same shape
void check_tensor_shape(const variable& x, const variable& y);
// check if two tensors are on the same device
void check_tensor_device(const variable& x, const variable& y);

node_list topological_sort(std::shared_ptr<Node> root);

#endif