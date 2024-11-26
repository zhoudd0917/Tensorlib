#ifndef TYPES_HPP
#define TYPES_HPP

#include <memory>
#include <vector>

class Tensor;
class Node;
class Edge;

using variable = std::shared_ptr<Tensor>;
using variable_list = std::vector<variable>;
using node_list = std::vector<std::shared_ptr<Node>>;
using edge_list = std::vector<Edge>;

// Enum for device type
enum class Device { CPU, GPU };

#endif
