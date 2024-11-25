#include "utils.hpp"

#include <algorithm>

#include "tensor.cuh"

size_t convert_to_index(size_t index, variable t) {
  const std::vector<size_t>&shape = t->shape(), &stride = t->stride();
  size_t remainder = index, result = 0;
  for (int dim = shape.size() - 1; dim >= 0; --dim) {
    int coord = remainder % shape[dim];
    remainder /= shape[dim];

    result += coord * stride[dim];
  }
  return result;
}

void check_tensor_shape(const variable& x, const variable& y) {
  if (x->shape() != y->shape()) {
    std::cerr << "Shape mismatch, x shape: ";
    for (auto& s : x->shape()) {
      std::cerr << s << " ";
    }
    std::cerr << ", y shape: ";
    for (auto& s : y->shape()) {
      std::cerr << s << " ";
    }
    std::cerr << std::endl;
    throw std::runtime_error("Shape mismatch");
  }
}

void check_tensor_device(const variable& x, const variable& y) {
  if (x->device() != y->device()) {
    throw std::runtime_error("Device mismatch");
  }
}

void dfs_topological_sort(std::shared_ptr<Node> node,
                          std::unordered_set<std::shared_ptr<Node>>& visited,
                          node_list& sorted_nodes) {
  visited.insert(node);

  for (auto& edge : node->next_edges()) {
    auto neighbor = edge.next;
    if (visited.find(neighbor) == visited.end()) {
      dfs_topological_sort(neighbor, visited, sorted_nodes);
    }
  }

  sorted_nodes.push_back(node);
}

node_list topological_sort(std::shared_ptr<Node> root) {
  node_list sorted_nodes;
  std::unordered_set<std::shared_ptr<Node>> visited;

  dfs_topological_sort(root, visited, sorted_nodes);

  std::reverse(sorted_nodes.begin(), sorted_nodes.end());

  return sorted_nodes;
}
