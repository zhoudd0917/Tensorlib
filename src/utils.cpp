#include <algorithm>
#include <tensorlib/node.cuh>
#include <tensorlib/tensor.cuh>
#include <tensorlib/utils.hpp>

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

// calculate index after droping the axis dimension
size_t calculate_index_after_drop_axis(size_t index, size_t axis,
                                       const std::vector<size_t>& shape) {
  size_t newIndex = 0;
  size_t stride = 1;
  size_t nDims = shape.size();

  for (size_t i = nDims; i > 0; i--) {
    if (i - 1 != axis) {
      size_t dimSize = shape[i - 1];
      size_t currentDimIndex = index % dimSize;
      newIndex += currentDimIndex * stride;
      stride *= dimSize;
    }
    index /= shape[i - 1];
  }
  return newIndex;
}

size_t calculate_index_after_add_axis(size_t index, size_t axis,
                                      const std::vector<size_t>& shape) {
  size_t new_index = 0;
  size_t old_stride = 1, new_stride = 1;
  size_t nd = shape.size();

  for (size_t i = nd; i > 0; i--) {
    size_t dim_size = shape[i - 1];
    if (i - 1 != axis) {
      size_t c_i = index % dim_size;
      new_index += c_i * new_stride;
      new_stride *= dim_size;
      old_stride *= dim_size;
      index /= dim_size;
    } else {
      new_stride *= dim_size;
    }
  }
  return new_index;
}

size_t calculate_size(const std::vector<size_t>& shape) {
  size_t size = 1;
  for (auto& s : shape) {
    size *= s;
  }
  return size;
}

std::vector<size_t> calculate_strides(const std::vector<size_t>& shape) {
  std::vector<size_t> strides(shape.size(), 1);
  for (size_t i = shape.size() - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * shape[i];
  }
  return strides;
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
