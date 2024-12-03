#ifndef NODE_HPP

#define NODE_HPP

#include <memory>
#include <string>
#include <tensorlib/types.hpp>
#include <vector>

class Edge;
class Node;
class Tensor;

struct Edge {
  std::shared_ptr<Node> next;
};

// Node class that represents the backward functions in the computation graph
class Node : std::enable_shared_from_this<Node> {
 public:
  virtual void apply() = 0;
  void set_next_edges();

  edge_list& next_edges() { return next_edges_; }
  const std::string& name() { return name_; }
  variable_list& inputs() { return inputs_; }
  std::weak_ptr<Tensor> output() { return output_; }
  void set_name(const std::string& name) { name_ = name; }

 protected:
  // output from the last node, used to access last gradient
  std::weak_ptr<Tensor> output_;
  // inputs for the function
  variable_list inputs_;
  // next nodes in the graph
  edge_list next_edges_;
  // name of the node
  std::string name_ = "Node";
};

class AddBackward : public Node {
 public:
  AddBackward(variable output, variable x, variable y);
  void apply() override;
};

class SubBackward : public Node {
 public:
  SubBackward(variable output, variable x, variable y);
  void apply() override;
};

class MulBackward : public Node {
 public:
  MulBackward(variable output, variable x, variable y);
  void apply() override;
};

class DivBackward : public Node {
 public:
  DivBackward(variable output, variable x, variable y);
  void apply() override;
};

class NegateBackward : public Node {
 public:
  NegateBackward(variable output, variable x);
  void apply() override;
};

class MatmulBackward : public Node {
 public:
  MatmulBackward(variable output, variable x, variable y);
  void apply() override;
};

class TransposeBackward : public Node {
 public:
  TransposeBackward(variable output, variable x);
  void apply() override;
};

class LogBackward : public Node {
 public:
  LogBackward(variable output, variable x);
  void apply() override;
};

class ExpBackward : public Node {
 public:
  ExpBackward(variable output, variable x);
  void apply() override;
};

class SinBackward : public Node {
 public:
  SinBackward(variable output, variable x);
  void apply() override;
};

class CosBackward : public Node {
 public:
  CosBackward(variable output, variable x);
  void apply() override;
};

class ReluBackward : public Node {
 public:
  ReluBackward(variable output, variable x);
  void apply() override;
};

class SigmoidBackward : public Node {
 public:
  SigmoidBackward(variable output, variable x);
  void apply() override;
};

class SelectBackward : public Node {
 public:
  SelectBackward(variable output, variable x, size_t index);
  void apply() override;

 private:
  size_t index_;
};

class ReshapeBackward : public Node {
 public:
  ReshapeBackward(variable output, variable x);
  void apply() override;
};

class BroadcastBackward : public Node {
 public:
  BroadcastBackward(variable output, variable x);
  void apply() override;
};

class SumBackward : public Node {
 public:
  SumBackward(variable output, variable x, size_t axis, float factor = 1.0);
  void apply() override;

 private:
  size_t axis_;
  float factor_;
};

class SumAllBackward : public Node {
 public:
  SumAllBackward(variable output, variable x, float factor = 1.0);
  void apply() override;

 private:
  float factor_;
};

// Backward for functions that reduce the dimension of the input into a scalar
// by selecting a specific index
// eg. max, min
class SelectorBackward : public Node {
 public:
  SelectorBackward(variable output, variable x, size_t axis,
                   size_t* index_list);
  ~SelectorBackward();
  void apply() override;

 private:
  size_t axis_;
  size_t* index_list_;
  Device device_;
};

class SelectAllBackward : public Node {
 public:
  SelectAllBackward(variable output, variable x, size_t* index);
  ~SelectAllBackward();
  void apply() override;

 private:
  size_t* index_;
  Device device_;
};

class SoftmaxBackward : public Node {
 public:
  SoftmaxBackward(variable output, variable x, size_t axis);
  void apply() override;

 private:
  size_t axis_;
};

class CrossEntropyBackward : public Node {
 public:
  CrossEntropyBackward(variable output, variable x, variable y);
  void apply() override;
};

#endif