#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <tensorlib/tensorlib.hpp>

namespace py = pybind11;

PYBIND11_MODULE(tensorlib, m) {
  py::enum_<Device>(m, "Device")
      .value("CPU", Device::CPU)
      .value("GPU", Device::GPU)
      .export_values();

  py::class_<Tensor, variable>(m, "Tensor")
      .def(py::init([](py::array_t<float> arr, Device device,
                       bool requires_grad) {
             auto buf = arr.request();
             float *ptr = static_cast<float *>(buf.ptr);
             std::vector<float> data(ptr, ptr + buf.size);
             std::vector<size_t> shape(buf.ndim);
             for (size_t i = 0; i < buf.ndim; i++) {
               shape[i] = buf.shape[i];
             }
             return TensorFactory::create(data, shape, device, requires_grad);
           }),
           py::arg("data"), py::arg("device") = Device::CPU,
           py::arg("requires_grad") = false)
      .def(py::init([](float data, Device device, bool requires_grad) {
             return TensorFactory::create(data, device, requires_grad);
           }),
           py::arg("data"), py::arg("device") = Device::CPU,
           py::arg("requires_grad") = false)
      .def("to_numpy",
           [](variable self) {
             if (self->device() == Device::GPU) {
               variable cpu_tensor = self->to_device(Device::CPU);
               return py::array_t<float>(cpu_tensor->shape(),
                                         cpu_tensor->data());
             }
             return py::array_t<float>(self->shape(), self->data());
           })
      .def("item", &Tensor::item)
      .def_property_readonly("shape", &Tensor::shape)
      .def_property_readonly("stride", &Tensor::stride)
      .def("size", &Tensor::size)
      .def("device", &Tensor::device)
      .def("to_device", &Tensor::to_device)
      .def_property("requires_grad", &Tensor::requires_grad,
                    &Tensor::set_requires_grad)
      .def_property("grad", &Tensor::grad, &Tensor::set_grad)
      .def("backward", &Tensor::backward,
           py::arg("grad") = TensorFactory::ones({1}))
      .def("zero_", &Tensor::zero_)
      .def("__add__", [](variable x, variable y) { return x + y; })
      .def("__add__", [](variable x, float y) { return x + y; })
      .def("__radd__", [](variable x, float y) { return x + y; })
      .def("__sub__", [](variable x, variable y) { return x - y; })
      .def("__sub__", [](variable x, float y) { return x - y; })
      .def("__rsub__", [](variable y, float x) { return x - y; })
      .def("__mul__", [](variable x, variable y) { return x * y; })
      .def("__mul__", [](variable x, float y) { return x * y; })
      .def("__rmul__", [](variable y, float x) { return x * y; })
      .def("__matmul__", [](variable x, variable y) { return matmul(x, y); })
      .def("__truediv__", [](variable x, variable y) { return x / y; })
      .def("__truediv__", [](variable x, float y) { return x / y; })
      .def("__rtruediv__", [](variable y, float x) { return x / y; })
      .def("__neg__", [](variable x) { return -x; })
      .def("__getitem__", &select_idx)
      .def("__repr__", &Tensor::to_string);

  // Create new tensors
  m.def("randn", &TensorFactory::randn, py::arg("shape"), py::arg("mean") = 0.f,
        py::arg("std") = 1.f, py::arg("seed") = -1,
        py::arg("device") = Device::CPU, py::arg("requires_grad") = false,
        "Create a tensor with random values");
  m.def("zeros", &TensorFactory::zeros, py::arg("shape"),
        py::arg("device") = Device::CPU, py::arg("requires_grad") = false,
        "Create a tensor with zeros");
  m.def("ones", &TensorFactory::ones, py::arg("shape"),
        py::arg("device") = Device::CPU, py::arg("requires_grad") = false,
        "Create a tensor with ones");

  // Operators and utility functions
  m.def("add", [](variable x, variable y) { return x + y; }, "Addition");
  m.def(
      "subtract", [](variable x, variable y) { return x - y; }, "Subtraction");
  m.def(
      "multiply", [](variable x, variable y) { return x * y; },
      "Multiplication");
  m.def("divide", [](variable x, variable y) { return x / y; }, "Division");
  m.def("matmul", &matmul, "Matrix multiplication");

  m.def("log", [](variable x) { return log(x); }, "Logarithm");
  m.def("exp", [](variable x) { return exp(x); }, "Exponential");
  m.def("sin", [](variable x) { return sin(x); }, "Exponential");
  m.def("cos", [](variable x) { return cos(x); }, "Exponential");
  m.def("transpose", &transpose, "Transpose");
  m.def("relu", &relu, "ReLU");
  m.def("sigmoid", &sigmoid, "Sigmoid");
  m.def("select_idx", &select_idx, "Select index");
  m.def("reshape", &reshape, "Reshape");
  m.def("flatten", &flatten, "Flatten");
  m.def(
      "sum",
      [](variable x, size_t idx, bool keepdims) {
        return sum(x, idx, keepdims);
      },
      py::arg("tensor"), py::arg("axis"), py::arg("keepdims") = false,
      "Sum along axis");
  m.def(
      "sum", [](variable x, bool keepdims) { return sum(x, keepdims); },
      py::arg("tensor"), py::arg("keepdims") = false, "Sum whole tensor");
  m.def(
      "mean",
      [](variable x, size_t idx, bool keepdims) {
        return mean(x, idx, keepdims);
      },
      py::arg("tensor"), py::arg("axis"), py::arg("keepdims") = false,
      "Mean along axis");
  m.def(
      "mean", [](variable x, bool keepdims) { return mean(x, keepdims); },
      py::arg("tensor"), py::arg("keepdims") = false, "Mean whole tensor");
  m.def(
      "max",
      [](variable x, size_t idx, bool keepdims) {
        return max(x, idx, keepdims);
      },
      py::arg("tensor"), py::arg("axis"), py::arg("keepdims") = false,
      "Max along axis");
  m.def(
      "max", [](variable x, bool keepdims) { return max(x, keepdims); },
      py::arg("tensor"), py::arg("keepdims") = false, "Max whole tensor");
  m.def(
      "min",
      [](variable x, size_t idx, bool keepdims) {
        return min(x, idx, keepdims);
      },
      py::arg("tensor"), py::arg("axis"), py::arg("keepdims") = false,
      "Min along axis");
  m.def(
      "min", [](variable x, bool keepdims) { return min(x, keepdims); },
      py::arg("tensor"), py::arg("keepdims") = false, "Min whole tensor");
  m.def(
      "argmax", [](variable x, size_t axis) { return argmax(x, axis); },
      py::arg("tensor"), py::arg("axis"), "Argmax along axis");
  m.def(
      "argmin", [](variable x, size_t axis) { return argmin(x, axis); },
      py::arg("tensor"), py::arg("axis"), "Argmin along axis");
  m.def("softmax", &softmax, py::arg("tensor"), py::arg("axis"),
        "Softmax along axis");
  m.def("cross_entropy", &cross_entropy, py::arg("x"), py::arg("y"),
        "Cross entropy loss");
  py::class_<NoGradScope>(m, "no_grad")
      .def(py::init<>())
      .def("__enter__",
           [](NoGradScope &self) { return &self; })  // Enter context
      .def("__exit__", [](NoGradScope &self, py::object, py::object,
                          py::object) {});  // Exit context
}
