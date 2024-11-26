#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tensorlib/operators.cuh>
#include <tensorlib/tensor.cuh>

namespace py = pybind11;

PYBIND11_MODULE(tensorlib, m) {
  py::enum_<Device>(m, "Device")
      .value("CPU", Device::CPU)
      .value("GPU", Device::GPU)
      .export_values();

  py::class_<Tensor, variable>(m, "Tensor")
      .def(py::init<std::vector<float>, std::vector<size_t>, Device, bool>(),
           py::arg("data"), py::arg("shape") = std::vector<size_t>{},
           py::arg("device") = Device::CPU, py::arg("requires_grad") = false)
      .def(py::init<std::vector<size_t>, Device, bool>(), py::arg("shape"),
           py::arg("device") = Device::CPU, py::arg("requires_grad") = false)
      .def("data",
           [](Tensor& self) {
             return py::array_t<float>(
                 self.size(), self.data(),
                 py::capsule(self.data(), [](void* p) {}));
           })
      .def_property_readonly("shape", &Tensor::shape)
      .def_property_readonly("stride", &Tensor::stride)
      .def("size", &Tensor::size)
      .def("device", &Tensor::device)
      .def("to_device", &Tensor::to_device)
      .def_property("requires_grad", &Tensor::requires_grad,
                    &Tensor::set_requires_grad)
      .def_property("grad", &Tensor::grad, &Tensor::set_grad)
      .def("backward", &Tensor::backward)
      .def("zero_", &Tensor::zero_)
      .def("__add__", [](const std::shared_ptr<Tensor>& a,
                         const std::shared_ptr<Tensor>& b) { return a + b; })
      .def("__repr__", &Tensor::to_string);

  // Operators and utility functions
  m.def(
      "add", [](const variable& x, const variable& y) { return x + y; },
      "Addition");
  m.def(
      "subtract", [](const variable& x, const variable& y) { return x - y; },
      "Subtraction");
  m.def(
      "multiply", [](const variable& x, const variable& y) { return x * y; },
      "Multiplication");
  m.def(
      "divide", [](const variable& x, const variable& y) { return x / y; },
      "Division");
  m.def("matmul", &matmul, "Matrix multiplication");

  m.def("log", [](const variable& x) { return log(x); }, "Logarithm");
  m.def("exp", [](const variable& x) { return exp(x); }, "Exponential");
  m.def("sin", [](const variable& x) { return sin(x); }, "Exponential");
  m.def("cos", [](const variable& x) { return cos(x); }, "Exponential");
  m.def("transpose", &transpose, "Transpose");
  m.def("relu", &relu, "ReLU");
  m.def("select_idx", &select_idx, "Select index");
  m.def("reshape", &reshape, "Reshape");
  m.def("flatten", &flatten, "Flatten");
  m.def("sum", &sum, "Sum along axis");
}
