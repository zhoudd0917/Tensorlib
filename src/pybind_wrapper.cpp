#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void scale_vector_with_cublas(float* x, int n, float alpha);

void scale_vector(py::array_t<float> x, float alpha) {
  // Get the size of the vector
  int n = x.size();

  scale_vector_with_cublas(x.mutable_data(), n, alpha);
}

PYBIND11_MODULE(tensorlib, m) {
  m.def("scale_vector", &scale_vector,
        "Scale a vector by a given scalar using cuBLAS");
}