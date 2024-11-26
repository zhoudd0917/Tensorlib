# TensorLib

- [x] Implement computational graph.
- [x] Implement backprop by extending computational graph.
- [x] Fix memory leakage due to circular dependency (sharedptr).
- [x] CPU implementation for most tensor functions.
- [ ] GPU implementation for most tensor functions.
- [ ] Simple neural network implementation.

## Running Instructions

### Python

First run
```bash
sh build.sh
```
which creates a build directory with the .so file, then set the PythonPath:
```bash
export PYTHONPATH=$(pwd)/build:$PYTHONPATH
```
Afterwards, you can use tensorlib by `import tensorlib`, see [example.py](example/example.py).

### C++(Cuda)

For any c++ file, simply link the nessesary libraries:
```sh
nvcc example/example.cpp -Iinclude/ -Lbuild/ -ltensorlib_cpp -lopenblas -lcudart -lcublas -o example/example
```
and then run:
```sh
./example/example
```
see [example.cpp](example/example.cpp) for a use case.

## Project Structure

- `include/`: Header files.
- `src/`: Source files.
- `example/`: Example files.
- `build/`: Build directory (created by `build.sh`).
- `build.sh`: Build script.

## Dependencies

- [CMake](https://cmake.org/), for building the project.
- [Python](https://www.python.org/), for running the Python code.
- [OpenMP](https://www.openmp.org/), for parallelizing the CPU code.
- [OpenBLAS](https://www.openblas.net/), for CPU implementation of many matrix operations.
- [CUDA](https://developer.nvidia.com/cuda-downloads), for GPU implementation of many operations.
- [CuBLAS](https://developer.nvidia.com/cublas), for GPU implementation of many matrix operations.