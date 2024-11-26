# TensorLib

- [x] Implement computational graph.
- [x] Implement backprop by extending computational graph.
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