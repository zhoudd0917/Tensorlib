#!/bin/bash
# Build the project into python module

mkdir -p build
cd build
cmake ..
make
cd ..

echo "Finished build"