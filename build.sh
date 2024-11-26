#!/bin/bash
# Build the project into python module

mkdir -p build
cd build
cmake ..
make -j4
cd ..

echo "Finished build"