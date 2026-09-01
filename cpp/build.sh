#!/bin/bash

rm -rf build
mkdir build
cd build
cmake ..
make -j$(sysctl -n hw.ncpu)

echo ""
echo "===== BUILD COMPLETE ====="
echo ""

cd ..
./build/neural_cpp
