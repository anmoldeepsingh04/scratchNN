#!/bin/bash
set -e

rm -rf build
mkdir build
cd build

cmake ..
make -j$(sysctl -n hw.ncpu)

cd ..
echo "===== BUILD COMPLETE ====="

./build/neural_cpp