#!/bin/bash

set -e

rm -rf build
mkdir build
cd build
cmake ..
make -j$(sysctl -n hw.ncpu)

echo ""
echo "===== BUILD COMPLETE ====="
echo ""

./neural_cpp
