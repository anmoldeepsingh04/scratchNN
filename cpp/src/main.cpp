#include "nn/tensor.h"
#include <iostream>
#include <vector>

int main(){
    std::vector<std::vector<float>> test_v{{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}};
    Tensor test_tensor(test_v);
    std::cout << test_tensor << std::endl;
}