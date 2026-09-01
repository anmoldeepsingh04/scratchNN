#include "nn/data/dataloader.h"
#include "nn/data/datasets.h"
#include "nn/modules/flatten.h"
#include "nn/modules/linear.h"
#include "nn/modules/loss.h"
#include "nn/modules/module.h"
#include "nn/modules/relu.h"
#include "nn/serialization.h"
#include "nn/sgd.h"
#include "nn/tensor.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

int main(){
    MNIST mnist_train = MNIST("data/MNIST/raw/train-images-idx3-ubyte", "data/MNIST/raw/train-labels-idx1-ubyte");

    std::cout<<"Datasets successfully loaded!"<<std::endl;
    
    int batch_size = 10;
    Dataloader mnist_train_loader(&mnist_train, batch_size);

    std::cout<<"Visualizing first batch of training data."<<std::endl;
    
    for(auto batch : mnist_train_loader){
        std::cout << "Batch received\n";
        for(auto item : batch){
            visualize_image(item.second);
            std::cout<<mnist_train.label_to_class(item.first)<<std::endl;
        }
        break;
    }
    return 0;
}