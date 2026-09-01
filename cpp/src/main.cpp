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

class NeuralNetwork : public Module{
    private:
        // layers of the network
        std::shared_ptr<Flatten> _flatten = std::make_shared<Flatten>();
        std::shared_ptr<Linear> _linear1 = std::make_shared<Linear>(28*28, 512);
        std::shared_ptr<Linear> _linear2 = std::make_shared<Linear>(512, 512);
        std::shared_ptr<Linear> _linear3 = std::make_shared<Linear>(512, 10);

        // activation functions
        std::shared_ptr<Relu> _relu = std::make_shared<Relu>();

    public:
        NeuralNetwork(){
            register_module("linear_1", _linear1);
            register_module("linear_2", _linear2);
            register_module("linear_3", _linear3);
        }
        std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input){
            std::shared_ptr<Tensor> flat = (*_flatten)(input);
            std::shared_ptr<Tensor> linear1 = (*_linear1)(flat);
            std::shared_ptr<Tensor> relu1 = (*_relu)(linear1);
            std::shared_ptr<Tensor> linear2 = (*_linear2)(relu1);
            std::shared_ptr<Tensor> relu2 = (*_relu)(linear2);
            std::shared_ptr<Tensor> linear3 = (*_linear3)(relu2);
            return linear3;
        }
};

void train(Dataloader& dataloader, NeuralNetwork& model, CrossEntropyLoss& loss_fxn, SGD& optimizer){
    std::size_t log_interval = 100;
    std::size_t batch_n = 0;
    std::size_t seen_samples = 0;

    for(const auto& batch : dataloader){
        std::shared_ptr<Tensor> total_loss = nullptr;
        std::size_t batch_size = batch.size();

        for(const auto& [label, tensor] : batch){
            auto output = model(tensor);
            auto loss = loss_fxn(output, label);
            if(total_loss == nullptr){
                total_loss = loss;
            }
            else{
                total_loss = (*total_loss) + loss;
            }
            seen_samples += 1;
        }
        total_loss -> item() /= batch_size;

        if(batch_n % log_interval == 0){
            std::cout<<"Loss: "<<std::fixed<<std::setprecision(6)<<total_loss->item()<<" ["<<seen_samples<<"/"<<dataloader.n_samples()<<"]"<<std::endl;
        }

        total_loss -> backward();
        optimizer.step();
        optimizer.zero_grad();
        batch_n += 1;
    }
}

void test(Dataloader& dataloader, NeuralNetwork& model, CrossEntropyLoss& loss_fxn){
    float running_loss = 0.0f;
    std::size_t correct = 0;
    std::size_t n_samples = 0;

    for(const auto& batch : dataloader){
        for(const auto& [label, tensor] : batch){
            auto output = model(tensor);
            // accuracy
            if(output -> argmax() == label){
                correct += 1;
            }
            running_loss += loss_fxn(output, label) -> item();
            n_samples += 1;
    }
}
    float accuracy = static_cast<float>(correct) / static_cast<float>(n_samples);
    float avg_loss = running_loss / n_samples;

    std::cout<<std::fixed<<std::setprecision(6)<<"Test error: \n accuracy: "<<std::setprecision(1)<<accuracy * 100.0<<"%\n"<<" average loss: "<<std::setprecision(6)<<avg_loss<<"\n";
}

void train_new_mnist_model(){
    std::cout<<"Loading dataset..."<<std::endl;
    MNIST mnist_train = MNIST("data/MNIST/raw/train-images-idx3-ubyte", "data/MNIST/raw/train-labels-idx1-ubyte");
    MNIST mnist_test = MNIST("data/MNIST/raw/t10k-images-idx3-ubyte", "data/MNIST/raw/t10k-labels-idx1-ubyte");
    std::cout<<"Dataset loaded!"<<std::endl;

    int batch_size = 10;
    Dataloader train_dataloader(&mnist_train, batch_size);
    Dataloader test_dataloader(&mnist_test, batch_size);

    NeuralNetwork model;
    CrossEntropyLoss loss_fxn;
    float learning_rate = 0.001f;
    SGD optimizer(model.parameters(), learning_rate);

    // training loop
    int n_epochs = 1;
    for(int epoch = 0; epoch < n_epochs; epoch++){
        std::cout<<"[Epoch "<<epoch<<"/"<<n_epochs<<"] Training..."<<std::endl;
        train(train_dataloader, model, loss_fxn, optimizer);
        std::cout<<"[Epoch "<<epoch<<"/"<<n_epochs<<"] Testing..."<<std::endl;
        test(test_dataloader, model, loss_fxn);
    }

    auto state_dict = model.state_dict();
    save(state_dict, "model/mnist.nn");
}

void inference_on_saved_model(){
    NeuralNetwork model;
    std::cout<<"Loading model..."<<std::endl;
    auto loaded_state_dict = load("models/mnist.nn");
    model.load_state_dict(loaded_state_dict);

    std::cout<<"Loading test set..."<<std::endl;
    MNIST mnist_test = MNIST("data/MNIST/raw/t10k-images-idx3-ubyte", "data/MNIST/raw/t10k-labels-idx1-ubyte");

    int n_samples = 10;

    std::vector<int> all_indices(mnist_test.get_length());
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(all_indices.begin(), all_indices.end(), g);
    std::vector<int> indices(all_indices.begin(), all_indices.begin() + n_samples);

    for(int i = 0; i < n_samples; i++){
        std::cout<<"Sample "<<i<<" of "<<n_samples<<std::endl;
        std::pair<int, std::shared_ptr<Tensor>> sample_image = mnist_test.get_item(indices[i]);
        visualize_image(sample_image.second);
        auto output = model(sample_image.second);
        int predicted_class = output -> argmax();
        std::cout<<"Predicted class: "<<mnist_test.label_to_class(predicted_class)<<std::endl;
        std::cout<<"Actual class: "<<mnist_test.label_to_class(sample_image.first)<<std::endl;
        std::cout<<"---------------------------------------------------------------"<<std::endl;
    }
}

int main(){
    train_new_mnist_model();
}