#include "nn/modules/loss.h"
#include "nn/modules/module.h"
#include "nn/modules/softmax.h"
#include "nn/tensor.h"

std::shared_ptr<Tensor> Loss::forward(std::shared_ptr<Tensor> input){
    throw std::runtime_error("Loss expects an input and target.");
}

std::shared_ptr<Tensor> Loss::forward(std::shared_ptr<Tensor> input, std::size_t target){
    throw std::runtime_error("Forward not implemented.");
}

std::shared_ptr<Tensor> Loss::operator()(std::shared_ptr<Tensor> input, std::size_t target){
    return forward(input, target);
}

std::shared_ptr<Tensor> NegativeLogLikelihood::forward(std::shared_ptr<Tensor> input, std::size_t target){
    if(input -> shape().size() != 1){
        throw std::runtime_error("Negative log-likelihood loss expects a 1D input tensor.");
    }
    if(target >= input -> numel()){
        throw std::runtime_error("Negative log-likelihood loss target out of bounds.");
    }
    // log(0) check
    float prob = std::max((*input)(target), 1e-12f);
    float loss = -std::log(prob);
    if(input -> requires_grad()){
        std::vector<std::shared_ptr<Tensor>> parents {input};
        std::function<void(const std::vector<float>&)> gradfn = [input, target](const std::vector<float>& grad_output){
            std::vector<float> grad_input;
            for(std::size_t i = 0; i < input -> numel(); i++){
                if(i == target){
                    grad_input.push_back(grad_output[0] * (-1.0f / (*input)(i)));
                }
                else{
                    grad_input.push_back(0.0f);
                }
            }
            input -> add_to_grad(grad_input);
        };
        return std::make_shared<Tensor>(loss, true, gradfn, parents);
    }
    return std::make_shared<Tensor>(loss);
}

std::shared_ptr<Tensor> CrossEntropyLoss::forward(std::shared_ptr<Tensor> input, std::size_t target){
    if(input -> shape().size() != 1){
        throw std::runtime_error("Cross-Entropy loss expects a 1D input tensor.");
    }
    if(target >= input -> numel()){
        throw std::runtime_error("Cross-Entropy loss target out of bounds.");
    }
    Softmax softmax;
    NegativeLogLikelihood nll_loss;
    std::shared_ptr<Tensor> softmax_output = softmax(input);
    return nll_loss(softmax_output, target);
}