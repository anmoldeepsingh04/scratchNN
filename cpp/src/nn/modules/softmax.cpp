#include "nn/modules/softmax.h"
#include "nn/tensor.h"
#include <memory>
#include <random>

std::shared_ptr<Tensor> Softmax::forward(std::shared_ptr<Tensor> input){
    // scalar
    if(input -> shape().size() == 0){
        float result = 1.0f;
        if(input -> requires_grad()){
            std::vector<std::shared_ptr<Tensor>> parents{input};
            std::function<void(const std::vector<float>&)> gradfn = [input](const std::vector<float>& grad_output){
                // differentiating a scalar gives 0
                std::vector<float> grad_input = {0.0f};
                input -> add_to_grad(grad_input);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result); 
    }

    // 1D
    if(input -> shape().size() == 1){
        // subtracting the maximums value to avoid overflow
        float max_val = (*input)(0);
        for(int i = 0; i < input -> numel(); i++){
            if((*input)(i) > max_val){
                max_val = (*input)(i);
            }
        }
        std::vector<float> s;
        float sum_exp = 0.0f;
        for(std::size_t i; i < input -> numel(); i++){
            sum_exp += std::exp((*input)(i) - max_val);
        }
        for(std::size_t i; i< input -> numel(); i++){
            s.push_back((std::exp((*input)(i) - max_val) / sum_exp));
        }
        if(input -> requires_grad()){
            std::vector<std::shared_ptr<Tensor>> parents{input};
            std::function<void(const std::vector<float>&)>gradfn = [input, s](const std::vector<float>& grad_output){
                std::vector<float> grad_input;
                for(std::size_t j = 0; j < input -> numel(); j++){
                    float grad_j = 0.0f;
                    for(std::size_t i = 0; i < grad_output.size(); i++){
                        if(i == j){
                            grad_j += (grad_output[i] * (s[i] * (1-s[i])));
                        }
                        else{
                            grad_j += (grad_output[i] * (-s[i] * s[j]));
                        }
                    }
                    grad_input.push_back(grad_j);
                }
                input -> add_to_grad(grad_input);
            };
            return std::make_shared<Tensor>(s, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(s);
    }
    else{
        throw std::runtime_error("Softmax is only allowed for 1D vectors.");
    }
};