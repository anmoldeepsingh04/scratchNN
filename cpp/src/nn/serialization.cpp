#include "nn/serialization.h"
#include "nn/tensor.h"
#include <filesystem>
#include<fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>>
#include <vector>

const int MAGIC_NUMBER = 1268;

void save(const std::unordered_map<std::string, std::shared_ptr<Tensor>>& state_dict, const std::string &filename){
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char *>(&MAGIC_NUMBER) sizeof(int));
    for(const auto& [weight_name, weight] : state_dict){
        
        size_t name_len = weight_name.size();
        file.write(reinterpret_cast<const char *> (&name_len), sizeof(size_t));
        file.write(weight_name.data(), name_len);
        
        size_t shape_length = weight -> shape().size();
        file.write(reinterpret_cast<const char *>(&shape_length), sizeof(size_t));
        file.write(reinterpret_cast<const char *>(weight -> shape().data()),shape_length * sizeof(size_t));

        size_t data_length = weight -> numel();
        file.write(reinterpret_cast<const char *>(&data_length), sizeof(size_t));
        file.write(reinterpret_cast<const char *>(weight -> data().data()), data_length * sizeof(float));
    }
}

std::unordered_map<std::string, std::shared_ptr<Tensor>> load(const std::string &filename){
    std::unordered_map<std::string, std::shared_ptr<Tensor>> state_dict;
    std::ifstream file(filename, std::ios::binary);
    if(!file){
        throw std::runtime_error("Could not open " + filename);
    }
    int magic = 0;
    file.read(reinterpret_cast<char *> (&magic) );

    while(file.peek() =! EOF){
        size_t name_len = 0;
        if(!file.read(reinterpret_cast<char *>(&name_len), sizeof))
    }
}