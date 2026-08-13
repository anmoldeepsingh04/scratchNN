#include "nn/tensor.h"
#include <iostream>
#include <string>
#include <vector>

Tensor::Tensor(float data): _data{data}, _shape{}, _stride{} {};
Tensor::Tensor(std::vector<float> data): _data{data}, _shape{data.size()}, _stride{1} {};
// here we can't directly assign the data directly as our data member is private. We will flatten the multidimensional input to a single dimension using row/column major ordering. PyTorch is row-major by default, and we will do the same
Tensor::Tensor(std::vector<std::vector<float>> data): _shape{data.size(), data[0].size()}, _stride{data[0].size(), 1} {

// check if dimensions match, #rows == #cols
    std::size_t n_expected_columns = data[0].size();
    for(std::size_t i = 0; i < data.size(); i++){
        if (data[i].size() != n_expected_columns){
            throw std::invalid_argument("Dimensions are inconsistent.");
        }
    }


    // store data in row-major format
    for(std::size_t i = 0; i < data.size(); i++){
        for(std::size_t j = 0; j < data[i].size(); j++){
            _data.push_back(data[i][j]);
        }
    }
}

// we also implement an item method which in case of a scalar/1D vector with single element, returns a reference to that element, thus, prohibiting us from indexing in case of scalars
const float &Tensor::item() const{
    // works only if the tensor is a scalar or 1D
    if(_data.size() == 1){
        return _data[0];
    }
    else{
        throw std::runtime_error("item() can only be called on tensors with a single element");
    }
}

// implementing functionalities for 1D and 2D accessors
// 1D tensor
const float &Tensor::operator()(std::size_t i) const {
    // checking if the tensor is not 0D
    if (_shape.size() == 0){
        throw std::invalid_argument("Can't index into a scalar. Use item() instead");
    }
    // checking if we're not accessing any out of bounds element
    if(_shape.size() == 1){
        if (i >= _shape[0]){
            throw std::invalid_argument("Index " + std::to_string(i) + " is out of bounds for array of size " + std::to_string(_shape[0]));
        }
        return _data[i];
    }
    // throwing error if the tensor is 2D
    throw std::invalid_argument("This is a 1D tensor. Use two indices for 2D tensors.");
}

// 2D tensor: we have to map the row index i and column index j into an index for our linear representation, which is done using the stride variable which decides how many elements do we have to move to get to the next element in a certain dimension.
const float &Tensor::operator()(std::size_t i, std::size_t j) const {
    // checking if we're not accessing any out of bounds element
    if(_shape.size() == 2){
        if (i >= _shape[0]){
            throw std::invalid_argument("Row index " + std::to_string(i) + " is out of bounds for tensor with " + std::to_string(_shape[0]) + " rows");
        }
        if (j >= _shape[1]){
            throw std::invalid_argument("Column index " + std::to_string(j) + " is out of bounds for tensor with " + std::to_string(_shape[1]) + " columns");
        }
        return _data[i * _stride[0] + j * _stride[1]];
    }
    // throwing error if the tensor is not 2D
    throw std::invalid_argument("Can only double index into 2D tensors.");
}

// methods to view the stride and shape of tensor
const std::vector<std::size_t> &Tensor::shape() const {return _shape;}
const std::vector<std::size_t> &Tensor::stride() const {return _stride;}

// constructing a tensor: defining a custom visualization method to see the data
std::ostream &operator<<(std::ostream &os, const Tensor &obj){
    std::string string_repr = "[";
    if(obj.shape().size() == 0){
        os << obj.item();
        return os;
    }
    else if (obj.shape().size() == 1){
        for(std::size_t i = 0; i < obj.shape()[0]; i++){
            string_repr += std::to_string(obj(i));
            if( i != obj.shape()[0] - 1){
                string_repr += ", ";
            }
        }
        string_repr += "]";
    }
    else{
        for(std::size_t i = 0; i < obj.shape()[0]; i++){
            string_repr += "[";
            for(std::size_t j = 0; j < obj.shape()[1]; j++){
                string_repr += std::to_string(obj(i,j));
                if (j != obj.shape()[1] - 1){
                    string_repr += ", ";
                }
            }
            string_repr += "]";
            if(i != obj.shape()[0] - 1){
                string_repr += ", ";
            }
        }
        string_repr += "]";
    }
    os << string_repr;
    return os;
}