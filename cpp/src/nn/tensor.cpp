#include "nn/tensor.h"
#include <iostream>
#include <string>
#include <vector>

Tensor::Tensor(float data, bool requires_grad,
                std::function<void(const std::vector<float>& )> gradfn,
                std::vector<std::shared_ptr<Tensor>> parents): _data{data}, _shape{}, _stride{}, _requires_grad(requires_grad), _gradfn(gradfn,
                _parents(parents)) {
                    if(_requires_grad){
                        zero_grad();
                    }
                }
Tensor::Tensor(std::vector<float> data, bool requires_grad,
                std::function<void(const std::vector<float>&)> gradfn,
                std::vector<std::sharedptr<Tensor>> parents): _data{data},
                _shape{data.size()}, _stride{1}, _requires_grad(requires_grad), _gradfn(gradfn),
                _parents(parents){
                    if(_requires_grad){
                        zero_grad();
                    }
                }
// here we can't directly assign the data directly as our data member is private. We will flatten the multidimensional input to a single dimension using row/column major ordering. PyTorch is row-major by default, and we will do the same
Tensor::Tensor(std::vector<std::vector<float>> data, bool requires_grad,
                std::function<void(const std::vector<float>&)> gradfn,
                std::vector<std::sharedptr<Tensor>> parents):
                _shape{data.size(), data[0].size()}, _stride{data[0].size(), 1}, _requires_grad(requires_grad), _gradfn(gradfn),
                _parents(parents){

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
    if(_requires_grad){
        zero_grad();
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

float &Tensor::item(){
    if(_data.size() == 1){
        return _data[0];
    }
    else{
        throw std::runtime_error(
            "item() can only be called on tensors with a single element"
        );
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

float &Tensor::operator()(std::size_t i) {
    if (_shape.size() == 0){
        throw std::invalid_argument(
            "Can't index into a scalar. Use item() instead"
        );
    }

    if (_shape.size() == 1){
        if (i >= _shape[0]){
            throw std::invalid_argument(
                "Index " + std::to_string(i) +
                " is out of bounds for array of size " +
                std::to_string(_shape[0])
            );
        }

        return _data[i];
    }

    throw std::invalid_argument(
        "This is a 1D tensor. Use two indices for 2D tensors."
    );
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

float &Tensor::operator()(std::size_t i, std::size_t j) {
    if (_shape.size() == 2){
        if (i >= _shape[0]){
            throw std::invalid_argument(
                "Row index " + std::to_string(i) +
                " is out of bounds for tensor with " +
                std::to_string(_shape[0]) + " rows"
            );
        }

        if (j >= _shape[1]){
            throw std::invalid_argument(
                "Column index " + std::to_string(j) +
                " is out of bounds for tensor with " +
                std::to_string(_shape[1]) + " columns"
            );
        }

        return _data[i * _stride[0] + j * _stride[1]];
    }

    throw std::invalid_argument(
        "Can only double index into 2D tensors."
    );
}

// methods to view the stride and shape of tensor
const std::vector<std::size_t> &Tensor::shape() const {return _shape;}
const std::vector<std::size_t> &Tensor::stride() const {return _stride;}

// method to see if _requirs_grad variable is true
const bool &Tensor::requires_grad() const{return _requirs_grad;}
// method to get the grad variable which is storing the gradients
const std::vector<float> &Tensor::grad() const {return _grad;}
// method to add a gradient update to our current gradients
void Tensor::add_to_grad(const std::vector<float>& grad_update){
    if(!_requires_grad){
        return;
    }
    if(_grad.size() != grad_update.size()){
        throw std::runtime_error("Gradient shape mismatch during accumulation.");
    }
    for(std::size_t i = 0; i < grad.size(); i++){
        _grad[i] += grad_update[i];
    }
}
// method to initialize gradients to a vector of the same size as data we're storing
void Tensor::zero_grad(){ _grad = std::vector<float>(_data.size(), 0.0f);}
// numel method like pytorch that tells us how many elements we're storing in data
std::size_t Tensor::numel() const{ return _data.size();}


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

// implementing the overloaded + operator for tensors
std::shared_ptr<Tensor> Tensor::operator+(std::shared_ptr<Tensor> other){
    // 0D + 0D
    if(_shape.size() == 0 && other -> shape().size() == 0){
        float result = item() + other -> item();
        return std::make_shared<Tensor>(result);
    }

    // 0D + 1D
    if(_shape.size() == 0 && other -> shape().size() == 1){
        std::vector<float> result;
        for(std::size_t i = 0; i < other -> shape()[0]; i++){
            result.push_back(item() + (*other)(i));
        }
        return std::make_shared<Tensor>(result);
    }

    // 0D + 2D
    if(_shape.size() == 0 && other -> shape().size() == 2){
        std::vector<std::vector<float>> result;
        for(std::size_t i = 0; i < other -> shape()[0]; i++){
            std::vector<float> result_i;
            
            for (std::size_t j = 0; j < other -> shape()[1]; j++){
                result_i.push_back(item() + (*other)(i,j));
            }
            result.push_back(result_i);
        }
        return std::make_shared<Tensor>(result);
    }

    // 1D + 0D
    if(_shape.size() == 1 && other -> shape().size() == 0){
        std::vector<float> result;
        for(std::size_t i = 0; i < shape()[0]; i++){
            result.push_back(operator()(i) + other -> item());
        }
        return std::make_shared<Tensor>(result);
    }

    // 2D + 0D
    if(_shape.size() == 2 && other -> shape().size() == 0){
        std::vector<std::vector<float>> result;
        for(std::size_t i = 0; i < shape()[0]; i++){
            std::vector<float> result_i;
            
            for (std::size_t j = 0; j < shape()[1]; j++){
                result_i.push_back(operator()(i, j) + other -> item());
            }
            result.push_back(result_i);
        }
        return std::make_shared<Tensor>(result);
    }

    // 1D + 1D
    // checking if both dimensions are equal
    if(_shape[0] != other -> shape()[0]){
        throw std::invalid_argument("First dimensions are not equal.");
    }

    // adding both 1D tensors
    if(_shape.size() == 1){
        std::vector<float> result;
        for(std::size_t i = 0; i < shape()[0]; i++){
            result.push_back(operator()(i) + (*other)(i));
        }
        return std::make_shared<Tensor>(result);
    }

    // 2D + 2D
    // checking if both second dimensions are equal
    else{
        if(shape()[1] != other -> shape()[1]){
            throw std::invalid_argument("Second dimensions are not equal.");
        }

        // adding both 2D tensors
        std::vector<std::vector<float>> result;
        for(std::size_t i = 0; i < shape()[0]; i++){
            std::vector<float> result_i;
            for(std::size_t j = 0; j < shape()[1]; j++){
                result_i.push_back(operator()(i, j) + (*other)(i, j));
            }
            result.push_back(result_i);
        }
        return std::make_shared<Tensor>(result);
    }
}

// implementing the overloaded * operator for tensors: Pytorch has * for elementwise multiplication and @ for matrix multiplication, but since @ is not really an operator in c++, we will overload the * operator
std::shared_ptr<Tensor> Tensor::operator*(std::shared_ptr<Tensor> other){
    // matmul is not defined for two scalars, so check for that
    if(_shape.size() == 0 || other -> shape().size() == 0){
        throw std::invalid_argument("Both arguments need to be atleast 1D for matrix multiplication.");
    }

    // checking if the #columns of first tensor is equal to #rows of second tensor
    if(_shape[_shape.size() - 1] != other -> shape()[0]){
        throw std::invalid_argument("Number of columns of first tensor is not equal to number of rows of second tensor.");
    }

    // 1D x 1D -> 0D
    if(_shape.size() == 1 && other -> shape().size() == 1){
        float result = 0.0;
        for(std::size_t i = 0; i < _shape[0]; i++){
            result += operator()(i) * (*other)(i);
        }
        return std::make_shared<Tensor>(result);
    }

    // 2D x 1D -> 1D
    else if(_shape.size() == 2 && other -> shape().size() == 1){
        std::vector<float> result;
        for(std::size_t i = 0; i < _shape[0]; i++){
            float result_i = 0.0;
            for (std::size_t j = 0; j < _shape[1]; j++){
                result_i += operator()(i, j) * (*other)(j);
            }
            result.push_back(result_i);
        }
        return std::make_shared<Tensor>(result);
    }

    // 1D x 2D -> 1D
    else if(_shape.size() == 1 && other -> shape().size() == 2){
        std::vector<float> result;
        for(std::size_t i = 0; i < other -> shape()[1]; i++){
            float result_i = 0.0;
            for (std::size_t j = 0; j < other -> shape()[0]; j++){
                result_i += operator()(j) * (*other)(j, i);
            }
            result.push_back(result_i);
        }
        return std::make_shared<Tensor>(result);
    }

    // 2D x 2D -> 2D
    else{
        if(other -> shape().size() < 2){
            throw std::invalid_argument("Expected second tensor to have at least two dimensions for matrix multiplication.");
        }
        std::vector<std::vector<float>> result;
        for(std::size_t i = 0; i < shape()[0]; i++){
            std::vector<float> result_i;
            for(std::size_t j = 0; j < other -> shape()[1]; j++){
                float result_i_j = 0.0;
                for(std::size_t k = 0; k < shape()[1]; k++){
                    result_i_j += operator()(i, k) * (*other)(k ,j);
                }
                result_i.push_back(result_i_j);
            }
            result.push_back(result_i);
        }
        return std::make_shared<Tensor>(result);
    }
}