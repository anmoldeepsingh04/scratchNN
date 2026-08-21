#pragma once
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class Tensor : public std::enable_shared_from_this<Tensor>{
    private:
        std::vector<float> _data;
        std::vector<std::size_t> _shape;
        std::vector<std::size_t> _stride;
        std::vector<float> _grad;
        std::function<void(const std::vector<float>&)> _gradfn;
        std::vector<std::shared_ptr<Tensor>> _parents;
        bool _requires_grad;

    public:
        Tensor(float data, bool requires_grad = false,
               std::function<void(const std::vector<float>&)> gradfn = nullptr,
               std::vector<std::shared_ptr<Tensor>> parents = {}); // scalar value
        Tensor(std::vector<float> data, bool requires_grad = false,
               std::function<void(const std::vector<float>&)> gradfn = nullptr,
               std::vector<std::shared_ptr<Tensor>> parents = {}); // 1D tensor == a vector
        Tensor(std::vector<std::vector<float>> data, bool requires_grad = false,
               std::function<void(const std::vector<float>&)> gradfn = nullptr,
               std::vector<std::shared_ptr<Tensor>> parents = {}); // 2D tensor == a matrix
        const float &item() const; // a read variant which returns a constant
        float &item(); // a write variant which allows us to modify the values stored in the tensor

        // we can use square brackts to access elements in pytorch but cpp only supports overloading the square brackets with a single index, and we also need support for multidimensional indexing, we will overload the call operator. The following are headers for 1D and 2D accessors
        const float &operator()(std::size_t i) const;
        float &operator() (std::size_t i);
        const float &operator() (std::size_t i, std::size_t j) const;
        float &operator() (std::size_t i, std::size_t j);

        // method to get the shape and stride
        const std::vector<std::size_t> &shape() const;
        const std::vector<std::size_t> &stride() const;

        // overloading the << operator to print data on the terminal
        friend std::ostream &operator<<(std::ostream &os, const Tensor &obj);

        // defining the only two operations required to train a neural network. We will overload the + and * operators
        // we will use a shared_ptr to overwrite the operators, which manages memory smartly and it is stored on the heap which is needed because to implement auto-differentiation we will use computational graphs where each tensor needs the pointers to the tensors that were used to create it
        std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> other);
        std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> other);
};