#pragma once
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class Tensor{
    private:
        std::vector<float> _data;
        std::vector<std::size_t> _shape;
        std::vector<std::size_t> _stride;

    public:
        Tensor(float data); // scalar value
        Tensor(std::vector<float> data); // 1D tensor == a vector
        Tensor(std::vector<std::vector<float>> data); // 2D tensor == a matrix
        const float &item() const; // a read variant which returns a constant
        float &item(); // a write variant which allows us to modify the values stored in the tensor

        // we can use square brackts to access elements in pytorch but cpp only supports overloading the square brackets with a single index, and we also need support for multidimensional indexing, we will overload the call operator. The following are headers for 1D and 2D accessors
        const float &operator()(std::size_t i) const;
        float &operator() (std::size_t i);
        const float &operator() (std::size_t i, std::size_t j) const;
        float &operator() (std::size_t i, std::size_t j);

        const std::vector<std::size_t> &shape() const;
        const std::vector<std::size_t> &stride() const;
        friend std::ostream &operator<<(std::ostream &os, const Tensor &obj);
};