#include "nn/tensor.h"
#include <gtest/gtest.h>

TEST(TensorTest, Creation){
    // test for scalar
    Tensor tensor = Tensor(5.0);
    EXPECT_EQ(tensor.shape(), std::vector<std::size_t>({}));
    EXPECT_THROW(tensor(0), std::invalid_argument);
    EXPECT_EQ(tensor.item(), 5.0);

    // test for 1D tensor
    std::vector<float> v1 = {1.0, 2.0, 3.0};
    Tensor tensor1 = Tensor(v1);
    EXPECT_EQ(tensor1.shape(), std::vector<std::size_t>({3}));
    EXPECT_EQ(tensor1(0), 1.0);
    EXPECT_EQ(tensor1(1), 2.0);
    EXPECT_EQ(tensor1(2), 3.0);
    EXPECT_THROW(tensor1(3), std::invalid_argument);
    EXPECT_THROW(tensor1.item(), std::runtime_error);

    // test for 2D tensor
    std::vector<std::vector<float>> v2 = {{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}};
    Tensor tensor2 = Tensor(v2);
    EXPECT_EQ(tensor2.shape(), std::vector<std::size_t>({3, 3}));
    EXPECT_EQ(tensor2.stride(), std::vector<std::size_t>({3, 1}));
    EXPECT_EQ(tensor2(0,0), 1.0);
    EXPECT_EQ(tensor2(0,1), 2.0);
    EXPECT_EQ(tensor2(0,2), 3.0);
    EXPECT_EQ(tensor2(1,0), 1.0);
    EXPECT_EQ(tensor2(1,1), 2.0);
    EXPECT_EQ(tensor2(1,2), 3.0);
    EXPECT_EQ(tensor2(2,0), 1.0);
    EXPECT_EQ(tensor2(2,1), 2.0);
    EXPECT_EQ(tensor2(2,2), 3.0);
    EXPECT_THROW(tensor2(3,0), std::invalid_argument);
    EXPECT_THROW(tensor2(0,3), std::invalid_argument);
    EXPECT_THROW(tensor2.item(), std::runtime_error);
}

