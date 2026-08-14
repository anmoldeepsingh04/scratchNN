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

TEST(TensorTest, Addition){
    // 0D + 0D
    std::shared_ptr<Tensor> tensor1 = std::make_shared<Tensor>(1.0);
    std::shared_ptr<Tensor> tensor2 = std::make_shared<Tensor>(2.0);
    std::shared_ptr<Tensor> tensor3 = (*tensor1) + (tensor2);
    EXPECT_EQ(tensor3 -> item(), 3.0);

    // 0D + 1D
    std::shared_ptr<Tensor> tensor4 = std::make_shared<Tensor>(1.0);
    std::shared_ptr<Tensor> tensor5 = std::make_shared<Tensor>(std::vector<float>({2.0, 3.0, 4.0}));
    std::shared_ptr<Tensor> tensor6 = (*tensor4) + (tensor5);
    EXPECT_EQ(tensor6 -> shape(), std::vector<std::size_t>({3}));
    EXPECT_EQ((*tensor6)(0), 3.0);
    EXPECT_EQ((*tensor6)(1), 4.0);
    EXPECT_EQ((*tensor6)(2), 5.0);

    // 1D + 0D
    std::shared_ptr<Tensor> tensor7 = std::make_shared<Tensor>(std::vector<float>({2.0, 3.0, 4.0}));
    std::shared_ptr<Tensor> tensor8 = std::make_shared<Tensor>(1.0);
    std::shared_ptr<Tensor> tensor9 = (*tensor7) + (tensor8);
    EXPECT_EQ(tensor9 -> shape(), std::vector<std::size_t>({3}));
    EXPECT_EQ((*tensor9)(0), 3.0);
    EXPECT_EQ((*tensor9)(1), 4.0);
    EXPECT_EQ((*tensor9)(2), 5.0);

    // 1D + 1D
    std::shared_ptr<Tensor> tensor10 = std::make_shared<Tensor>(std::vector<float>({1.0, 2.0, 3.0}));
    std::shared_ptr<Tensor> tensor11 = std::make_shared<Tensor>(std::vector<float>({4.0, 5.0, 6.0}));
    std::shared_ptr<Tensor> tensor12 = (*tensor10) + (tensor11);
    EXPECT_EQ(tensor12 -> shape(), std::vector<std::size_t>({3}));
    EXPECT_EQ((*tensor12)(0), 5.0);
    EXPECT_EQ((*tensor12)(1), 7.0);
    EXPECT_EQ((*tensor12)(2), 9.0);

    // 2D + 2D
    std::shared_ptr<Tensor> tensor13 = std::make_shared<Tensor>(std::vector<std::vector<float>>({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}));
    std::shared_ptr<Tensor> tensor14 = std::make_shared<Tensor>(std::vector<std::vector<float>>({{7.0, 8.0, 9.0}, {10.0, 11.0, 12.0}}));
    std::shared_ptr<Tensor> tensor15 = (*tensor13) + (tensor14);
    EXPECT_EQ(tensor15 -> shape(), std::vector<std::size_t>({2, 3}));
    EXPECT_EQ((*tensor15)(0, 0), 8.0);
    EXPECT_EQ((*tensor15)(0, 1), 10.0);
    EXPECT_EQ((*tensor15)(0, 2), 12.0);
    EXPECT_EQ((*tensor15)(1, 0), 14.0);
    EXPECT_EQ((*tensor15)(1, 1), 16.0);
    EXPECT_EQ((*tensor15)(1, 2), 18.0);

    // 2D + 0D
    std::shared_ptr<Tensor> tensor16 = std::make_shared<Tensor>(std::vector<std::vector<float>>({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}));
    std::shared_ptr<Tensor> tensor17 = std::make_shared<Tensor>(7.0);
    std::shared_ptr<Tensor> tensor18 = (*tensor16) + (tensor17);
    EXPECT_EQ(tensor18 -> shape(), std::vector<std::size_t>({2, 3}));
    EXPECT_EQ((*tensor18)(0, 0), 8.0);
    EXPECT_EQ((*tensor18)(0, 1), 9.0);
    EXPECT_EQ((*tensor18)(0, 2), 10.0);
    EXPECT_EQ((*tensor18)(1, 0), 11.0);
    EXPECT_EQ((*tensor18)(1, 1), 12.0);
    EXPECT_EQ((*tensor18)(1, 2), 13.0);

    // 0D + 2D
    std::shared_ptr<Tensor> tensor19 = std::make_shared<Tensor>(7.0);
    std::shared_ptr<Tensor> tensor20 = std::make_shared<Tensor>(std::vector<std::vector<float>>({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}));
    std::shared_ptr<Tensor> tensor21 = (*tensor19) + (tensor20);
    EXPECT_EQ(tensor21 -> shape(), std::vector<std::size_t>({2, 3}));
    EXPECT_EQ((*tensor21)(0, 0), 8.0);
    EXPECT_EQ((*tensor21)(0, 1), 9.0);
    EXPECT_EQ((*tensor21)(0, 2), 10.0);
    EXPECT_EQ((*tensor21)(1, 0), 11.0);
    EXPECT_EQ((*tensor21)(1, 1), 12.0);
    EXPECT_EQ((*tensor21)(1, 2), 13.0);
}

TEST(TensorTest, Matmul){
    // 0D * 0D
    std::shared_ptr<Tensor> tensor1 = std::make_shared<Tensor>(1.0);
    std::shared_ptr<Tensor> tensor2 = std::make_shared<Tensor>(2.0);
    EXPECT_THROW((*tensor1) * tensor2, std::invalid_argument);

    // 0D * 1D
    std::shared_ptr<Tensor> tensor4 = std::make_shared<Tensor>(1.0);
    std::shared_ptr<Tensor> tensor5 = std::make_shared<Tensor>(std::vector<float>({2.0, 3.0, 4.0}));
    EXPECT_THROW((*tensor4) * tensor5, std::invalid_argument);

    // 1D * 1D with mismatched dimensions
    std::shared_ptr<Tensor> tensor6 = std::make_shared<Tensor>(std::vector<float>({1.0, 2.0, 3.0}));
    std::shared_ptr<Tensor> tensor7 = std::make_shared<Tensor>(std::vector<float>({4.0, 5.0}));
    EXPECT_THROW((*tensor6) * tensor7, std::invalid_argument);

    // 1D * 1D
    std::shared_ptr<Tensor> tensor10 = std::make_shared<Tensor>(std::vector<float>({1.0, 2.0, 3.0}));
    std::shared_ptr<Tensor> tensor11 = std::make_shared<Tensor>(std::vector<float>({4.0, 5.0, 6.0}));
    std::shared_ptr<Tensor> tensor12 = (*tensor10) * (tensor11);
    EXPECT_EQ(tensor12 -> shape(), std::vector<std::size_t>({}));
    EXPECT_EQ(tensor12 -> item(), 32.0);

    // 1D * 2D
    std::shared_ptr<Tensor> tensor13 = std::make_shared<Tensor>(std::vector<float>({1.0, 2.0}));
    std::shared_ptr<Tensor> tensor14 = std::make_shared<Tensor>(std::vector<std::vector<float>>({{4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}));
    std::shared_ptr<Tensor> tensor15 = (*tensor13) * (tensor14);
    EXPECT_EQ(tensor15 -> shape(), std::vector<std::size_t>({3}));
    EXPECT_EQ((*tensor15)(0), 18.0);
    EXPECT_EQ((*tensor15)(1), 21.0);
    EXPECT_EQ((*tensor15)(2), 24.0);

    // 2D * 1D
    std::shared_ptr<Tensor> tensor16 = std::make_shared<Tensor>(std::vector<std::vector<float>>({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}));
    std::shared_ptr<Tensor> tensor17 = std::make_shared<Tensor>(std::vector<float>({1.0, 2.0, 3.0}));
    std::shared_ptr<Tensor> tensor18 = (*tensor16) * (tensor17);
    EXPECT_EQ(tensor18 -> shape(), std::vector<std::size_t>({2}));
    EXPECT_EQ((*tensor18)(0), 14.0);
    EXPECT_EQ((*tensor18)(1), 32.0);

    // 2D * 2D with mismatched dimensions
    std::shared_ptr<Tensor> tensor19 = std::make_shared<Tensor>(std::vector<std::vector<float>>({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}));
    std::shared_ptr<Tensor> tensor20 = std::make_shared<Tensor>(std::vector<std::vector<float>>({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}));
    EXPECT_THROW((*tensor19) * tensor20, std::invalid_argument);

    // 2D * 2D
    std::shared_ptr<Tensor> tensor21 = std::make_shared<Tensor>(std::vector<std::vector<float>>({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}));
    std::shared_ptr<Tensor> tensor22 = std::make_shared<Tensor>(std::vector<std::vector<float>>({{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}}));
    std::shared_ptr<Tensor> tensor23 = (*tensor21) * (tensor22);
    EXPECT_EQ(tensor23 -> shape(), std::vector<std::size_t>({2, 2}));
    EXPECT_EQ((*tensor23)(0, 0), 58.0);
    EXPECT_EQ((*tensor23)(0, 1), 64.0);
    EXPECT_EQ((*tensor23)(1, 0), 139.0);
    EXPECT_EQ((*tensor23)(1, 1), 154.0);

    // 2D * 2D with large size
    std::shared_ptr<Tensor> tensor24 = std::make_shared<Tensor>(std::vector<std::vector<float>>(200, std::vector<float>(300, 1.0)));
    std::shared_ptr<Tensor> tensor25 = std::make_shared<Tensor>(std::vector<std::vector<float>>(300, std::vector<float>(400, 1.0)));
    std::shared_ptr<Tensor> tensor26 = (*tensor24) * (tensor25);
    EXPECT_EQ(tensor26 -> shape(), std::vector<std::size_t>({200, 400}));
    for(std::size_t i = 0; i < 200; i++){
        for(std::size_t j = 0; j < 400; j++){
            EXPECT_EQ((*tensor26)(i, j), 300.0);
        }
    }
}
