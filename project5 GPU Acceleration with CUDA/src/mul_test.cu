#include "matrix.h"
#include <gtest/gtest.h>

// 暴力矩阵乘法
bool mul(const Matrix<float> &lhs, const Matrix<float> &rhs, Matrix<float> &dst)
{
    if (lhs.data == nullptr || rhs.data == nullptr || dst.data == nullptr)
    {
        std::cerr << "Null pointer.\n";
        return false;
    }

    if (lhs.cols != rhs.rows)
    {
        std::cerr << "Incompatible dimensions for multiplication: A.cols != B.rows\n";
        return false;
    }
    if (dst.rows != lhs.rows || dst.cols != rhs.cols)
    {
        std::cerr << "Output matrix dimensions do not match the product dimensions.\n";
        return false;
    }

    for (int i = 0; i < lhs.rows; i++)
    {
        for (int j = 0; j < rhs.cols; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < lhs.cols; k++)
            {
                sum += lhs.data[i * lhs.cols + k] * rhs.data[k * rhs.cols + j];
            }
            dst.data[i * dst.cols + j] = sum;
        }
    }

    return true;
}


class MatrixMultiplicationTest : public ::testing::Test {
protected:
    Matrix<float> A, B, resultCPU, resultGPU, expected;

    MatrixMultiplicationTest() 
        : A(256, 128), B(128, 256), resultCPU(256, 256), resultGPU(256, 256), expected(256, 256) {
    }

    void SetUp() override {
        A.randomize();
        B.randomize();

        mul(A, B, expected);
    }

    void TearDown() override {
    }
};

// 测试 CPU 计算结果是否正确
TEST_F(MatrixMultiplicationTest, TestCPUImplementation) {
    ASSERT_TRUE(mulMatrixCPU(A, B, resultCPU));
    for (size_t i = 0; i < resultCPU.rows; i++) {
        for (size_t j = 0; j < resultCPU.cols; j++) {
            EXPECT_NEAR(resultCPU.data[i * resultCPU.cols + j], expected.data[i * expected.cols + j], 1e-4);
        }
    }
}

// 测试 GPU 计算结果是否正确
TEST_F(MatrixMultiplicationTest, TestGPUImplementation) {
    ASSERT_TRUE(mulMatrixGPU(A, B, resultGPU));
    for (size_t i = 0; i < resultGPU.rows; i++) {
        for (size_t j = 0; j < resultGPU.cols; j++) {
            EXPECT_NEAR(resultGPU.data[i * resultGPU.cols + j], expected.data[i * expected.cols + j], 1e-4);
        }
    }
}

// 测试输入矩阵为空指针的情况
TEST_F(MatrixMultiplicationTest, NullPointerTest) {
    Matrix<float> nullMatrix(256, 128);
    nullMatrix.data = nullptr;  // 故意将数据指针设置为nullptr

    // 测试 CPU 实现
    EXPECT_FALSE(mulMatrixCPU(nullMatrix, B, resultCPU));
    EXPECT_FALSE(mulMatrixCPU(A, nullMatrix, resultCPU));
    EXPECT_FALSE(mulMatrixCPU(A, B, nullMatrix));

    // 测试 GPU 实现
    EXPECT_FALSE(mulMatrixGPU(nullMatrix, B, resultGPU));
    EXPECT_FALSE(mulMatrixGPU(A, nullMatrix, resultGPU));
    EXPECT_FALSE(mulMatrixGPU(A, B, nullMatrix));
}

// 测试维度不匹配的情况
TEST_F(MatrixMultiplicationTest, DimensionMismatchTest) {
    Matrix<float> wrongDimsMatrix(256, 256);  // 错误的维度

    // 测试 CPU 实现
    EXPECT_FALSE(mulMatrixCPU(wrongDimsMatrix, B, resultCPU));  // 维度 A.cols != B.rows
    EXPECT_FALSE(mulMatrixCPU(A, wrongDimsMatrix, resultCPU));  // 输出维度不匹配

    // 测试 GPU 实现
    EXPECT_FALSE(mulMatrixGPU(wrongDimsMatrix, B, resultGPU));
    EXPECT_FALSE(mulMatrixGPU(A, wrongDimsMatrix, resultGPU));
}

// 测试输出维度不匹配的情况
TEST_F(MatrixMultiplicationTest, OutputDimensionMismatchTest) {
    Matrix<float> wrongOutputDimsMatrix(128, 128);  // 错误的输出维度

    // 测试 CPU 实现
    EXPECT_FALSE(mulMatrixCPU(A, B, wrongOutputDimsMatrix));

    // 测试 GPU 实现
    EXPECT_FALSE(mulMatrixGPU(A, B, wrongOutputDimsMatrix));
}