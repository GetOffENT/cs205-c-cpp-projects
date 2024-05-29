#include "matrix.hpp"

#include <gtest/gtest.h>

#define CHECK_MATRIX_INIT_STATE(m)    \
    EXPECT_EQ(m.getCols(), 0);        \
    EXPECT_EQ(m.getRows(), 0);        \
    EXPECT_EQ(m.getChannels(), 1);    \
    EXPECT_EQ(m.getStep(), 0);        \
    EXPECT_EQ(m.getROICols(), 0);     \
    EXPECT_EQ(m.getROIRows(), 0);     \
    EXPECT_EQ(m.getROIStartCol(), 0); \
    EXPECT_EQ(m.getROIStartRow(), 0); \
    EXPECT_EQ(m.getData(), nullptr)

// 默认构造函数测试
TEST(ConstructorTest, DefaultConstructor)
{
    Matrix<unsigned char> m1;
    CHECK_MATRIX_INIT_STATE(m1);

    Matrix<short> m2;
    CHECK_MATRIX_INIT_STATE(m2);

    Matrix<int> m3;
    CHECK_MATRIX_INIT_STATE(m3);

    Matrix<float> m4;
    CHECK_MATRIX_INIT_STATE(m4);

    Matrix<double> m5;
    CHECK_MATRIX_INIT_STATE(m5);
}

#define CHECK_MATRIX_PARAMS(m, expected_rows, expected_cols, expected_channels, type)      \
    EXPECT_EQ(m.getRows(), expected_rows);                                                 \
    EXPECT_EQ(m.getCols(), expected_cols);                                                 \
    EXPECT_EQ(m.getChannels(), expected_channels);                                         \
    EXPECT_EQ(m.getStep(), (expected_cols * expected_channels * sizeof(type) + 63) & ~63); \
    EXPECT_EQ(m.getROIRows(), expected_rows);                                              \
    EXPECT_EQ(m.getROICols(), expected_cols);                                              \
    EXPECT_EQ(m.getROIStartCol(), 0);                                                      \
    EXPECT_EQ(m.getROIStartRow(), 0);                                                      \
    EXPECT_NE(m.getData(), nullptr)

// 参数化构造函数测试
TEST(ConstructorTest, ParameterizedConstructor)
{
    Matrix<unsigned char> m(10, 15, 3);
    CHECK_MATRIX_PARAMS(m, 10, 15, 3, unsigned char);

    Matrix<short> m2(10, 15, 3);
    CHECK_MATRIX_PARAMS(m2, 10, 15, 3, short);

    Matrix<int> m3(10, 15, 3);
    CHECK_MATRIX_PARAMS(m3, 10, 15, 3, int);

    Matrix<float> m4(10, 15, 3);
    CHECK_MATRIX_PARAMS(m4, 10, 15, 3, float);

    Matrix<double> m5(10, 15, 3);
    CHECK_MATRIX_PARAMS(m5, 10, 15, 3, double);
}

template <typename T>
bool compareMatrixData(const Matrix<T> &m, const std::vector<T> &expected)
{
    size_t idx = 0;
    for (size_t row = 0; row < m.getRows(); ++row)
    {
        for (size_t col = 0; col < m.getCols(); ++col)
        {
            for (size_t channel = 0; channel < m.getChannels(); ++channel)
            {
                if (m.at(row, col, channel) != expected[idx++])
                {
                    return false;
                }
            }
        }
    }
    return true;
}

// 二维数组构造函数测试
TEST(ConstructorTest, 2DArrayConstructor)
{
    Matrix<unsigned char> m({{1, 2, 3}, {4, 5, 6}, {9, 8, 7}});
    CHECK_MATRIX_PARAMS(m, 3, 3, 1, unsigned char);
    std::vector<unsigned char> expected = {1, 2, 3, 4, 5, 6, 9, 8, 7};
    EXPECT_TRUE(compareMatrixData(m, expected));

    Matrix<short> m2({{1, 2, 3}, {4, 5, 6}, {9, 8, 7}});
    CHECK_MATRIX_PARAMS(m2, 3, 3, 1, short);
    std::vector<short> expected2 = {1, 2, 3, 4, 5, 6, 9, 8, 7};
    EXPECT_TRUE(compareMatrixData(m2, expected2));

    Matrix<int> m3({{1, 2, 3}, {4, 5, 6}, {9, 8, 7}});
    CHECK_MATRIX_PARAMS(m3, 3, 3, 1, int);
    std::vector<int> expected3 = {1, 2, 3, 4, 5, 6, 9, 8, 7};
    EXPECT_TRUE(compareMatrixData(m3, expected3));

    Matrix<float> m4({{1.2f, 2.4f, 3.2f}, {4.1f, 5.5f, 6.8f}, {9.4f, 8.7f, 7.2f}});
    CHECK_MATRIX_PARAMS(m4, 3, 3, 1, float);
    std::vector<float> expected4 = {1.2f, 2.4f, 3.2f, 4.1f, 5.5f, 6.8f, 9.4f, 8.7f, 7.2f};
    EXPECT_TRUE(compareMatrixData(m4, expected4));

    Matrix<double> m5({{1.2, 2.4, 3.2}, {4.1, 5.5, 6.8}, {9.4, 8.7, 7.2}});
    CHECK_MATRIX_PARAMS(m5, 3, 3, 1, double);
    std::vector<double> expected5 = {1.2, 2.4, 3.2, 4.1, 5.5, 6.8, 9.4, 8.7, 7.2};
    EXPECT_TRUE(compareMatrixData(m5, expected5));
}

// 三维数组构造函数测试
TEST(ConstructorTest, 3DArrayConstructor)
{
    Matrix<unsigned char> m({{{1, 2, 3}, {4, 5, 6}, {9, 8, 7}}, {{3, 5, 2}, {1, 4, 6}, {7, 8, 9}}});
    CHECK_MATRIX_PARAMS(m, 2, 3, 3, unsigned char);
    std::vector<unsigned char> expected = {1, 2, 3, 4, 5, 6, 9, 8, 7, 3, 5, 2, 1, 4, 6, 7, 8, 9};
    EXPECT_TRUE(compareMatrixData(m, expected));

    Matrix<short> m2({{{1, 2, 3}, {4, 5, 6}, {9, 8, 7}}, {{3, 5, 2}, {1, 4, 6}, {7, 8, 9}}});
    CHECK_MATRIX_PARAMS(m2, 2, 3, 3, short);
    std::vector<short> expected2 = {1, 2, 3, 4, 5, 6, 9, 8, 7, 3, 5, 2, 1, 4, 6, 7, 8, 9};
    EXPECT_TRUE(compareMatrixData(m2, expected2));

    Matrix<int> m3({{{1, 2, 3}, {4, 5, 6}, {9, 8, 7}}, {{3, 5, 2}, {1, 4, 6}, {7, 8, 9}}});
    CHECK_MATRIX_PARAMS(m3, 2, 3, 3, int);
    std::vector<int> expected3 = {1, 2, 3, 4, 5, 6, 9, 8, 7, 3, 5, 2, 1, 4, 6, 7, 8, 9};
    EXPECT_TRUE(compareMatrixData(m3, expected3));

    Matrix<float> m4({{{1.2f, 2.4f, 3.2f}, {4.1f, 5.5f, 6.8f}, {9.4f, 8.7f, 7.2f}}, {{3.2f, 5.1f, 2.3f}, {1.4f, 4.6f, 6.7f}, {7.8f, 8.9f, 9.1f}}});
    CHECK_MATRIX_PARAMS(m4, 2, 3, 3, float);
    std::vector<float> expected4 = {1.2f, 2.4f, 3.2f, 4.1f, 5.5f, 6.8f, 9.4f, 8.7f, 7.2f, 3.2f, 5.1f, 2.3f, 1.4f, 4.6f, 6.7f, 7.8f, 8.9f, 9.1f};
    EXPECT_TRUE(compareMatrixData(m4, expected4));

    Matrix<double> m5({{{1.2, 2.4, 3.2}, {4.1, 5.5, 6.8}, {9.4, 8.7, 7.2}}, {{3.2, 5.1, 2.3}, {1.4, 4.6, 6.7}, {7.8, 8.9, 9.1}}});
    CHECK_MATRIX_PARAMS(m5, 2, 3, 3, double);
    std::vector<double> expected5 = {1.2, 2.4, 3.2, 4.1, 5.5, 6.8, 9.4, 8.7, 7.2, 3.2, 5.1, 2.3, 1.4, 4.6, 6.7, 7.8, 8.9, 9.1};
    EXPECT_TRUE(compareMatrixData(m5, expected5));
}

// 拷贝构造函数测试
TEST(ConstructorTest, CopyConstructor)
{
    Matrix<unsigned char> m({{1, 2, 3}, {4, 5, 6}, {9, 8, 7}});
    Matrix<unsigned char> m2(m);
    CHECK_MATRIX_PARAMS(m2, 3, 3, 1, unsigned char);
    std::vector<unsigned char> expected = {1, 2, 3, 4, 5, 6, 9, 8, 7};
    EXPECT_TRUE(compareMatrixData(m2, expected));

    Matrix<short> m3({{1, 2, 3}, {4, 5, 6}, {9, 8, 7}});
    Matrix<short> m4(m3);
    CHECK_MATRIX_PARAMS(m4, 3, 3, 1, short);
    std::vector<short> expected2 = {1, 2, 3, 4, 5, 6, 9, 8, 7};
    EXPECT_TRUE(compareMatrixData(m4, expected2));

    Matrix<int> m5({{1, 2, 3}, {4, 5, 6}, {9, 8, 7}});
    Matrix<int> m6(m5);
    CHECK_MATRIX_PARAMS(m6, 3, 3, 1, int);
    std::vector<int> expected3 = {1, 2, 3, 4, 5, 6, 9, 8, 7};
    EXPECT_TRUE(compareMatrixData(m6, expected3));

    Matrix<float> m7({{1.2f, 2.4f, 3.2f}, {4.1f, 5.5f, 6.8f}, {9.4f, 8.7f, 7.2f}});
    Matrix<float> m8(m7);
    CHECK_MATRIX_PARAMS(m8, 3, 3, 1, float);
    std::vector<float> expected4 = {1.2f, 2.4f, 3.2f, 4.1f, 5.5f, 6.8f, 9.4f, 8.7f, 7.2f};
    EXPECT_TRUE(compareMatrixData(m8, expected4));

    Matrix<double> m9({{1.2, 2.4, 3.2}, {4.1, 5.5, 6.8}, {9.4, 8.7, 7.2}});
    Matrix<double> m10(m9);
    CHECK_MATRIX_PARAMS(m10, 3, 3, 1, double);
    std::vector<double> expected5 = {1.2, 2.4, 3.2, 4.1, 5.5, 6.8, 9.4, 8.7, 7.2};
    EXPECT_TRUE(compareMatrixData(m10, expected5));
}

// 移动构造函数测试
TEST(ConstructorTest, MoveConstructor)
{
    Matrix<unsigned char> m({{1, 2, 3}, {4, 5, 6}, {9, 8, 7}});
    Matrix<unsigned char> m2(std::move(m));
    CHECK_MATRIX_PARAMS(m2, 3, 3, 1, unsigned char);
    std::vector<unsigned char> expected = {1, 2, 3, 4, 5, 6, 9, 8, 7};
    EXPECT_TRUE(compareMatrixData(m2, expected));
    CHECK_MATRIX_INIT_STATE(m);

    Matrix<short> m3({{1, 2, 3}, {4, 5, 6}, {9, 8, 7}});
    Matrix<short> m4(std::move(m3));
    CHECK_MATRIX_PARAMS(m4, 3, 3, 1, short);
    std::vector<short> expected2 = {1, 2, 3, 4, 5, 6, 9, 8, 7};
    EXPECT_TRUE(compareMatrixData(m4, expected2));
    CHECK_MATRIX_INIT_STATE(m3);

    Matrix<int> m5({{1, 2, 3}, {4, 5, 6}, {9, 8, 7}});
    Matrix<int> m6(std::move(m5));
    CHECK_MATRIX_PARAMS(m6, 3, 3, 1, int);
    std::vector<int> expected3 = {1, 2, 3, 4, 5, 6, 9, 8, 7};
    EXPECT_TRUE(compareMatrixData(m6, expected3));
    CHECK_MATRIX_INIT_STATE(m5);

    Matrix<float> m7({{1.2f, 2.4f, 3.2f}, {4.1f, 5.5f, 6.8f}, {9.4f, 8.7f, 7.2f}});
    Matrix<float> m8(std::move(m7));
    CHECK_MATRIX_PARAMS(m8, 3, 3, 1, float);
    std::vector<float> expected4 = {1.2f, 2.4f, 3.2f, 4.1f, 5.5f, 6.8f, 9.4f, 8.7f, 7.2f};
    EXPECT_TRUE(compareMatrixData(m8, expected4));
    CHECK_MATRIX_INIT_STATE(m7);

    Matrix<double> m9({{1.2, 2.4, 3.2}, {4.1, 5.5, 6.8}, {9.4, 8.7, 7.2}});
    Matrix<double> m10(std::move(m9));
    CHECK_MATRIX_PARAMS(m10, 3, 3, 1, double);
    std::vector<double> expected5 = {1.2, 2.4, 3.2, 4.1, 5.5, 6.8, 9.4, 8.7, 7.2};
    EXPECT_TRUE(compareMatrixData(m10, expected5));
    CHECK_MATRIX_INIT_STATE(m9);
}

// 检查参数
#define CHECK_ROI_PARAMS(roi, base, expected_rows, expected_cols, expected_channels, expected_start_row, expected_start_col) \
    EXPECT_EQ(roi.getRows(), base.getRows());                                                                                \
    EXPECT_EQ(roi.getCols(), base.getCols());                                                                                \
    EXPECT_EQ(roi.getChannels(), expected_channels);                                                                         \
    EXPECT_EQ(roi.getStep(), base.getStep());                                                                                \
    EXPECT_EQ(roi.getROIStartCol(), expected_start_col);                                                                     \
    EXPECT_EQ(roi.getROIStartRow(), expected_start_row);                                                                     \
    EXPECT_EQ(roi.getROICols(), expected_cols);                                                                              \
    EXPECT_EQ(roi.getROIRows(), expected_rows);                                                                              \
    EXPECT_EQ(roi.getData(), base.getData());

// ROI构造函数测试
TEST(ConstructorTest, ROIConstructor)
{
    Matrix<unsigned char> base(10, 15, 3);       // 创建一个基础矩阵，大小为10x15，3个通道
    Matrix<unsigned char> roi(base, 2, 3, 5, 7); // 创建一个起始于(2, 3)的5x7的ROI区域
    CHECK_ROI_PARAMS(roi, base, 5, 7, 3, 2, 3);

    Matrix<unsigned char> base2(10, 10, 1); // 创建一个基础矩阵，大小为10x10，1个通道
    // 检查构造一个超出基础矩阵范围的ROI是否抛出异常
    EXPECT_THROW(Matrix<unsigned char> roi(base2, 8, 8, 5, 5), std::out_of_range);
}



template <typename T>
bool compareMatrixData(const Matrix<T> &m, const Matrix<T> &expected)
{
    size_t idx = 0;
    for (size_t row = 0; row < m.getRows(); ++row)
    {
        for (size_t col = 0; col < m.getCols(); ++col)
        {
            for (size_t channel = 0; channel < m.getChannels(); ++channel)
            {
                if (m.at(row, col, channel) != expected.at(row, col, channel))
                {
                    return false;
                }
            }
        }
    }
    return true;
}


TEST(MatrixAdditionTest, MatrixPlusMatrix_Int) {
    Matrix<int> m1({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
    Matrix<int> m2({{{6, 5}, {4, 3}}, {{2, 1}, {0, -1}}});
    Matrix<int> expected({{{7, 7}, {7, 7}}, {{7, 7}, {7, 7}}});

    auto result = m1 + m2;
    EXPECT_TRUE(compareMatrixData(result, expected));
}

TEST(MatrixAdditionTest, MatrixPlusMatrix_Float) {
    Matrix<float> m1({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}});
    Matrix<float> m2({{{6.0, 5.0}, {4.0, 3.0}}, {{2.0, 1.0}, {0.0, -1.0}}});
    Matrix<float> expected({{{7.0, 7.0}, {7.0, 7.0}}, {{7.0, 7.0}, {7.0, 7.0}}});

    auto result = m1 + m2;
    EXPECT_TRUE(compareMatrixData(result, expected));
}

TEST(MatrixAdditionTest, MatrixPlusScalar_Int) {
    Matrix<int> m({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
    int scalar = 5;
    Matrix<int> expected({{{6, 7}, {8, 9}}, {{10, 11}, {12, 13}}});

    auto result = m + scalar;
    EXPECT_TRUE(compareMatrixData(result, expected));
}

TEST(MatrixAdditionTest, MatrixPlusScalar_Float) {
    Matrix<float> m({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}});
    float scalar = 0.5;
    Matrix<float> expected({{{1.5, 2.5}, {3.5, 4.5}}, {{5.5, 6.5}, {7.5, 8.5}}});

    auto result = m + scalar;
    EXPECT_TRUE(compareMatrixData(result, expected));
}

TEST(MatrixAdditionTest, PlusEqualsMatrix_Int) {
    Matrix<int> m({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
    Matrix<int> other({{{6, 5}, {4, 3}}, {{2, 1}, {0, -1}}});

    m += other;
    Matrix<int> expected({{{7, 7}, {7, 7}}, {{7, 7}, {7, 7}}});
    EXPECT_TRUE(compareMatrixData(m, expected));
}

TEST(MatrixAdditionTest, PlusEqualsScalar_Float) {
    Matrix<float> m({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}});
    float scalar = 0.5;

    m += scalar;
    Matrix<float> expected({{{1.5, 2.5}, {3.5, 4.5}}, {{5.5, 6.5}, {7.5, 8.5}}});
    EXPECT_TRUE(compareMatrixData(m, expected));
}

TEST(MatrixSubtractionTest, MatrixMinusMatrix_Int) {
    Matrix<int> m1({{{10, 20}, {30, 40}}, {{50, 60}, {70, 80}}});
    Matrix<int> m2({{{5, 10}, {15, 20}}, {{25, 30}, {35, 40}}});
    Matrix<int> expected({{{5, 10}, {15, 20}}, {{25, 30}, {35, 40}}});

    auto result = m1 - m2;
    EXPECT_TRUE(compareMatrixData(result, expected));
}

TEST(MatrixSubtractionTest, MatrixMinusMatrix_Float) {
    Matrix<float> m1({{{10.5, 20.5}, {30.5, 40.5}}, {{50.5, 60.5}, {70.5, 80.5}}});
    Matrix<float> m2({{{5.5, 10.5}, {15.5, 20.5}}, {{25.5, 30.5}, {35.5, 40.5}}});
    Matrix<float> expected({{{5.0, 10.0}, {15.0, 20.0}}, {{25.0, 30.0}, {35.0, 40.0}}});

    auto result = m1 - m2;
    EXPECT_TRUE(compareMatrixData(result, expected));
}

TEST(MatrixSubtractionTest, MatrixMinusScalar_Int) {
    Matrix<int> m({{{10, 20}, {30, 40}}, {{50, 60}, {70, 80}}});
    int scalar = 5;
    Matrix<int> expected({{{5, 15}, {25, 35}}, {{45, 55}, {65, 75}}});

    auto result = m - scalar;
    EXPECT_TRUE(compareMatrixData(result, expected));
}

TEST(MatrixSubtractionTest, MatrixMinusScalar_Float) {
    Matrix<float> m({{{10.5, 20.5}, {30.5, 40.5}}, {{50.5, 60.5}, {70.5, 80.5}}});
    float scalar = 0.5;
    Matrix<float> expected({{{10.0, 20.0}, {30.0, 40.0}}, {{50.0, 60.0}, {70.0, 80.0}}});

    auto result = m - scalar;
    EXPECT_TRUE(compareMatrixData(result, expected));
}

TEST(MatrixSubtractionTest, MinusEqualsMatrix_Int) {
    Matrix<int> m({{{10, 20}, {30, 40}}, {{50, 60}, {70, 80}}});
    Matrix<int> other({{{5, 10}, {15, 20}}, {{25, 30}, {35, 40}}});

    m -= other;
    Matrix<int> expected({{{5, 10}, {15, 20}}, {{25, 30}, {35, 40}}});
    EXPECT_TRUE(compareMatrixData(m, expected));
}

TEST(MatrixSubtractionTest, MinusEqualsScalar_Float) {
    Matrix<float> m({{{10.5, 20.5}, {30.5, 40.5}}, {{50.5, 60.5}, {70.5, 80.5}}});
    float scalar = 0.5;

    m -= scalar;
    Matrix<float> expected({{{10.0, 20.0}, {30.0, 40.0}}, {{50.0, 60.0}, {70.0, 80.0}}});
    EXPECT_TRUE(compareMatrixData(m, expected));
}


TEST(MatrixMultiplicationTest, MatrixTimesScalar_Int) {
    Matrix<int> m({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
    int scalar = 2;
    Matrix<int> expected({{{2, 4}, {6, 8}}, {{10, 12}, {14, 16}}});

    auto result = m * scalar;
    EXPECT_TRUE(compareMatrixData(result, expected));
}

TEST(MatrixMultiplicationTest, ScalarTimesMatrix_Float) {
    Matrix<float> m({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}});
    float scalar = 0.5;
    Matrix<float> expected({{{0.5, 1.0}, {1.5, 2.0}}, {{2.5, 3.0}, {3.5, 4.0}}});

    auto result = scalar * m;
    EXPECT_TRUE(compareMatrixData(result, expected));
}

TEST(MatrixMultiplicationTest, TimesEqualsScalar_Int) {
    Matrix<int> m({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
    int scalar = 2;

    m *= scalar;
    Matrix<int> expected({{{2, 4}, {6, 8}}, {{10, 12}, {14, 16}}});
    EXPECT_TRUE(compareMatrixData(m, expected));
}

TEST(MatrixDivisionTest, MatrixDividedByScalar_Int) {
    Matrix<int> m({{{10, 20}, {30, 40}}, {{50, 60}, {70, 80}}});
    int scalar = 10;
    Matrix<int> expected({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});

    auto result = m / scalar;
    EXPECT_TRUE(compareMatrixData(result, expected));
}

TEST(MatrixDivisionTest, ThrowsWhenDividedByZero) {
    Matrix<int> m({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
    EXPECT_THROW(m / 0, std::invalid_argument);
}

TEST(MatrixDivisionTest, DivideEqualsScalar_Float) {
    Matrix<float> m({{{10.0, 20.0}, {30.0, 40.0}}, {{50.0, 60.0}, {70.0, 80.0}}});
    float scalar = 10.0;

    m /= scalar;
    Matrix<float> expected({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}});
    EXPECT_TRUE(compareMatrixData(m, expected));
}

TEST(MatrixDivisionTest, ThrowsWhenDivideEqualsByZero) {
    Matrix<float> m({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}});
    EXPECT_THROW(m /= 0.0, std::invalid_argument);
}


TEST(MatrixMultiplication, MultiChannelMultiplication)
{
    // 创建两个简单的2x2两通道矩阵
    Matrix<int> m1({{{1, 2}, {3, 4}},
                    {{5, 6}, {7, 8}}});
    Matrix<int> m2({{{1, 0}, {0, 1}},
                    {{1, 1}, {1, 1}}});

    // 执行矩阵乘法
    Matrix<int> result = m1 * m2;

    // 预期结果，每个通道的乘法都是独立的
    Matrix<int> expected({{{4, 4}, {3, 6}},
                          {{12, 8}, {7, 14}}});
    ASSERT_TRUE(compareMatrixData(result, expected));
}

TEST(MatrixMultiplication, ComplexMultiChannelMultiplication)
{
    // 创建一个3x3三通道矩阵
    Matrix<float> m1({{{1.0f, 0.5f, 0.2f}, {2.0f, 1.0f, 0.5f}, {3.0f, 1.5f, 0.8f}},
                      {{4.0f, 2.0f, 1.0f}, {5.0f, 2.5f, 1.3f}, {6.0f, 3.0f, 1.6f}},
                      {{7.0f, 3.5f, 1.8f}, {8.0f, 4.0f, 2.1f}, {9.0f, 4.5f, 2.4f}}});
    Matrix<float> m2({{{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
                      {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}},
                      {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}});

    // 执行矩阵乘法
    Matrix<float> result = m1 * m2;

    // 预期结果
    Matrix<float> expected({{{6.0f, 2.5f, 1.3f}, {5.0f, 3.0f, 1.3f}, {5.0f, 2.5f, 1.5f}},
                            {{15.0f, 5.5f, 2.9f}, {11.0f, 7.5f, 2.9f}, {11.0f, 5.5f, 3.9f}},
                            {{24.0f, 8.5f, 4.5f}, {17.0f, 12.0f, 4.5f}, {17.0f, 8.5f, 6.3f}}});
    ASSERT_TRUE(compareMatrixData(result, expected));
}
