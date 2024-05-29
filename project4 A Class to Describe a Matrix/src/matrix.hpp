#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#include <memory>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <vector>
#include <iostream>
#include <random>
#include <cblas.h>

template <typename T>
class Matrix
{
private:
    std::shared_ptr<T[]> data;       // 智能指针管理动态数组，存储矩阵数据
    size_t cols, rows;               // 列数和行数
    size_t channels;                 // 通道数
    size_t step;                     // 对齐后的每行步长（字节）
    size_t roiStartCol, roiStartRow; // ROI的起始列和起始行
    size_t roiCols, roiRows;         // ROI的列数和行数

    // 私有成员函数，用于实现矩阵乘法
    Matrix<T> multiply_using_openblas(const Matrix<T> &other) const;

public:
    // 获取矩阵维度
    size_t getCols() const { return cols; }
    size_t getRows() const { return rows; }
    size_t getChannels() const { return channels; }
    size_t getStep() const { return step; }
    size_t getROICols() const { return roiCols; }
    size_t getROIRows() const { return roiRows; }
    size_t getROIStartCol() const { return roiStartCol; }
    size_t getROIStartRow() const { return roiStartRow; }

    // 获取内部数据
    T *getData() const { return data.get(); }

    // 判断是否是方阵
    bool isSquare() const { return rows == cols; }

    // 默认构造函数
    Matrix();

    // 构造函数，指定行数、列数和通道数
    Matrix(size_t rows, size_t cols, size_t channels = 1);

    // ROI构造函数，基于现有矩阵和指定的ROI创建矩阵视图
    Matrix(const Matrix &base, size_t startRow, size_t startCol, size_t roiRows, size_t roiCols);

    // 拷贝构造函数（浅拷贝）
    Matrix(const Matrix &other);

    // 拷贝赋值运算符（浅拷贝）
    Matrix &operator=(const Matrix &other);

    // 移动构造函数
    Matrix(Matrix<T> &&other) noexcept;

    // 移动赋值运算符
    Matrix<T> &operator=(Matrix<T> &&other) noexcept;

    // std::vector<std::vector<std::vector<T>>> 到 Matrix 的转换
    Matrix(const std::vector<std::vector<std::vector<T>>> &vec);

    // std::vector<std::vector<T>> 到 Matrix 的转换(单通道)
    Matrix(const std::vector<std::vector<T>> &vec);

    // 覆盖或填充矩阵数据
    void fillData(const std::vector<std::vector<std::vector<T>>> &vec);
    void fillData(const std::vector<std::vector<T>> &vec);

    // 访问特定位置的元素
    T &at(size_t row, size_t col, size_t channel = 0);
    const T &at(size_t row, size_t col, size_t channel = 0) const;

    // 简化的访问运算符()
    T &operator()(size_t row, size_t col, size_t channel = 0);
    const T &operator()(size_t row, size_t col, size_t channel = 0) const;

    // 矩阵加法
    Matrix<T> operator+(const Matrix<T> &other) const;
    Matrix<T> operator+(T scalar) const;
    template <typename U>
    friend Matrix<U> operator+(U scalar, const Matrix<U> &mat);
    Matrix<T> &operator+=(const Matrix<T> &other);
    Matrix<T> &operator+=(T scalar);

    // 矩阵减法
    Matrix<T> operator-(const Matrix<T> &other) const;
    Matrix<T> operator-(T scalar) const;
    Matrix<T> &operator-=(const Matrix<T> &other);
    Matrix<T> &operator-=(T scalar);

    // 标量乘法
    Matrix<T> operator*(T scalar) const;
    Matrix<T> &operator*=(T scalar);
    template <typename U>
    friend Matrix<U> operator*(U scalar, const Matrix<U> &mat);

    // 标量除法
    Matrix<T> operator/(T scalar) const;
    Matrix<T> &operator/=(T scalar);

    // 矩阵乘法
    Matrix<T> operator*(const Matrix<T> &other) const;

    // 比较运算符
    bool operator==(const Matrix<T> &other) const;

    // 重写输入输出
    template <typename U>
    friend std::ostream &operator<<(std::ostream &out, const Matrix<U> &mat);
    template <typename U>
    friend std::istream &operator>>(std::istream &in, Matrix<U> &mat);
    void print() const;

    // 深拷贝
    Matrix<T> deepCopy() const;

    // 最大值
    T max() const;

    // 最小值
    T min() const;

    // 矩阵清空
    void clear();

    // 返回 ROI 区域
    Matrix getROI() const;

    // 返回目标区域矩阵
    Matrix subMatrix(size_t startRow, size_t startCol, size_t numRows, size_t numCols) const;

    // 提取指定通道
    Matrix<T> extractChannel(size_t channel) const;

    // 静态函数
    static Matrix<T> zero(size_t rows, size_t cols, size_t channels = 1);                               // 零矩阵
    static Matrix<T> identity(size_t size);                                                             // 单位矩阵
    static Matrix<T> random(size_t rows, size_t cols, size_t channels = 1, T minVal = 0, T maxVal = 1); // 随机矩阵
    static Matrix<T> transpose(const Matrix<T> &mat);                                                   // 矩阵转置
};

template <typename T>
Matrix<T>::Matrix()
    : cols(0), rows(0), channels(1), step(0), roiStartCol(0), roiStartRow(0), roiCols(0), roiRows(0), data(nullptr) {}

template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols, size_t channels)
    : cols(cols), rows(rows), channels(channels), roiStartCol(0), roiStartRow(0), roiCols(cols), roiRows(rows)
{
    if (cols == 0 || rows == 0 || channels == 0)
    {
        throw std::invalid_argument("Matrix dimensions and channels must be non-zero.");
    }

    size_t elementSize = sizeof(T);
    size_t pitch = cols * channels * elementSize;
    size_t alignment = 64;                             // 选择64字节对齐
    step = (pitch + alignment - 1) & ~(alignment - 1); // 计算对齐后的步长

    void *ptr = aligned_alloc(alignment, step * rows);
    if (!ptr)
    {
        throw std::bad_alloc();
    }
    data = std::shared_ptr<T[]>(static_cast<T *>(ptr), free); // 确保正确删除

    // 初始化内存为零
    std::memset(ptr, 0, step * rows);
}

template <typename T>
Matrix<T>::Matrix(const Matrix &base, size_t startRow, size_t startCol, size_t roiRows, size_t roiCols)
    : cols(base.cols), rows(base.rows), channels(base.channels), step(base.step), data(base.data),
      roiStartCol(startCol), roiStartRow(startRow), roiCols(roiCols), roiRows(roiRows)
{
    if (startCol >= cols || startCol + roiCols > cols ||
        startRow >= rows || startRow + roiRows > rows)
    {
        throw std::out_of_range("ROI is out of original matrix's bounds.");
    }
}

template <typename T>
Matrix<T>::Matrix(const Matrix &other)
    : data(other.data), cols(other.cols), rows(other.rows), channels(other.channels),
      step(other.step), roiStartCol(other.roiStartCol), roiStartRow(other.roiStartRow),
      roiCols(other.roiCols), roiRows(other.roiRows) {}

template <typename T>
Matrix<T> &Matrix<T>::operator=(const Matrix &other)
{
    if (this != &other)
    {
        data = other.data;
        cols = other.cols;
        rows = other.rows;
        channels = other.channels;
        step = other.step;
        roiStartCol = other.roiStartCol;
        roiStartRow = other.roiStartRow;
        roiCols = other.roiCols;
        roiRows = other.roiRows;
    }
    return *this;
}

template <typename T>
Matrix<T>::Matrix(Matrix<T> &&other) noexcept
    : data(std::move(other.data)), // 接管 data 智能指针
      cols(other.cols), rows(other.rows), channels(other.channels),
      step(other.step), roiStartCol(other.roiStartCol), roiStartRow(other.roiStartRow),
      roiCols(other.roiCols), roiRows(other.roiRows)
{
    // 重置原对象，避免原对象销毁时对数据的影响
    other.cols = 0;
    other.rows = 0;
    other.channels = 1;
    other.step = 0;
    other.roiStartCol = 0;
    other.roiStartRow = 0;
    other.roiCols = 0;
    other.roiRows = 0;
}

template <typename T>
Matrix<T> &Matrix<T>::operator=(Matrix<T> &&other) noexcept
{
    // 防止自赋值
    if (this != &other)
    {
        // 释放当前对象持有的资源
        data.reset();

        // 从 other 接管资源
        data = std::move(other.data);
        cols = other.cols;
        rows = other.rows;
        channels = other.channels;
        step = other.step;
        roiStartCol = other.roiStartCol;
        roiStartRow = other.roiStartRow;
        roiCols = other.roiCols;
        roiRows = other.roiRows;

        // 重置 other 对象
        other.cols = 0;
        other.rows = 0;
        other.channels = 0;
        other.step = 0;
        other.roiStartCol = 0;
        other.roiStartRow = 0;
        other.roiCols = 0;
        other.roiRows = 0;
        other.data = nullptr;
    }
    return *this;
}

template <typename T>
Matrix<T>::Matrix(const std::vector<std::vector<std::vector<T>>> &vec)
{
    if (vec.empty() || vec[0].empty() || vec[0][0].empty())
    {
        throw std::invalid_argument("3D vector cannot be empty.");
    }

    rows = vec.size();
    cols = vec[0].size();
    channels = vec[0][0].size();
    size_t elementSize = sizeof(T);
    size_t alignment = 64; // 假设64字节对齐
    size_t pitch = cols * channels * elementSize;
    step = (pitch + (alignment - 1)) & ~(alignment - 1); // 计算对齐后的步长
    roiCols = cols;
    roiRows = rows;
    roiStartCol = 0;
    roiStartRow = 0;

    size_t totalElements = rows * step / elementSize;
    data = std::shared_ptr<T[]>(static_cast<T *>(aligned_alloc(alignment, totalElements * elementSize)), std::free);

    for (size_t i = 0; i < rows; ++i)
    {
        if (vec[i].size() != cols)
        {
            throw std::invalid_argument("All inner vectors must have the same number of columns.");
        }
        for (size_t j = 0; j < cols; ++j)
        {
            if (vec[i][j].size() != channels)
            {
                throw std::invalid_argument("All inner inner vectors must have the same number of channels.");
            }
            for (size_t c = 0; c < channels; ++c)
            {
                size_t index = (i * step / elementSize + j * channels + c);
                data[index] = vec[i][j][c];
            }
        }
    }
}

template <typename T>
Matrix<T>::Matrix(const std::vector<std::vector<T>> &vec)
{
    if (vec.empty() || vec[0].empty())
    {
        throw std::invalid_argument("2D vector cannot be empty.");
    }

    rows = vec.size();
    cols = vec[0].size();
    channels = 1; // 单通道矩阵
    size_t elementSize = sizeof(T);
    size_t alignment = 64; // 假设64字节对齐
    size_t pitch = cols * channels * elementSize;
    step = (pitch + (alignment - 1)) & ~(alignment - 1); // 计算对齐后的步长
    roiCols = cols;
    roiRows = rows;
    roiStartCol = 0;
    roiStartRow = 0;

    size_t totalElements = rows * step / elementSize;
    data = std::shared_ptr<T[]>(static_cast<T *>(aligned_alloc(alignment, totalElements * elementSize)), std::free);

    for (size_t i = 0; i < rows; ++i)
    {
        if (vec[i].size() != cols)
        {
            throw std::invalid_argument("All inner vectors must have the same size.");
        }
        for (size_t j = 0; j < cols; ++j)
        {
            size_t index = (i * step / elementSize + j * channels);
            data[index] = vec[i][j];
        }
    }
}

template <typename T>
void Matrix<T>::fillData(const std::vector<std::vector<T>> &vec)
{
    if (vec.empty() || vec[0].empty())
    {
        throw std::invalid_argument("2D vector cannot be empty.");
    }

    size_t newRows = vec.size();
    size_t newCols = vec[0].size();
    size_t newChannels = 1;
    size_t elementSize = sizeof(T);
    size_t alignment = 64;
    size_t newPitch = newCols * newChannels * elementSize;
    size_t newStep = (newPitch + (alignment - 1)) & ~(alignment - 1);
    size_t totalElements = newRows * newStep / elementSize;

    // 仅在维度或步长改变时重新分配内存
    if (newRows != rows || newCols != cols || newChannels != channels || newStep != step)
    {
        rows = newRows;
        cols = newCols;
        channels = newChannels;
        step = newStep;

        data.reset(static_cast<T *>(aligned_alloc(alignment, totalElements * elementSize)), std::free);
    }

    for (size_t i = 0; i < rows; ++i)
    {
        if (vec[i].size() != cols)
        {
            throw std::invalid_argument("All inner vectors must have the same size.");
        }
        for (size_t j = 0; j < cols; ++j)
        {
            size_t index = (i * step / elementSize + j * channels);
            data[index] = vec[i][j];
        }
    }
}

template <typename T>
void Matrix<T>::fillData(const std::vector<std::vector<std::vector<T>>> &vec)
{
    if (vec.empty() || vec[0].empty() || vec[0][0].empty())
    {
        throw std::invalid_argument("3D vector cannot be empty.");
    }

    size_t newRows = vec.size();
    size_t newCols = vec[0].size();
    size_t newChannels = vec[0][0].size();
    size_t elementSize = sizeof(T);
    size_t alignment = 64;
    size_t newPitch = newCols * newChannels * elementSize;
    size_t newStep = (newPitch + (alignment - 1)) & ~(alignment - 1);
    size_t totalElements = newRows * newStep / elementSize;

    // 仅在维度或步长改变时重新分配内存
    if (newRows != rows || newCols != cols || newChannels != channels || newStep != step)
    {
        rows = newRows;
        cols = newCols;
        channels = newChannels;
        step = newStep;
        data.reset(static_cast<T *>(aligned_alloc(alignment, totalElements * elementSize)), std::free);
    }

    for (size_t i = 0; i < rows; ++i)
    {
        if (vec[i].size() != cols)
        {
            throw std::invalid_argument("All inner vectors must have the same number of columns.");
        }
        for (size_t j = 0; j < cols; ++j)
        {
            if (vec[i][j].size() != channels)
            {
                throw std::invalid_argument("All inner inner vectors must have the same number of channels.");
            }
            for (size_t c = 0; c < channels; ++c)
            {
                size_t index = (i * step / elementSize + j * channels + c);
                data[index] = vec[i][j][c];
            }
        }
    }
}

template <typename T>
T &Matrix<T>::operator()(size_t row, size_t col, size_t channel)
{
    size_t index = (roiStartRow + row) * step / sizeof(T) + (roiStartCol + col) * channels + channel;
    return data[index];
}

template <typename T>
const T &Matrix<T>::operator()(size_t row, size_t col, size_t channel) const
{
    size_t index = (roiStartRow + row) * step / sizeof(T) + (roiStartCol + col) * channels + channel;
    return data[index];
}

template <typename T>
T &Matrix<T>::at(size_t row, size_t col, size_t channel)
{
    if (row >= roiRows || col >= roiCols || channel >= channels)
    {
        throw std::out_of_range("Index out of range");
    }
    size_t index = (roiStartRow + row) * step / sizeof(T) + (roiStartCol + col) * channels + channel;
    return data[index];
}

template <typename T>
const T &Matrix<T>::at(size_t row, size_t col, size_t channel) const
{
    if (row >= roiRows || col >= roiCols || channel >= channels)
    {
        throw std::out_of_range("Index out of range");
    }
    size_t index = (roiStartRow + row) * step / sizeof(T) + (roiStartCol + col) * channels + channel;
    return data[index];
}

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &other) const
{
    if (rows != other.rows || cols != other.cols || channels != other.channels)
    {
        throw std::invalid_argument("Matrices dimensions and channels must match for addition.");
    }

    Matrix<T> result(rows, cols, channels);         // 创建结果矩阵
    size_t totalElements = rows * step / sizeof(T); // 计算总元素数量

    for (size_t i = 0; i < totalElements; i++)
    {
        result.data[i] = this->data[i] + other.data[i];
    }

    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator+(T scalar) const
{
    Matrix<T> result(rows, cols, channels);
    size_t totalElements = rows * step / sizeof(T);

    for (size_t i = 0; i < totalElements; i++)
    {
        result.data[i] = this->data[i] + scalar;
    }

    return result;
}

template <typename U>
Matrix<U> operator+(U scalar, const Matrix<U> &mat)
{
    return mat + scalar;
}

template <typename T>
Matrix<T> &Matrix<T>::operator+=(const Matrix<T> &other)
{
    if (rows != other.rows || cols != other.cols || channels != other.channels)
    {
        throw std::invalid_argument("Matrices dimensions and channels must match for addition.");
    }

    size_t totalElements = rows * step / sizeof(T); // 计算总元素数量

    for (size_t i = 0; i < totalElements; i++)
    {
        this->data[i] += other.data[i];
    }

    return *this;
}

template <typename T>
Matrix<T> &Matrix<T>::operator+=(T scalar)
{
    size_t totalElements = rows * step / sizeof(T); // 计算总元素数量

    for (size_t i = 0; i < totalElements; i++)
    {
        this->data[i] += scalar;
    }

    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T> &other) const
{
    if (rows != other.rows || cols != other.cols || channels != other.channels)
    {
        throw std::invalid_argument("Matrices dimensions and channels must match for subtraction.");
    }

    Matrix<T> result(rows, cols, channels);
    size_t totalElements = rows * step / sizeof(T);

    for (size_t i = 0; i < totalElements; i++)
    {
        result.data[i] = this->data[i] - other.data[i];
    }

    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(T scalar) const
{
    Matrix<T> result(rows, cols, channels);
    size_t totalElements = rows * step / sizeof(T);

    for (size_t i = 0; i < totalElements; i++)
    {
        result.data[i] = this->data[i] - scalar;
    }

    return result;
}

template <typename T>
Matrix<T> &Matrix<T>::operator-=(const Matrix<T> &other)
{
    if (rows != other.rows || cols != other.cols || channels != other.channels)
    {
        throw std::invalid_argument("Matrices dimensions and channels must match for subtraction.");
    }

    size_t totalElements = rows * step / sizeof(T);

    for (size_t i = 0; i < totalElements; i++)
    {
        this->data[i] -= other.data[i];
    }

    return *this;
}

template <typename T>
Matrix<T> &Matrix<T>::operator-=(T scalar)
{
    size_t totalElements = rows * step / sizeof(T); // 计算总元素数量

    for (size_t i = 0; i < totalElements; i++)
    {
        this->data[i] -= scalar;
    }

    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(T scalar) const
{
    Matrix<T> result(rows, cols, channels);
    size_t totalElements = rows * step / sizeof(T);

    for (size_t i = 0; i < totalElements; i++)
    {
        result.data[i] = this->data[i] * scalar;
    }

    return result;
}

template <typename U>
Matrix<U> operator*(U scalar, const Matrix<U> &mat)
{
    return mat * scalar;
}

template <typename T>
Matrix<T> &Matrix<T>::operator*=(T scalar)
{
    size_t totalElements = rows * step / sizeof(T);

    for (size_t i = 0; i < totalElements; i++)
    {
        this->data[i] *= scalar;
    }

    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator/(T scalar) const
{
    if (scalar == 0)
    {
        throw std::invalid_argument("Division by zero.");
    }

    Matrix<T> result(rows, cols, channels);
    size_t totalElements = rows * step / sizeof(T);

    for (size_t i = 0; i < totalElements; i++)
    {
        result.data[i] = this->data[i] / scalar;
    }

    return result;
}

template <typename T>
Matrix<T> &Matrix<T>::operator/=(T scalar)
{
    if (scalar == 0)
    {
        throw std::invalid_argument("Division by zero.");
    }

    size_t totalElements = rows * step / sizeof(T);

    for (size_t i = 0; i < totalElements; i++)
    {
        this->data[i] /= scalar;
    }

    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::multiply_using_openblas(const Matrix<T> &other) const
{
    if constexpr (!(std::is_same_v<T, float> || std::is_same_v<T, double>))
    {
        throw std::runtime_error("OpenBLAS multiplication only supports float and double types.");
    }
    Matrix<T> result(rows, other.cols, channels);

    for (size_t c = 0; c < channels; ++c)
    {
        // 提取
        Matrix<T> channelMatrix1 = this->extractChannel(c);
        Matrix<T> channelMatrix2 = other.extractChannel(c);

        // 复制数据到新的连续内存数组
        T *data1 = new T[channelMatrix1.rows * channelMatrix1.cols];
        T *data2 = new T[channelMatrix2.rows * channelMatrix2.cols];

        for (size_t i = 0; i < channelMatrix1.rows; ++i)
        {
            std::memcpy(data1 + i * channelMatrix1.cols, channelMatrix1.data.get() + i * channelMatrix1.step / sizeof(T), channelMatrix1.cols * sizeof(T));
        }

        for (size_t i = 0; i < channelMatrix2.rows; ++i)
        {
            std::memcpy(data2 + i * channelMatrix2.cols, channelMatrix2.data.get() + i * channelMatrix2.step / sizeof(T), channelMatrix2.cols * sizeof(T));
        }

        T *data_result = new T[channelMatrix1.rows * channelMatrix2.cols];

        if constexpr (std::is_same_v<T, float>)
        {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        channelMatrix1.rows, channelMatrix2.cols, channelMatrix1.cols,
                        1.0f, data1, channelMatrix1.cols,
                        data2, channelMatrix2.cols,
                        0.0f, data_result, channelMatrix2.cols);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        channelMatrix1.rows, channelMatrix2.cols, channelMatrix1.cols,
                        1.0, data1, channelMatrix1.cols,
                        data2, channelMatrix2.cols,
                        0.0, data_result, channelMatrix2.cols);
        }

        // 写回结果矩阵
        for (size_t i = 0; i < channelMatrix1.rows; ++i)
        {
            for (size_t j = 0; j < channelMatrix2.cols; ++j)
            {
                result(i, j, c) = data_result[i * channelMatrix2.cols + j];
            }
        }

        // 释放临时数组内存
        delete[] data1;
        delete[] data2;
        delete[] data_result;
    }

    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &other) const
{
    if (cols != other.rows || channels != other.channels)
    {
        throw std::invalid_argument("Matrix dimensions and channels must be compatible for multiplication.");
    }

    // 检查是否应该使用 OpenBLAS
    if constexpr ( std::is_same_v<T, double>)
    {
        return this->multiply_using_openblas(other);
    }
    else
    {
        Matrix result(rows, other.cols, channels);
        for (size_t c = 0; c < channels; ++c)
        {
            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < other.cols; ++j)
                {
                    T sum = 0;
                    for (size_t k = 0; k < cols; ++k)
                    {
                        sum += (*this)(i, k, c) * other(k, j, c);
                    }
                    result(i, j, c) = sum;
                }
            }
        }
        return result;
    }
}

template <typename T>
bool Matrix<T>::operator==(const Matrix<T> &other) const
{
    if (rows != other.rows || cols != other.cols || channels != other.channels)
    {
        return false;
    }

    // 比较每个元素
    size_t totalElements = rows * step / sizeof(T);
    for (size_t i = 0; i < totalElements; i++)
    {
        if (this->data[i] != other.data[i])
        {
            return false;
        }
    }

    return true;
}

template <typename T>
std::ostream &operator<<(std::ostream &out, const Matrix<T> &mat)
{
    for (size_t i = 0; i < mat.rows; ++i)
    {
        for (size_t j = 0; j < mat.cols; ++j)
        {
            if (mat.channels > 1)
                out << '{';
            for (size_t c = 0; c < mat.channels; ++c)
            {
                out << mat.data[i * mat.step / sizeof(T) + j * mat.channels + c];
                if (c < mat.channels - 1)
                    out << ", "; // 通道与通道之间的分隔符
            }
            if (mat.channels > 1)
                out << '}';
            if (j < mat.cols - 1)
                out << " | "; // 列与列之间的分隔符
        }
        out << '\n';
    }
    return out;
}

template <typename T>
std::istream &operator>>(std::istream &in, Matrix<T> &mat)
{
    for (size_t i = 0; i < mat.rows; ++i)
    {
        for (size_t j = 0; j < mat.cols; ++j)
        {
            for (size_t c = 0; c < mat.channels; ++c)
            {
                in >> mat.data[i * mat.step / sizeof(T) + j * mat.channels + c];
            }
        }
    }
    return in;
}

template <typename T>
void Matrix<T>::print() const
{
    std::cout << *this;
}

template <typename T>
Matrix<T> Matrix<T>::deepCopy() const
{
    Matrix<T> copy(rows, cols, channels);
    size_t totalElements = rows * step / sizeof(T);
    copy.data = std::shared_ptr<T[]>(new T[totalElements], std::default_delete<T[]>());
    std::copy(this->data.get(), this->data.get() + totalElements, copy.data.get());
    return copy;
}

template <typename T>
T Matrix<T>::min() const
{
    if (rows * cols * channels == 0)
    {
        throw std::runtime_error("Matrix is empty.");
    }
    T min_value = data[0];
    for (size_t i = 0; i < rows * cols * channels; ++i)
    {
        if (data[i] < min_value)
        {
            min_value = data[i];
        }
    }
    return min_value;
}

template <typename T>
T Matrix<T>::max() const
{
    if (rows * cols * channels == 0)
    {
        throw std::runtime_error("Matrix is empty.");
    }
    T max_value = data[0];
    for (size_t i = 0; i < rows * cols * channels; ++i)
    {
        if (data[i] > max_value)
        {
            max_value = data[i];
        }
    }
    return max_value;
}

template <typename T>
void Matrix<T>::clear()
{
    std::fill(data.get(), data.get() + rows * cols * channels, T{});
}

template <typename T>
Matrix<T> Matrix<T>::getROI() const
{
    if (roiStartRow + roiRows > rows || roiStartCol + roiCols > cols)
    {
        throw std::out_of_range("ROI exceeds matrix dimensions.");
    }

    Matrix roi(roiRows, roiCols, channels);
    for (size_t i = 0; i < roiRows; ++i)
    {
        for (size_t j = 0; j < roiCols; ++j)
        {
            for (size_t c = 0; c < channels; ++c)
            {
                roi(i, j, c) = (*this)(roiStartRow + i, roiStartCol + j, c);
            }
        }
    }
    return roi;
}

template <typename T>
Matrix<T> Matrix<T>::subMatrix(size_t startRow, size_t startCol, size_t numRows, size_t numCols) const
{
    if (startRow + numRows > rows || startCol + numCols > cols)
    {
        throw std::out_of_range("Requested subMatrix exceeds matrix dimensions.");
    }

    Matrix sub(numRows, numCols, channels);
    for (size_t i = 0; i < numRows; ++i)
    {
        for (size_t j = 0; j < numCols; ++j)
        {
            for (size_t c = 0; c < channels; ++c)
            {
                sub(i, j, c) = (*this)(startRow + i, startCol + j, c);
            }
        }
    }
    return sub;
}

template <typename T>
Matrix<T> Matrix<T>::extractChannel(size_t channel) const
{
    if (channel >= channels)
    {
        throw std::out_of_range("Requested channel exceeds available channels.");
    }

    Matrix<T> result(rows, cols, 1); // 创建一个新的单通道矩阵
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            result(i, j, 0) = (*this)(i, j, channel); // 复制指定通道的数据
        }
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::zero(size_t rows, size_t cols, size_t channels)
{
    Matrix<T> mat(rows, cols, channels);
    std::fill(mat.data.get(), mat.data.get() + rows * cols * channels, T(0));
    return mat;
}

template <typename T>
Matrix<T> Matrix<T>::identity(size_t size)
{
    Matrix<T> mat(size, size, 1);
    for (size_t i = 0; i < size; ++i)
    {
        mat(i, i, 0) = 1;
    }
    return mat;
}

template <typename T>
Matrix<T> Matrix<T>::random(size_t rows, size_t cols, size_t channels, T minVal, T maxVal)
{
    Matrix<T> mat(rows, cols, channels);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(minVal, maxVal);

    for (size_t i = 0; i < rows * cols * channels; ++i)
    {
        mat.data[i] = static_cast<T>(dis(gen));
    }
    return mat;
}

template <typename T>
Matrix<T> Matrix<T>::transpose(const Matrix<T> &mat)
{
    Matrix<T> transposed(mat.cols, mat.rows, mat.channels);
    for (size_t i = 0; i < mat.rows; ++i)
    {
        for (size_t j = 0; j < mat.cols; ++j)
        {
            for (size_t c = 0; c < mat.channels; ++c)
            {
                transposed(j, i, c) = mat(i, j, c);
            }
        }
    }
    return transposed;
}

#endif // _MATRIX_HPP_

// g++ matrix.cpp -lopenblas