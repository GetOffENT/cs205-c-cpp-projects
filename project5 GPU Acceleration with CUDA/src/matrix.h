#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <cuda_runtime.h>
#include <random>
template <typename T>
class Matrix
{
public:
    size_t rows;
    size_t cols;
    T *data;        // CPU memory
    T *data_device; // GPU memory

    // Constructor
    Matrix() : rows(0), cols(0), data(nullptr), data_device(nullptr) {}
    Matrix(size_t r, size_t c) : rows(r), cols(c), data(nullptr), data_device(nullptr)
    {
        size_t len = r * c;
        if (len == 0)
        {
            std::cerr << "Invalid size. The input should be > 0." << std::endl;
            throw std::invalid_argument("Matrix dimensions should be greater than 0.");
        }
        data = (T *)malloc(len * sizeof(T));
        if (data == nullptr)
        {
            std::cerr << "Allocate host memory failed." << std::endl;
            throw std::bad_alloc();
        }
        memset(data, 0, len * sizeof(T));

        cudaError_t status = cudaMalloc(&data_device, len * sizeof(T));
        if (status != cudaSuccess)
        {
            std::cerr << "Allocate device memory failed." << std::endl;
            free(data);
            throw std::bad_alloc();
        }
        cudaMemset(data_device, 0, len * sizeof(T));
    }

    // Destructor
    ~Matrix()
    {
        free(data);
        cudaFree(data_device);
    }

    // Set all elements to the same value
    void set(T value)
    {
        size_t len = rows * cols;
        for (size_t i = 0; i < len; i++)
        {
            data[i] = value;
        }
        // Also update GPU memory
        cudaMemcpy(data_device, data, len * sizeof(T), cudaMemcpyHostToDevice);
    }

    // Randomize matrix elements
    void randomize()
    {
        std::random_device rd;                          // Obtain a random number from hardware
        std::mt19937 gen(rd());                         // Seed the generator
        std::uniform_real_distribution<> dis(0.0, 1.0); // Define the range

        size_t len = rows * cols;
        for (size_t i = 0; i < len; i++)
        {
            data[i] = static_cast<T>(dis(gen)); // Generate random float number and assign it
        }
        // Copy updated data to GPU memory
        cudaMemcpy(data_device, data, len * sizeof(T), cudaMemcpyHostToDevice);
    }

    // Print matrix elements
    void print() const
    {
        for (size_t i = 0; i < rows; i++)
        {
            for (size_t j = 0; j < cols; j++)
            {
                std::cout << data[i * cols + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    // Overload << operator for output
    friend std::ostream &operator<<(std::ostream &os, const Matrix &mat)
    {
        for (size_t i = 0; i < mat.rows; i++)
        {
            for (size_t j = 0; j < mat.cols; j++)
            {
                os << mat.data[i * mat.cols + j] << " ";
            }
            os << std::endl;
        }
        return os;
    }

    // Overload >> operator for input
    friend std::istream &operator>>(std::istream &is, Matrix &mat)
    {
        for (size_t i = 0; i < mat.rows * mat.cols; i++)
        {
            is >> mat.data[i];
        }
        // Also update GPU memory
        cudaMemcpy(mat.data_device, mat.data, mat.rows * mat.cols * sizeof(T), cudaMemcpyHostToDevice);
        return is;
    }

    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
};

template <typename T>
bool mulAddCPU(const Matrix<T> &pMatA, T a, T b, Matrix<T> &pSptB);

template <typename T>
__global__ void mulAddKernel(const T *inputA, T a, T b, T *outputB, size_t len);

template <typename T>
bool mulAddGPU(const Matrix<T> &pMatA, T a, T b, Matrix<T> &pMatB);

bool mulMatrixCPU(const Matrix<float> &matA, const Matrix<float> &matB, Matrix<float> &matC);

bool mulMatrixGPU(const Matrix<float> &matA, const Matrix<float> &matB, Matrix<float> &matC);

bool mul(const Matrix<float> &lhs, const Matrix<float> &rhs, Matrix<float> &result);

#endif // MATRIX_H
