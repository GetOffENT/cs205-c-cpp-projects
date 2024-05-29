#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "matrix.h"

#define TIME_START gettimeofday(&t_start, NULL);
#define TIME_END(name)                                         \
    gettimeofday(&t_end, NULL);                                \
    elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;    \
    elapsedTime += (t_end.tv_usec - t_start.tv_usec) / 1000.0; \
    printf(#name " Time = %f ms.\n", elapsedTime);

template <typename T>
bool mulAddCPU(const Matrix<T> &MatA, T a, T b, Matrix<T> &MatB)
{
    if (MatA.data == nullptr || MatB.data == nullptr)
    {
        fprintf(stderr, "Null pointer.\n");
        return false;
    }
    if (MatA.rows != MatB.rows || MatA.cols != MatB.cols)
    {
        fprintf(stderr, "The input and output matrices are not the same size.\n");
        return false;
    }

    size_t len = MatA.rows * MatA.cols;
    for (int i = 0; i < len; i++)
    {
        MatB.data[i] = MatA.data[i] * a + b;
    }
    return true;
}

template <typename T>
__global__ void mulAddKernel(const T *inputA, T a, T b, T *outputB, size_t len)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < len)
    {
        outputB[i] = inputA[i] * a + b;
    }
}

template <typename T>
bool mulAddGPU(const Matrix<T> &MatA, T a, T b, Matrix<T> &MatB)
{
    if (MatA.data == nullptr || MatB.data == nullptr)
    {
        fprintf(stderr, "Null pointer.\n");
        return false;
    }
    if (MatA.rows != MatB.rows || MatA.cols != MatB.cols)
    {
        fprintf(stderr, "The input and output matrices are not the same size.\n");
        return false;
    }

    cudaError_t ecode = cudaSuccess;
    size_t len = MatA.rows * MatA.cols;

    cudaMemcpy(MatA.data_device, MatA.data, sizeof(T) * len, cudaMemcpyHostToDevice);
    mulAddKernel<<<(len + 255) / 256, 256>>>(MatA.data_device, a, b, MatB.data_device, len);
    if ((ecode = cudaGetLastError()) != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(ecode));
        return false;
    }
    cudaMemcpy(MatB.data, MatB.data_device, sizeof(T) * len, cudaMemcpyDeviceToHost);

    return true;
}

int main()
{
    struct timeval t_start, t_end;
    double elapsedTime = 0;

    // 输入矩阵A和B的大小及输入初始值
    int rows, cols;
    printf("Please input the size of the matrix (rows cols): ");
    scanf("%d %d", &rows, &cols);
    Matrix<float> matA(rows, cols);
    Matrix<float> matB(rows, cols);
    matA.set(1.0);
    matB.set(0.0);

    TIME_START;
    mulAddCPU<float>(matA, 2.0, 3.0, matB);
    TIME_END(CPU)
    // matB.print();

    TIME_START;
    mulAddGPU<float>(matA, 2.0, 3.0, matB);
    TIME_END(GPU);
    // matB.print();
}