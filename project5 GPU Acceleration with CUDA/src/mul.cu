#include "matrix.h"
#include <cublas_v2.h>
#include <cblas.h>
#include <iostream>

// 使用 cblas 库在 CPU 上计算矩阵乘法
bool mulMatrixCPU(const Matrix<float> &lhs, const Matrix<float> &rhs, Matrix<float> &dst)
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

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                lhs.rows, rhs.cols, lhs.cols, alpha,
                lhs.data, lhs.cols, rhs.data, rhs.cols, beta, dst.data, dst.cols);

    return true;
}

// 使用 cuBLAS 库在 GPU 上计算矩阵乘法
bool mulMatrixGPU(const Matrix<float> &lhs, const Matrix<float> &rhs, Matrix<float> &dst)
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

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "CUBLAS initialization failed\n";
        return false;
    }

    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rhs.cols, lhs.rows, lhs.cols,
                         &alpha, rhs.data_device, rhs.cols, lhs.data_device, rhs.rows,
                         &beta, dst.data_device, dst.cols);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "CUBLAS SGEMM failed\n";
        cublasDestroy(handle);
        return false;
    }

    cudaMemcpy(dst.data, dst.data_device, sizeof(float) * dst.rows * dst.cols, cudaMemcpyDeviceToHost);
    cublasDestroy(handle);
    return true;
}