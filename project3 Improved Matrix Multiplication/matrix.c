#include "matrix.h"

#include <stdlib.h>
#include <stdio.h>

Matrix *MatrixConstructor(size_t rows, size_t cols)
{
    // 分配Matrix结构体的内存
    Matrix *matrix = (Matrix *)malloc(sizeof(Matrix));
    if (matrix == NULL)
    {
        fprintf(stderr, "Memory allocation failed for Matrix struct.\n");
        return NULL; // 如果内存分配失败，则返回NULL
    }

    // 初始化行和列
    matrix->rows = rows;
    matrix->cols = cols;

    // 为矩阵数据分配内存
    matrix->data = (float *)aligned_alloc(1024, rows * cols * sizeof(float));
    if (matrix->data == NULL)
    {
        fprintf(stderr, "Memory allocation failed for Matrix data.\n");
        free(matrix); // 如果数据分配失败，释放已分配的结构体内存
        return NULL;
    }

    // 初始化矩阵数据为0
    for (size_t i = 0; i < rows * cols; i++)
    {
        matrix->data[i] = 0.0f;
    }

    return matrix; // 返回指向新分配和初始化的Matrix结构体的指针
}

void MatrixDestructor(Matrix *matrix)
{
    if (matrix != NULL)
    {
        free(matrix->data); // 释放矩阵数据内存
        free(matrix);       // 释放Matrix结构体内存
    }
}

void MatrixRandomize(Matrix *matrix)
{
    if (matrix == NULL || matrix->data == NULL)
    {
        fprintf(stderr, "Input matrices cannot be NULL.\n");
        return; // 检查矩阵指针和数据指针是否为NULL
    }

    // srand(time(NULL)); // 使用当前时间作为随机种子

    size_t size = matrix->rows * matrix->cols;
    for (size_t i = 0; i < size; i++)
    {
        matrix->data[i] = (float)(rand() / RAND_MAX * 10.0f); // 生成0到10之间的随机浮点数
    }
}

Matrix *matmul_plain(const Matrix *MatA, const Matrix *MatB)
{
    // 检查输入矩阵是否为NULL
    if (MatA == NULL || MatB == NULL || MatA->data == NULL || MatB->data == NULL)
    {
        fprintf(stderr, "Input matrices cannot be NULL.\n");
        return NULL;
    }

    // 检查矩阵的尺寸是否匹配
    if (MatA->cols != MatB->rows)
    {
        fprintf(stderr, "Matrix dimensions do not match for multiplication.\n");
        return NULL;
    }

    // 创建结果矩阵
    Matrix *MatC = MatrixConstructor(MatA->rows, MatB->cols);
    if (!MatC)
    {
        fprintf(stderr, "Memory allocation failed for result matrix.\n");
        return NULL;
    }

    // 计算矩阵乘法
    plain_gemm_ikj(MatA->rows, MatB->cols, MatA->cols, MatA->data, MatB->data, MatC->data);


    // 返回计算后的结果矩阵
    return MatC;
}

Matrix *matmul_improved(const Matrix *MatA, const Matrix *MatB)
{
    // 检查输入矩阵是否为NULL
    if (MatA == NULL || MatB == NULL || MatA->data == NULL || MatB->data == NULL)
    {
        fprintf(stderr, "Input matrices cannot be NULL.\n");
        return NULL;
    }

    // 检查矩阵的尺寸是否匹配
    if (MatA->cols != MatB->rows)
    {
        fprintf(stderr, "Matrix dimensions do not match for multiplication.\n");
        return NULL;
    }

    // 创建结果矩阵
    Matrix *MatC = MatrixConstructor(MatA->rows, MatB->cols);
    if (!MatC)
    {
        fprintf(stderr, "Memory allocation failed for result matrix.\n");
        return NULL;
    }

    // 计算矩阵乘法
    gemm_blocked_packed_avx_2_OpenMP(MatA->rows, MatB->cols, MatA->cols, MatA->data, MatB->data, MatC->data);

    // 返回计算后的结果矩阵
    return MatC;
}


int main()
{
    // 创建两个矩阵
    Matrix *A = MatrixConstructor(1024, 512);
    Matrix *B = MatrixConstructor(512, 1024);

    // 随机初始化矩阵数据
    MatrixRandomize(A);
    MatrixRandomize(B);

    // 比较两种矩阵乘法的结果，若不同输出-1
    Matrix *C = matmul_improved(A, B);
    Matrix *D = matmul_plain(A, B);

    // 输出结果
    for (size_t i = 0; i < C->rows * C->cols; i++)
    {
        if (C->data[i] != D->data[i])
        {
            printf("-1\n");
            break;
        }
    }

    printf("Test Passed\n");

    // 释放矩阵内存
    MatrixDestructor(A);
    MatrixDestructor(B);
    MatrixDestructor(C);
    MatrixDestructor(D);

    return 0;
}