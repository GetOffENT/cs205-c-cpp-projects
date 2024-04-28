#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <stddef.h>

typedef struct
{
    size_t rows;
    size_t cols;
    float *data;
} Matrix;

Matrix *matmul_plain(const Matrix *MatA, const Matrix *MatB);
Matrix *matmul_improved(const Matrix *MatA, const Matrix *MatB);

void plain_gemm_ijk(const size_t N, const size_t M, const size_t K, const float *mat1, const float *mat2, float *mat);
void plain_gemm_ikj(const size_t N, const size_t M, const size_t K, const float *mat1, const float *mat2, float *mat);
void plain_gemm_kij(const size_t N, const size_t M, const size_t K, const float *mat1, const float *mat2, float *mat);
void plain_gemm_jik(const size_t N, const size_t M, const size_t K, const float *mat1, const float *mat2, float *mat);
void plain_gemm_kji(const size_t N, const size_t M, const size_t K, const float *mat1, const float *mat2, float *mat);
void plain_gemm_jki(const size_t N, const size_t M, const size_t K, const float *mat1, const float *mat2, float *mat);
void gemm_blocked(const size_t N, const size_t M, const size_t K, const float *mat1, const float *mat2, float *mat);
void gemm_blocked_packed(const size_t M, const size_t N, const size_t K, const float *mat1, const float *mat2, float *mat);
void gemm_blocked_packed_writecache(const size_t M, const size_t N, const size_t K, const float *mat1, const float *mat2, float *mat);
void gemm_blocked_packed_avx_1(const size_t M, const size_t N, const size_t K, const float *mat1, const float *mat2, float *mat);
void gemm_blocked_packed_avx_2(const size_t M, const size_t N, const size_t K, const float *mat1, const float *mat2, float *mat);
void gemm_ikj_OpenMP(const size_t M, const size_t N, const size_t K, const float *mat1, const float *mat2, float *mat);
void gemm_blocked_OpenMP(const size_t N, const size_t M, const size_t K, const float *mat1, const float *mat2, float *mat);
void gemm_blocked_packed_OpenMP(const size_t M, const size_t N, const size_t K, const float *mat1, const float *mat2, float *mat);
void gemm_blocked_packed_avx_1_OpenMP(const size_t M, const size_t N, const size_t K, const float *mat1, const float *mat2, float *mat);
void gemm_blocked_packed_avx_2_OpenMP(const size_t M, const size_t N, const size_t K, const float *mat1, const float *mat2, float *mat);

void gemm_cblas(const size_t N, const size_t M, const size_t K, const float *mat1, const float *mat2, float *mat);

#endif