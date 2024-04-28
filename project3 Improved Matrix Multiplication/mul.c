#include "matrix.h"
#include <cblas.h>
#include <string.h>
#include <immintrin.h> // AVX2
#include <omp.h>

void plain_gemm_ijk(const size_t M, const size_t N, const size_t K, const float *mat1, const float *mat2, float *mat)
{
    if (mat1 == NULL || mat2 == NULL || mat == NULL)
        return;

    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++)
            for (size_t k = 0; k < K; k++)
                mat[i * N + j] += mat1[i * K + k] * mat2[k * N + j];
}

void plain_gemm_ikj(const size_t M, const size_t N, const size_t K, const float *mat1, const float *mat2, float *mat)
{
    if (mat1 == NULL || mat2 == NULL || mat == NULL)
        return;

    for (size_t i = 0; i < M; i++)
        for (size_t k = 0; k < K; k++)
            for (size_t j = 0; j < N; j++)
                mat[i * N + j] += mat1[i * K + k] * mat2[k * N + j];
}

void plain_gemm_kij(const size_t M, const size_t N, const size_t K, const float *mat1, const float *mat2, float *mat)
{
    if (mat1 == NULL || mat2 == NULL || mat == NULL)
        return;

    for (size_t k = 0; k < K; k++)
        for (size_t i = 0; i < M; i++)
            for (size_t j = 0; j < N; j++)
                mat[i * N + j] += mat1[i * K + k] * mat2[k * N + j];
}

void plain_gemm_jik(const size_t M, const size_t N, const size_t K, const float *mat1, const float *mat2, float *mat)
{
    if (mat1 == NULL || mat2 == NULL || mat == NULL)
        return;

    for (size_t j = 0; j < N; j++)
        for (size_t i = 0; i < M; i++)
            for (size_t k = 0; k < K; k++)
                mat[i * N + j] += mat1[i * K + k] * mat2[k * N + j];
}

void plain_gemm_kji(const size_t M, const size_t N, const size_t K, const float *mat1, const float *mat2, float *mat)
{
    if (mat1 == NULL || mat2 == NULL || mat == NULL)
        return;

    for (size_t k = 0; k < K; k++)
        for (size_t j = 0; j < N; j++)
            for (size_t i = 0; i < M; i++)
                mat[i * N + j] += mat1[i * K + k] * mat2[k * N + j];
}

void plain_gemm_jki(const size_t M, const size_t N, const size_t K, const float *mat1, const float *mat2, float *mat)
{
    if (mat1 == NULL || mat2 == NULL || mat == NULL)
        return;

    for (size_t j = 0; j < N; j++)
        for (size_t k = 0; k < K; k++)
            for (size_t i = 0; i < M; i++)
                mat[i * N + j] += mat1[i * K + k] * mat2[k * N + j];
}

// 分块
void gemm_blocked(const size_t M, const size_t N, const size_t K, const float *mat1, const float *mat2, float *mat)
{
    if (mat1 == NULL || mat2 == NULL || mat == NULL)
        return;

    const size_t MBlockSize = M < 128 ? M : 128;
    const size_t KBlockSize = K < 128 ? K : 128;
    const size_t NBlockSize = N < 128 ? N : 128;
    for (size_t i = 0; i < M; i += MBlockSize)
        for (size_t k = 0; k < K; k += KBlockSize)
            for (size_t j = 0; j < N; j += NBlockSize)
                for (size_t ii = i; ii < i + MBlockSize; ii++)
                    for (size_t kk = k; kk < k + KBlockSize; kk++)
                        for (size_t jj = j; jj < j + NBlockSize; jj++)
                            mat[ii * N + jj] += mat1[ii * K + kk] * mat2[kk * N + jj];
}

// 分块 + 重排
void gemm_blocked_packed(const size_t M, const size_t N, const size_t K, const float *mat1, const float *mat2, float *mat)
{
    if (mat1 == NULL || mat2 == NULL || mat == NULL)
        return;

    const size_t MBlockSize = M < 128 ? M : 128;
    const size_t KBlockSize = K < 128 ? K : 128;
    const size_t NBlockSize = N < 128 ? N : 128;

    float *packed_mat1 = (float *)aligned_alloc(1024, MBlockSize * KBlockSize * sizeof(float));
    float *packed_mat2 = (float *)aligned_alloc(1024, KBlockSize * NBlockSize * sizeof(float));

    for (size_t i = 0; i < M; i += MBlockSize)
    {
        for (size_t k = 0; k < K; k += KBlockSize)
        {
            // Pack mat1's current block
            for (size_t ii = i, p1_idx = 0; ii < i + MBlockSize; ++ii)
                for (size_t kk = k; kk < k + KBlockSize; ++kk)
                    packed_mat1[p1_idx++] = mat1[ii * K + kk];

            for (size_t j = 0; j < N; j += NBlockSize)
            {
                // Pack mat2's current block
                for (size_t kk = k, p2_idx = 0; kk < k + KBlockSize; ++kk)
                    for (size_t jj = j; jj < j + NBlockSize; ++jj)
                        packed_mat2[p2_idx++] = mat2[kk * N + jj];

                // Compute block product
                for (size_t ii = 0; ii < MBlockSize; ++ii)
                    for (size_t kk = 0; kk < KBlockSize; ++kk)
                        for (size_t jj = 0; jj < NBlockSize; ++jj)
                            mat[(i + ii) * N + j + jj] += packed_mat1[ii * KBlockSize + kk] * packed_mat2[kk * NBlockSize + jj];
            }
        }
    }

    free(packed_mat1);
    free(packed_mat2);
}

// 分块 + 重排 + 写缓存优化(没啥用)
void gemm_blocked_packed_writecache(const size_t M, const size_t N, const size_t K, const float *mat1, const float *mat2, float *mat)
{
    if (mat1 == NULL || mat2 == NULL || mat == NULL)
    {
        printf("One or more input pointers are NULL.\n");
        return;
    }

    const size_t MBlockSize = M < 128 ? M : 128;
    const size_t KBlockSize = K < 128 ? K : 128;
    const size_t NBlockSize = N < 128 ? N : 128;

    float *packed_mat1 = (float *)aligned_alloc(1024, MBlockSize * KBlockSize * sizeof(float));
    float *packed_mat2 = (float *)aligned_alloc(1024, KBlockSize * NBlockSize * sizeof(float));

    float *write_cache = (float *)aligned_alloc(1024, M * N * sizeof(float));
    if (packed_mat1 == NULL || packed_mat2 == NULL || write_cache == NULL)
    {
        printf("Memory allocation failed.\n");
        free(packed_mat1);
        free(packed_mat2);
        free(write_cache);
        return;
    }
    memset(write_cache, 0, M * N * sizeof(float));
    for (size_t i = 0; i < M; i += MBlockSize)
    {
        for (size_t k = 0; k < K; k += KBlockSize)
        {
            // Pack mat1's current block
            for (size_t ii = i, p1_idx = 0; ii < i + MBlockSize; ++ii)
                for (size_t kk = k; kk < k + KBlockSize; ++kk)
                    packed_mat1[p1_idx++] = mat1[ii * K + kk];

            for (size_t j = 0; j < N; j += NBlockSize)
            {
                // Pack mat2's current block
                for (size_t kk = k, p2_idx = 0; kk < k + KBlockSize; ++kk)
                    for (size_t jj = j; jj < j + NBlockSize; ++jj)
                        packed_mat2[p2_idx++] = mat2[kk * N + jj];

                float *ptr = write_cache + (MBlockSize * NBlockSize) * ((i / MBlockSize) * (N / NBlockSize) + (j / NBlockSize));
                // Compute block product
                for (size_t ii = 0; ii < MBlockSize; ++ii)
                    for (size_t jj = 0; jj < NBlockSize; ++jj)
                    {
                        float x = 0.0;
                        for (size_t kk = 0; kk < KBlockSize; ++kk)
                            x += packed_mat1[ii * KBlockSize + kk] * packed_mat2[kk * NBlockSize + jj];
                        *ptr++ += x;
                    }
            }
        }
    }

    for (size_t i = 0; i < M * N; i++)
    {
        int block = i / (MBlockSize * NBlockSize);
        int numInBlock = i % (MBlockSize * NBlockSize);

        int row = block / (N / NBlockSize) * MBlockSize;
        int col = block % (N / NBlockSize) * NBlockSize;

        mat[(row + numInBlock / NBlockSize) * N + (col + numInBlock % NBlockSize)] = write_cache[i];
    }

    free(packed_mat1);
    free(packed_mat2);
    free(write_cache);
}

// 分块 + 重排 + AVX
void gemm_blocked_packed_avx_1(const size_t M, const size_t N, const size_t K, const float *mat1, const float *mat2, float *mat)
{
    if (mat1 == NULL || mat2 == NULL || mat == NULL)
        return;

    const size_t MBlockSize = M < 128 ? M : 128;
    const size_t KBlockSize = K < 128 ? K : 128;
    const size_t NBlockSize = N < 128 ? N : 128;

    float *packed_mat1 = (float *)aligned_alloc(1024, MBlockSize * KBlockSize * sizeof(float));
    float *packed_mat2 = (float *)aligned_alloc(1024, KBlockSize * NBlockSize * sizeof(float));

    for (size_t i = 0; i < M; i += MBlockSize)
    {
        for (size_t k = 0; k < K; k += KBlockSize)
        {
            // Pack mat1's current block
            for (size_t ii = i, p1_idx = 0; ii < i + MBlockSize; ++ii)
                for (size_t kk = k; kk < k + KBlockSize; ++kk)
                    packed_mat1[p1_idx++] = mat1[ii * K + kk];

            for (size_t j = 0; j < N; j += NBlockSize)
            {
                // Pack mat2's current block
                for (size_t kk = k, p2_idx = 0; kk < k + KBlockSize; ++kk)
                    for (size_t jj = j; jj < j + NBlockSize; ++jj)
                        packed_mat2[p2_idx++] = mat2[kk * N + jj];

                // Compute block product using AVX2
                for (size_t ii = 0; ii < MBlockSize; ii++)
                {
                    for (size_t kk = 0; kk < KBlockSize; kk++)
                        for (size_t jj = 0; jj < NBlockSize; jj += 8)
                        {
                            __m256 sum = _mm256_load_ps(&mat[(i + ii) * N + j + jj]);
                            __m256 a = _mm256_broadcast_ss(&packed_mat1[ii * KBlockSize + kk]);
                            __m256 b = _mm256_load_ps(&packed_mat2[kk * NBlockSize + jj]);
                            sum = _mm256_fmadd_ps(a, b, sum);
                            _mm256_store_ps(&mat[(i + ii) * N + j + jj], sum);
                        }
                }
            }
        }
    }

    free(packed_mat1);
    free(packed_mat2);
}

// 分块 + 重排 + AVX
void gemm_blocked_packed_avx_2(const size_t M, const size_t N, const size_t K, const float *mat1, const float *mat2, float *mat)
{
    if (mat1 == NULL || mat2 == NULL || mat == NULL)
        return;

    const size_t MBlockSize = M < 128 ? M : 128;
    const size_t KBlockSize = K < 128 ? K : 128;
    const size_t NBlockSize = N < 128 ? N : 128;

    float *packed_mat1 = (float *)aligned_alloc(1024, MBlockSize * KBlockSize * sizeof(float));
    float *packed_mat2 = (float *)aligned_alloc(1024, KBlockSize * NBlockSize * sizeof(float));

    for (size_t i = 0; i < M; i += MBlockSize)
    {
        for (size_t k = 0; k < K; k += KBlockSize)
        {
            // Pack mat1's current block
            for (size_t ii = i, p1_idx = 0; ii < i + MBlockSize; ++ii)
                for (size_t kk = k; kk < k + KBlockSize; ++kk)
                    packed_mat1[p1_idx++] = mat1[ii * K + kk];

            for (size_t j = 0; j < N; j += NBlockSize)
            {
                // Pack mat2's current block
                for (size_t kk = k, p2_idx = 0; kk < k + KBlockSize; ++kk)
                    for (size_t jj = j; jj < j + NBlockSize; ++jj)
                        packed_mat2[p2_idx++] = mat2[kk * N + jj];

                // Compute block product using AVX2
                for (size_t ii = 0; ii < MBlockSize; ii++)
                {
                    for (size_t jj = 0; jj < NBlockSize; jj += 64)
                    {
                        __m256 sum0 = _mm256_load_ps(&mat[(i + ii) * N + j + jj]);
                        __m256 sum1 = _mm256_load_ps(&mat[(i + ii) * N + j + jj + 8]);
                        __m256 sum2 = _mm256_load_ps(&mat[(i + ii) * N + j + jj + 16]);
                        __m256 sum3 = _mm256_load_ps(&mat[(i + ii) * N + j + jj + 24]);
                        __m256 sum4 = _mm256_load_ps(&mat[(i + ii) * N + j + jj + 32]);
                        __m256 sum5 = _mm256_load_ps(&mat[(i + ii) * N + j + jj + 40]);
                        __m256 sum6 = _mm256_load_ps(&mat[(i + ii) * N + j + jj + 48]);
                        __m256 sum7 = _mm256_load_ps(&mat[(i + ii) * N + j + jj + 56]);
                        for (size_t kk = 0; kk < KBlockSize; kk++)
                        {
                            __m256 a = _mm256_broadcast_ss(&packed_mat1[ii * KBlockSize + kk]);

                            __m256 b0 = _mm256_load_ps(&packed_mat2[kk * NBlockSize + jj]);
                            __m256 b1 = _mm256_load_ps(&packed_mat2[kk * NBlockSize + jj + 8]);
                            __m256 b2 = _mm256_load_ps(&packed_mat2[kk * NBlockSize + jj + 16]);
                            __m256 b3 = _mm256_load_ps(&packed_mat2[kk * NBlockSize + jj + 24]);
                            __m256 b4 = _mm256_load_ps(&packed_mat2[kk * NBlockSize + jj + 32]);
                            __m256 b5 = _mm256_load_ps(&packed_mat2[kk * NBlockSize + jj + 40]);
                            __m256 b6 = _mm256_load_ps(&packed_mat2[kk * NBlockSize + jj + 48]);
                            __m256 b7 = _mm256_load_ps(&packed_mat2[kk * NBlockSize + jj + 56]);

                            sum0 = _mm256_fmadd_ps(a, b0, sum0);
                            sum1 = _mm256_fmadd_ps(a, b1, sum1);
                            sum2 = _mm256_fmadd_ps(a, b2, sum2);
                            sum3 = _mm256_fmadd_ps(a, b3, sum3);
                            sum4 = _mm256_fmadd_ps(a, b4, sum4);
                            sum5 = _mm256_fmadd_ps(a, b5, sum5);
                            sum6 = _mm256_fmadd_ps(a, b6, sum6);
                            sum7 = _mm256_fmadd_ps(a, b7, sum7);
                        }
                        _mm256_store_ps(&mat[(i + ii) * N + j + jj], sum0);
                        _mm256_store_ps(&mat[(i + ii) * N + j + jj + 8], sum1);
                        _mm256_store_ps(&mat[(i + ii) * N + j + jj + 16], sum2);
                        _mm256_store_ps(&mat[(i + ii) * N + j + jj + 24], sum3);
                        _mm256_store_ps(&mat[(i + ii) * N + j + jj + 32], sum4);
                        _mm256_store_ps(&mat[(i + ii) * N + j + jj + 40], sum5);
                        _mm256_store_ps(&mat[(i + ii) * N + j + jj + 48], sum6);
                        _mm256_store_ps(&mat[(i + ii) * N + j + jj + 56], sum7);
                    }
                }
            }
        }
    }

    free(packed_mat1);
    free(packed_mat2);
}

// IKJ + OpenMP
void gemm_ikj_OpenMP(const size_t M, const size_t N, const size_t K, const float *mat1, const float *mat2, float *mat)
{
    if (mat1 == NULL || mat2 == NULL || mat == NULL)
        return;
#pragma omp parallel for
    for (size_t i = 0; i < M; i++)
        for (size_t k = 0; k < K; k++)
            for (size_t j = 0; j < N; j++)
                mat[i * N + j] += mat1[i * K + k] * mat2[k * N + j];
}

// 分块 + OpenMP
void gemm_blocked_OpenMP(const size_t M, const size_t N, const size_t K, const float *mat1, const float *mat2, float *mat)
{
    if (mat1 == NULL || mat2 == NULL || mat == NULL)
        return;

    const size_t MBlockSize = M < 128 ? M : 128;
    const size_t KBlockSize = K < 128 ? K : 128;
    const size_t NBlockSize = N < 128 ? N : 128;
#pragma omp parallel for collapse(3)
    for (size_t i = 0; i < M; i += MBlockSize)
        for (size_t k = 0; k < K; k += KBlockSize)
            for (size_t j = 0; j < N; j += NBlockSize)
                for (size_t ii = i; ii < i + MBlockSize; ii++)
                    for (size_t kk = k; kk < k + KBlockSize; kk++)
                        for (size_t jj = j; jj < j + NBlockSize; jj++)
                            mat[ii * N + jj] += mat1[ii * K + kk] * mat2[kk * N + jj];
}

// 分块 + 重排 + OpenMP
void gemm_blocked_packed_OpenMP(const size_t M, const size_t N, const size_t K, const float *mat1, const float *mat2, float *mat)
{
    if (mat1 == NULL || mat2 == NULL || mat == NULL)
        return;

    const size_t MBlockSize = M < 128 ? M : 128;
    const size_t KBlockSize = K < 128 ? K : 128;
    const size_t NBlockSize = N < 128 ? N : 128;

#pragma omp parallel
    {
        float *packed_mat1 = (float *)aligned_alloc(1024, MBlockSize * KBlockSize * sizeof(float));
        float *packed_mat2 = (float *)aligned_alloc(1024, KBlockSize * NBlockSize * sizeof(float));

#pragma omp for collapse(3)
        for (size_t i = 0; i < M; i += MBlockSize)
        {
            for (size_t j = 0; j < N; j += NBlockSize)
            {
                for (size_t k = 0; k < K; k += KBlockSize)
                {
                    // Pack mat1's current block
                    for (size_t ii = i, p1_idx = 0; ii < i + MBlockSize; ++ii)
                        for (size_t kk = k; kk < k + KBlockSize; ++kk)
                            packed_mat1[p1_idx++] = mat1[ii * K + kk];

                    // Pack mat2's current block
                    for (size_t kk = k, p2_idx = 0; kk < k + KBlockSize; ++kk)
                        for (size_t jj = j; jj < j + NBlockSize; ++jj)
                            packed_mat2[p2_idx++] = mat2[kk * N + jj];

                    // Compute block product
                    for (size_t ii = 0; ii < MBlockSize; ++ii)
                        for (size_t kk = 0; kk < KBlockSize; ++kk)
                            for (size_t jj = 0; jj < NBlockSize; ++jj)
                                mat[(i + ii) * N + j + jj] += packed_mat1[ii * KBlockSize + kk] * packed_mat2[kk * NBlockSize + jj];
                }
            }
        }

        free(packed_mat1);
        free(packed_mat2);
    }
}

// 分块 + 重排 + AVX + OpenMP
void gemm_blocked_packed_avx_1_OpenMP(const size_t M, const size_t N, const size_t K, const float *mat1, const float *mat2, float *mat)
{
    if (mat1 == NULL || mat2 == NULL || mat == NULL)
        return;

    const size_t MBlockSize = M < 128 ? M : 128;
    const size_t KBlockSize = K < 128 ? K : 128;
    const size_t NBlockSize = N < 128 ? N : 128;

#pragma omp parallel
    {
        float *packed_mat1 = (float *)aligned_alloc(1024, MBlockSize * KBlockSize * sizeof(float));
        float *packed_mat2 = (float *)aligned_alloc(1024, KBlockSize * NBlockSize * sizeof(float));

#pragma omp for collapse(2) schedule(static)
        for (size_t i = 0; i < M; i += MBlockSize)
        {
            for (size_t j = 0; j < N; j += NBlockSize)
            {
                for (size_t k = 0; k < K; k += KBlockSize)
                {
                    // Pack mat1's current block
                    for (size_t ii = i, p1_idx = 0; ii < i + MBlockSize; ++ii)
                        for (size_t kk = k; kk < k + KBlockSize; ++kk)
                            packed_mat1[p1_idx++] = mat1[ii * K + kk];

                    // Pack mat2's current block
                    for (size_t kk = k, p2_idx = 0; kk < k + KBlockSize; ++kk)
                        for (size_t jj = j; jj < j + NBlockSize; ++jj)
                            packed_mat2[p2_idx++] = mat2[kk * N + jj];

                    // Compute block product using AVX2
                    for (size_t ii = 0; ii < MBlockSize; ii++)
                    {
                        for (size_t kk = 0; kk < KBlockSize; kk++)
                            for (size_t jj = 0; jj < NBlockSize; jj += 8)
                            {
                                __m256 sum = _mm256_load_ps(&mat[(i + ii) * N + j + jj]);
                                __m256 a = _mm256_broadcast_ss(&packed_mat1[ii * KBlockSize + kk]);
                                __m256 b = _mm256_load_ps(&packed_mat2[kk * NBlockSize + jj]);
                                sum = _mm256_fmadd_ps(a, b, sum);
                                _mm256_store_ps(&mat[(i + ii) * N + j + jj], sum);
                            }
                    }
                }
            }
        }

        free(packed_mat1);
        free(packed_mat2);
    }
}

// 分块 + 重排 + AVX + OpenMP
void gemm_blocked_packed_avx_2_OpenMP(const size_t M, const size_t N, const size_t K, const float *mat1, const float *mat2, float *mat)
{
    if (mat1 == NULL || mat2 == NULL || mat == NULL)
        return;

    const size_t MBlockSize = M < 128 ? M : 128;
    const size_t KBlockSize = K < 128 ? K : 128;
    const size_t NBlockSize = N < 128 ? N : 128;

#pragma omp parallel
    {
        float *packed_mat1 = (float *)aligned_alloc(1024, MBlockSize * KBlockSize * sizeof(float));
        float *packed_mat2 = (float *)aligned_alloc(1024, KBlockSize * NBlockSize * sizeof(float));

#pragma omp for collapse(2) schedule(dynamic)
        for (size_t i = 0; i < M; i += MBlockSize)
        {
            for (size_t j = 0; j < N; j += NBlockSize)
            {
                for (size_t k = 0; k < K; k += KBlockSize)
                {
                    // Pack mat1's current block
                    for (size_t ii = i, p1_idx = 0; ii < i + MBlockSize; ++ii)
                        for (size_t kk = k; kk < k + KBlockSize; ++kk)
                            packed_mat1[p1_idx++] = mat1[ii * K + kk];

                    // Pack mat2's current block
                    for (size_t kk = k, p2_idx = 0; kk < k + KBlockSize; ++kk)
                        for (size_t jj = j; jj < j + NBlockSize; ++jj)
                            packed_mat2[p2_idx++] = mat2[kk * N + jj];

                    // Compute block product using AVX2
                    for (size_t ii = 0; ii < MBlockSize; ii++)
                    {
                        for (size_t jj = 0; jj < NBlockSize; jj += 64)
                        {
                            __m256 sum0 = _mm256_load_ps(&mat[(i + ii) * N + j + jj]);
                            __m256 sum1 = _mm256_load_ps(&mat[(i + ii) * N + j + jj + 8]);
                            __m256 sum2 = _mm256_load_ps(&mat[(i + ii) * N + j + jj + 16]);
                            __m256 sum3 = _mm256_load_ps(&mat[(i + ii) * N + j + jj + 24]);
                            __m256 sum4 = _mm256_load_ps(&mat[(i + ii) * N + j + jj + 32]);
                            __m256 sum5 = _mm256_load_ps(&mat[(i + ii) * N + j + jj + 40]);
                            __m256 sum6 = _mm256_load_ps(&mat[(i + ii) * N + j + jj + 48]);
                            __m256 sum7 = _mm256_load_ps(&mat[(i + ii) * N + j + jj + 56]);
                            for (size_t kk = 0; kk < KBlockSize; kk++)
                            {
                                __m256 a = _mm256_broadcast_ss(&packed_mat1[ii * KBlockSize + kk]);

                                __m256 b0 = _mm256_load_ps(&packed_mat2[kk * NBlockSize + jj]);
                                __m256 b1 = _mm256_load_ps(&packed_mat2[kk * NBlockSize + jj + 8]);
                                __m256 b2 = _mm256_load_ps(&packed_mat2[kk * NBlockSize + jj + 16]);
                                __m256 b3 = _mm256_load_ps(&packed_mat2[kk * NBlockSize + jj + 24]);
                                __m256 b4 = _mm256_load_ps(&packed_mat2[kk * NBlockSize + jj + 32]);
                                __m256 b5 = _mm256_load_ps(&packed_mat2[kk * NBlockSize + jj + 40]);
                                __m256 b6 = _mm256_load_ps(&packed_mat2[kk * NBlockSize + jj + 48]);
                                __m256 b7 = _mm256_load_ps(&packed_mat2[kk * NBlockSize + jj + 56]);

                                sum0 = _mm256_fmadd_ps(a, b0, sum0);
                                sum1 = _mm256_fmadd_ps(a, b1, sum1);
                                sum2 = _mm256_fmadd_ps(a, b2, sum2);
                                sum3 = _mm256_fmadd_ps(a, b3, sum3);
                                sum4 = _mm256_fmadd_ps(a, b4, sum4);
                                sum5 = _mm256_fmadd_ps(a, b5, sum5);
                                sum6 = _mm256_fmadd_ps(a, b6, sum6);
                                sum7 = _mm256_fmadd_ps(a, b7, sum7);
                            }
                            _mm256_store_ps(&mat[(i + ii) * N + j + jj], sum0);
                            _mm256_store_ps(&mat[(i + ii) * N + j + jj + 8], sum1);
                            _mm256_store_ps(&mat[(i + ii) * N + j + jj + 16], sum2);
                            _mm256_store_ps(&mat[(i + ii) * N + j + jj + 24], sum3);
                            _mm256_store_ps(&mat[(i + ii) * N + j + jj + 32], sum4);
                            _mm256_store_ps(&mat[(i + ii) * N + j + jj + 40], sum5);
                            _mm256_store_ps(&mat[(i + ii) * N + j + jj + 48], sum6);
                            _mm256_store_ps(&mat[(i + ii) * N + j + jj + 56], sum7);
                        }
                    }
                }
            }
        }

        free(packed_mat1);
        free(packed_mat2);
    }
}

void gemm_cblas(const size_t M, const size_t N, const size_t K, const float *mat1, const float *mat2, float *mat)
{
    if (mat1 == NULL || mat2 == NULL || mat == NULL)
        return;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, mat1, K, mat2, N, 0.0f, mat, N);
}
