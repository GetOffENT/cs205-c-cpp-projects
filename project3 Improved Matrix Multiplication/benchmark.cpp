#include <benchmark/benchmark.h>
#include <random>
#include <utility>
#include <functional>
#include <immintrin.h>

#include "matrix.h"

using namespace std;

static random_device rd;
static mt19937 rng{rd()};
static uniform_real_distribution<float> distrib(-10.0f, 10.0f);

class MatrixWrapper
{
    size_t size;
    float *p_data;

public:
    MatrixWrapper(const size_t rows, const size_t cols)
    {
        size = rows * cols;
        p_data = static_cast<float *>(aligned_alloc(1024, size * sizeof(p_data[0])));
        // p_data = static_cast<float*>(_mm_malloc(size * sizeof(float), 32));
        zero();
    }

    ~MatrixWrapper()
    {
        free(p_data);
    }

    void randomize()
    {
        for (size_t i = 0; i < size; i++)
            p_data[i] = distrib(rng);
    }

    void zero()
    {
        for (size_t i = 0; i < size; i++)
            p_data[i] = 0;
    }

    float *data()
    {
        return p_data;
    }
};

class Executor
{
    // using func_t = function<matrix *(const matrix *, const matrix *)>;
    using func_t = function<void(size_t, size_t, size_t, const float *, const float *, float *)>;
    func_t func;

public:
    explicit Executor(func_t func) : func(std::move(func)) {}

    void execute(benchmark::State &state)
    {
        const size_t N = state.range(0);

        MatrixWrapper lhs(N, N), rhs(N, N), dst(N, N);
        lhs.randomize();
        rhs.randomize();

        for (auto _ : state)
        {
            func(N, N, N, lhs.data(), rhs.data(), dst.data());
            benchmark::DoNotOptimize(dst.data()[rng() % (N * N)]);
            benchmark::ClobberMemory();
        }
    }
};

#define ADD_BENCHMARK(FUNC, BENCHMARK_NAME)             \
    static void BENCHMARK_NAME(benchmark::State &state) \
    {                                                   \
        Executor(FUNC).execute(state);                  \
        state.SetComplexityN(state.range(0));           \
    }                                                   \
    BENCHMARK(BENCHMARK_NAME)->Arg(16)->Arg(128)->Arg(1 << 10)->Arg(1 << 13)->Complexity(benchmark::oNCubed);
// BENCHMARK(BENCHMARK_NAME)->DenseRange(128, 8192, 128)->Complexity(benchmark::oNCubed);

// ADD_BENCHMARK(plain_gemm_ijk, BM_Plain_GEMM_IJK)
// ADD_BENCHMARK(plain_gemm_ikj, BM_Plain_GEMM_IKJ_2)
// ADD_BENCHMARK(plain_gemm_kij, BM_Plain_GEMM_KIJ)
// ADD_BENCHMARK(plain_gemm_jik, BM_Plain_GEMM_JIK)
// ADD_BENCHMARK(plain_gemm_kji, BM_Plain_GEMM_KJI)
// ADD_BENCHMARK(plain_gemm_jki, BM_Plain_GEMM_JKI)
// ADD_BENCHMARK(gemm_blocked, GEMM_BLOCKED)
// ADD_BENCHMARK(gemm_blocked_packed, GEMM_BLOCKED_PACKED)
// ADD_BENCHMARK(gemm_blocked_packed_writecache, GEMM_BLOCKED_PACKED_WRITECACHE)
// ADD_BENCHMARK(gemm_blocked_packed_avx_1, GEMM_BLOCKED_PACKED_AVX)
// ADD_BENCHMARK(gemm_blocked_packed_avx_2, GEMM_BLOCKED_PACKED_AVX_2)
// ADD_BENCHMARK(gemm_ikj_OpenMP, GEMM_IKJ_OPENMP)
// ADD_BENCHMARK(gemm_blocked_OpenMP, GEMM_BLOCKED_OPENMP)
// ADD_BENCHMARK(gemm_blocked_packed_OpenMP, GEMM_BLOCKED_PACKED_OPENMP)
// ADD_BENCHMARK(gemm_blocked_packed_avx_1_OpenMP, GEMM_BLOCKED_PACKED_AVX_1_OPENMP)
// ADD_BENCHMARK(gemm_blocked_packed_avx_2_OpenMP, GEMM_BLOCKED_PACKED_AVX_2_OPENMP)
ADD_BENCHMARK(gemm_cblas, BM_GEMM_CBLAS)

BENCHMARK_MAIN();