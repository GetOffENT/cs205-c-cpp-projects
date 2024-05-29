#include "benchmark/benchmark.h"
#include <functional>
#include "matrix.h" 

using namespace std;
using func_t = function<bool(const Matrix<float>&, const Matrix<float>&, Matrix<float>&)>;

// Executor 类用于执行测试
class Executor {
public:
    explicit Executor(func_t func) : func(std::move(func)) {}

    void execute(benchmark::State &state) {
        const size_t N = state.range(0);

        Matrix<float> lhs(N, N);
        Matrix<float> rhs(N, N);
        Matrix<float> dst(N, N);

        lhs.randomize();
        rhs.randomize();

        for (auto _ : state) {
            func(lhs, rhs, dst);
            benchmark::DoNotOptimize(dst.data);
            benchmark::DoNotOptimize(dst.data_device);
            benchmark::ClobberMemory();
        }
        state.SetComplexityN(state.range(0));
    }
private:
    func_t func;
};

#define ADD_BENCHMARK(FUNC, BENCHMARK_NAME) \
    static void BENCHMARK_NAME(benchmark::State &state) { \
        Executor(FUNC).execute(state); \
        state.SetComplexityN(state.range(0)); \
    } \
    BENCHMARK(BENCHMARK_NAME)->DenseRange(32, 4096, 32)->Complexity(benchmark::oNCubed);

ADD_BENCHMARK(mulMatrixCPU, BM_MulMatrixCPU);
ADD_BENCHMARK(mulMatrixGPU, BM_MulMatrixGPU);

BENCHMARK_MAIN();
// ./benchmark_test --benchmark_format=<console|json|csv>