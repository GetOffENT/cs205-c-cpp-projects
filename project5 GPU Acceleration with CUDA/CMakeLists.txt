cmake_minimum_required(VERSION 3.14)
project(Project LANGUAGES CXX CUDA)

# Enable CUDA
find_package(CUDA REQUIRED)
enable_language(CUDA)

# Set CUDA architectures (you can adjust these according to the target devices)
set(CUDA_ARCHITECTURES 75)

# Add Google Benchmark
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests")
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON CACHE BOOL "Allow benchmark to download dependencies")
add_subdirectory(tools/benchmark)

# Add the source files
add_executable(benchmark_test src/benchmark.cu src/mul.cu)

# Set CUDA properties for the target
set_target_properties(benchmark_test PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")

# Set compile options for optimization
target_compile_options(benchmark_test PRIVATE -O3)

# Link CUDA and Google Benchmark
target_link_libraries(benchmark_test benchmark::benchmark cuda cublas openblas)

# cmake -DCMAKE_BUILD_TYPE=Release ..