cmake_minimum_required(VERSION 3.14)
project(my_project)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable CUDA
find_package(CUDA REQUIRED)
enable_language(CUDA)

enable_testing()  


add_executable(
    mul_test
    mul_test.cu
    mul.cu
)

target_link_libraries(
  mul_test
  GTest::gtest_main
  openblas
  cublas
)

target_compile_options(mul_test PRIVATE -O3)

include(GoogleTest)
gtest_discover_tests(mul_test)