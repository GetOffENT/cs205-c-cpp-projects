#include "matrix.hpp"

int main(){
    Matrix<float> m1({
        {{1.0f, 0.5f, 0.2f}, {2.0f, 1.0f, 0.5f}, {3.0f, 1.5f, 0.8f}},
        {{4.0f, 2.0f, 1.0f}, {5.0f, 2.5f, 1.3f}, {6.0f, 3.0f, 1.6f}},
        {{7.0f, 3.5f, 1.8f}, {8.0f, 4.0f, 2.1f}, {9.0f, 4.5f, 2.4f}}
    });
    Matrix<float> m2({
        {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
        {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}},
        {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}
    });

    // 执行矩阵乘法
    Matrix<float> result = m1 * m2;
    result.print();
    return 0;
}

// valgrind --leak-check=full --track-origins=yes --verbose ./matrix