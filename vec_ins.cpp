#include <iostream>
#include <immintrin.h>
#include <xmmintrin.h>

#include "calc_time.h"

const int N = 1024;
const int M = 10;
int matA[N][N];
int matB[N][N];
int matC[N][N];
int matD[N][N];

void init_mat() {
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matA[i][j] = rand() % M;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matB[i][j] = rand() % M;
        }
    }  
}

const int BAND_WIDTH = 256 / 32;
const int TIMES = N / BAND_WIDTH;

int BUFF[BAND_WIDTH];

/**
 * the matB here is transposed
*/
void mul_one_element(int matA[N][N], int matB[N][N], int matC[N][N], int row_a, int col_b) {
    // 定义 256 位的寄存器
    __m256i line_a, line_b, v = _mm256_setzero_si256();

    for (int i = 0; i < TIMES; i++) {
        // 加载 256 位，即连续的 8 个 32 位
        line_a = _mm256_loadu_si256((__m256i*) &matA[row_a][i * BAND_WIDTH]);
        line_b = _mm256_loadu_si256((__m256i*) &matB[col_b][i * BAND_WIDTH]);
        // v += a * b
        v = _mm256_add_epi32(v, _mm256_mullo_epi32(line_a, line_b));
    }

    // v 中 8 个 32 位整数分别是按组成绩的累加，需要将这 8 个元素再求和，得到一个 32 位整数
    // 这个最终求和的整数才是矩阵 c 该处的值
    _mm256_storeu_si256((__m256i*) BUFF, v);
    
    for (int i = 0; i < BAND_WIDTH; i++) {
        matC[row_a][col_b] += BUFF[i];
    }
}

void mul_one_row(int matA[N][N], int matB[N][N], int matC[N][N], int row_a) {
    for (int i = 0; i < N; i++) {
        mul_one_element(matA, matB, matC, row_a, i);
    }
}

/**
 * 使用 intel 的 avx 指令集
*/
void mul_simd() {
    for (int i = 0; i < N; i++) {
        mul_one_row(matA, matB, matC, i);
    }
}

void mul_transpose() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int r = 0;
            for (int k = 0; k < N; k++) {
                r += matA[i][k] * matB[j][k];
            }
            matD[i][j] = r;
        }
    }
}

bool check() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (matC[i][j] != matD[i][j]) {
                return false;
            }
        }
    }
    return true;
}

int main () {
    init_mat();

    // 默认直接采用转置的方式，不再加上转置的预处理了
    // 将 B 转置后再乘
    long long time_transpose = calc_time(mul_transpose);
    std::cout << "mul_transpose: " << time_transpose << "us" << std::endl;
    // 2213664us

    // 将 B 转置后，使用 256 位向量指令进行乘法加速
    long long time_simd = calc_time(mul_simd);
    std::cout << "mul2: " << time_simd << "us" << std::endl;
    // 718855us

    // 检查得到的矩阵是否正确
    std::cout << "check: " << check() << std::endl;
    return 0;
}

// g++ -mavx2 -std=c++17 -g -o vec_ins ./vec_ins.cpp

