#include <iostream>
#include <thread>
#include <immintrin.h>
#include <xmmintrin.h>

#include "calc_time.h"

const int N = 1024;
const int M = 10;
int matA[N][N];
int matB[N][N];
/**
 * 用最原始的方式计算
*/
int matC[N][N];
/**
 * 加速计算
 */
int matD[N][N];
int tmp[N][N];

void output(int mat[N][N], int up) {
    for (int i = 0; i < up; i++) {
        for (int j = 0; j < up; j++) {
            std::cout << mat[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void baseline() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int r = 0;
            for (int k = 0; k < N; k++) {
                r += matA[i][k] * matB[k][j];
            }
            matC[i][j] = r;
        }
    }
}

bool check() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (matC[i][j] != matD[i][j]) {
                std::cout << i << " " << j << " " << matC[i][j] << " " << matD[i][j] << std::endl;
                return false;
            }
        }
    }
    return true;
}


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

void mul_one_element(int mat_a[N][N], int mat_b[N][N], int mat_c[N][N], int row_a, int col_b) {
    // 定义 256 位的寄存器
    __m256i line_a, line_b, v = _mm256_setzero_si256();

    for (int i = 0; i < TIMES; i++) {
        // 加载 256 位，即连续的 8 个 32 位
        line_a = _mm256_loadu_si256((__m256i*) &mat_a[row_a][i * BAND_WIDTH]);
        line_b = _mm256_loadu_si256((__m256i*) &mat_b[col_b][i * BAND_WIDTH]);
        // v += a * b
        v = _mm256_add_epi32(v, _mm256_mullo_epi32(line_a, line_b));
    }

    // v 中 8 个 32 位整数分别是按组成绩的累加，需要将这 8 个元素再求和，得到一个 32 位整数
    // 这个最终求和的整数才是矩阵 c 该处的值
    _mm256_storeu_si256((__m256i*) BUFF, v);
    
    for (int i = 0; i < BAND_WIDTH; i++) {
        mat_c[row_a][col_b] += BUFF[i];
    }
}

void mul_one_row(int mat_a[N][N], int mat_b[N][N], int mat_c[N][N], int row_a) {
    for (int i = 0; i < N; i++) {
        mul_one_element(mat_a, mat_b, mat_c, row_a, i);
    }
}

void calc_transpose() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            tmp[i][j] = matB[j][i];
        }
    }
}

void batch(int start, int block_size) {
    for (int i = start; i < start + block_size; i++) {
        mul_one_row(matA, tmp, matD, i);
    }
}

void accelerate() {
    const int thread_num = 8;
    const int block_size = N / thread_num;
    std::thread threads[thread_num];
    calc_transpose();
    for (int i = 0; i < thread_num; i++) {
        threads[i] = std::thread(batch, i * block_size, block_size);
    }
    for (int i = 0; i < thread_num; i++) {
        threads[i].join();
    }
}

int main() {
    init_mat();

    long long baseline_time = calc_time(baseline);
    std::cout << "baseline: " << baseline_time << "us" << std::endl;

    long long accelerate_time = calc_time(accelerate);
    std::cout << "accelerate: " << accelerate_time << "us" << std::endl;

    std::cout << "check: " << check() << std::endl;

    // output(matC, 16);
    // std::cout << std::endl;
    // output(matD, 16);

    return 0;
}

// g++ -mavx2 -std=c++17 -g all.cpp -o all

