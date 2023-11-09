#include <iostream>
#include <string>

#include "calc_time.h"

const int N = 1024;
const int M = 10;
int matA[N][N];
int matB[N][N];
int matC[N][N];
int matD[N][N];
int tmp[N][N];

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

/**
 * 按照最原始的方式进行矩阵乘法
*/
void mul1() {
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

/**
 * 利用 B 的转置
*/
void mul2_transpose() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            tmp[i][j] = matB[j][i];
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int r = 0;
            for (int k = 0; k < N; k++) {
                r += matA[i][k] * tmp[j][k];
            }
            matD[i][j] = r;
        }
    }
}

/**
 * 利用矩阵分块来加速
 * 但是单纯的分块并没有达到理想的效果，反而会降低效率
*/
void mul3_partition() {
    int block_size = N / 4;
    for (int i = 0; i < N; i += block_size) {
        for (int j = 0; j < N; j += block_size) {
            for (int k = 0; k < N; k += block_size) {
                for (int ii = i; ii < i + block_size; ii++) {
                    for (int jj = j; jj < j + block_size; jj++) {
                        for (int kk = k; kk < k + block_size; kk++) {
                            matD[ii][jj] += matA[ii][kk] * matB[kk][jj];
                        }
                    }
                }
            }
        }
    }
}

/**
 * 转置 + 分块
 * 这里的提升并没有很大，可能是因为矩阵的大小，导致分块并没有达到较高的效率
 * 原本连续的几行应该可以一次性在同一页内访问到，而强行分块后，相同一行的数据将在不同的时间点被访问
 * 这样做反而丢失了时间的局部性
*/
void mul4_partition_transpose() {
    int block_size = N / 4;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            tmp[i][j] = matB[j][i];
        }
    }
    for (int i = 0; i < N; i += block_size) {
        for (int j = 0; j < N; j += block_size) {
            for (int k = 0; k < N; k += block_size) {
                for (int ii = i; ii < i + block_size; ii++) {
                    for (int jj = j; jj < j + block_size; jj++) {
                        for (int kk = k; kk < k + block_size; kk++) {
                            matD[ii][jj] += matA[ii][kk] * tmp[jj][kk];
                        }
                    }
                }
            }
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

void reset_mat_d() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matD[i][j] = 0;
        }
    }
}

int main() {
    init_mat();
    long long mul1_time = calc_time(mul1);
    std::cout << "mul1: " << mul1_time << "us"<< std::endl;

    reset_mat_d();
    long long mul2_time = calc_time(mul2_transpose);
    std::cout << "mul2: " << mul2_time << "us, check: " << check() << std::endl;

    reset_mat_d();
    long long mul3_time = calc_time(mul3_partition);
    std::cout << "mul3: " << mul3_time << "us, check: " << check() << std::endl;

    reset_mat_d();
    long long mul4_time = calc_time(mul4_partition_transpose);
    std::cout << "mul4: " << mul4_time << "us, check: " << check() << std::endl;

    // msvc
    // mul1: 5861801us
    // mul2: 2763300us
    // mul3: 4446739us
    // mul4: 2916981us

    // gcc
    // mul1: 3761237us
    // mul2: 2342269us, check: 1
    // mul3: 5742044us, check: 1
    // mul4: 3230772us, check: 1

    return 0;
}

// g++ -g locality.cpp -o locality
