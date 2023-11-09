#include <iostream>
#include <thread>

#include "calc_time.h"

const int N = 1024;
const int M = 10;
int matA[N][N];
int matB[N][N];
int matC[N][N];
int matD[N][N];

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

void batch(int start, int block_size) {
    for (int i = start; i < start + block_size; i++) {
        for (int j = 0; j < N; j++) {
            int r = 0;
            for (int k = 0; k < N; k++) {
                r += matA[i][k] * matB[k][j];
            }
            matD[i][j] = r;
        }
    }
}

void multi_thread() {
    const int thread_num = 8;
    const int block_size = N / thread_num;
    std::thread threads[thread_num];
    for (int i = 0; i < thread_num; i++) {
        threads[i] = std::thread(batch, i * block_size, block_size);
    }
    for (int i = 0; i < thread_num; i++) {
        threads[i].join();
    }
}

int main() {
    init_mat();

    long long mul2_time = calc_time(multi_thread);
    std::cout << "multi_thread: " << mul2_time << "us" << std::endl;

    long long mul1_time = calc_time(mul1);
    std::cout << "baseline: " << mul1_time << "us" << std::endl;

    //     baseline: 6919118us
    // multi_thread: 1651639us

    std::cout << "check: " << check() << std::endl;
    return 0;
}

// g++ -std=c++17 -g -o multi-thread multi-thread.cpp
