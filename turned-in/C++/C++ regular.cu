/*
# ---------------------------------------------------------------------------
# C++ Reg.c++
# Regular C++ implementation of the CUDA code
# Using the GPU to accelerate an electrical power grid visualization
#
# Date:                   03/14/2023
# Authors:                Pragati Dode, Breanna Powell, and William Selke
#
# +++++++++++++++++ DETAILS ABOUT SYSTEM ++++++++++++++
# IDEs:                   Visual Studio Code; PyCharm
# Processor Used:         11th Gen Intel(R) Core(TM) i7-11700 @ 2.50GHz 2.50 GHz
# GPU Used:               NVIDIA GeForce RTX 3060 Ti
# Device Architecture:    Ampere
#
# +++++++++++++++++ COMPILE INSTRUCTIONS +++++++++++++++++
# nvcc "C++ regular.cu" -o regular
 */

#include <iostream>
#include <chrono>
#include "cuda.h"

// INPUT
#define INPUT_SIZE 4096

// OUTPUT
#define POOL_SIZE 2 // NO TOUCHY
#define TOTAL_POOL 64

// #define debug

void populateMatrix(double* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = 1;
        }
    }
}

void printMatrix(double* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // pad with spaces to make it look nice
            double temp = matrix[i * cols + j];
            if (temp < 10) {
                std::cout << " ";
            }
            std::cout << temp << " ";
        }
        std::cout << std::endl;
    }
}

__global__ void cudaPool(double* input, double* output, int last_pool) {
    // set up the shared memory
    __shared__ double shared[POOL_SIZE];
    int x1 = blockIdx.y * POOL_SIZE;
    int y1 = blockIdx.x * POOL_SIZE + threadIdx.x;
    int BLOCK = blockIdx.x * gridDim.y + blockIdx.y;

    int x2 = x1 + 1;
    int y2 = y1;

    // save the sum of the two values
    shared[threadIdx.x] = input[x1 * INPUT_SIZE + y1] + input[x2 * INPUT_SIZE + y2];

    // wait for all threads to finish
    __syncthreads();

    // save the sum of the two values only first thread saves
    if (threadIdx.x == 0) {
        if (last_pool == 1) {
            output[BLOCK] = shared[0] + shared[1];
        } else {
            input[blockIdx.x * INPUT_SIZE + blockIdx.y] = shared[0] + shared[1];
        }
    }
    __syncthreads();
}

int main() {
    // allocate memory on host
    double* inputMatrix = (double*)(malloc(INPUT_SIZE * INPUT_SIZE * sizeof(double)));
    double* outputMatrix = (double*)(malloc(INPUT_SIZE / TOTAL_POOL * INPUT_SIZE / TOTAL_POOL * sizeof(double)));

    // populate input matrix
    populateMatrix(inputMatrix, INPUT_SIZE, INPUT_SIZE);

#ifdef debug
    std::cout << "Input Matrix:" << std::endl;
    printMatrix(inputMatrix, INPUT_SIZE, INPUT_SIZE);
#endif

    // allocate memory on device
    double *d_inputMatrix;
    double *d_outputMatrix;
    cudaMalloc((void **) &d_inputMatrix, INPUT_SIZE * INPUT_SIZE * sizeof(double));
    cudaMalloc((void **) &d_outputMatrix, INPUT_SIZE / TOTAL_POOL * INPUT_SIZE / TOTAL_POOL * sizeof(double));

    // start total time
    auto start = std::chrono::high_resolution_clock::now();

    // copy input matrix to device
    cudaMemcpy(d_inputMatrix, inputMatrix, INPUT_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    int counter = 0;
    // start calculating time
    auto start2 = std::chrono::high_resolution_clock::now();
    // call kernel
    for (int i = POOL_SIZE; i <= TOTAL_POOL; i *= POOL_SIZE) {
        dim3 dimGrid(INPUT_SIZE / i, INPUT_SIZE / i);
        dim3 dimBlock(POOL_SIZE);
        // check for last pool
        int last_pool = 0;
        if (i == TOTAL_POOL) {
            last_pool = 1;
        }

        cudaPool<<<dimGrid, dimBlock>>>(d_inputMatrix, d_outputMatrix, last_pool);
        counter++;

#ifdef debug
        printf("Pool %d\n", counter);
        cudaMemcpy(inputMatrix, d_inputMatrix, INPUT_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
        printMatrix(inputMatrix, INPUT_SIZE, INPUT_SIZE);
#endif
    }
    // end calculating time
    auto end2 = std::chrono::high_resolution_clock::now();

    // copy output matrix to host
    cudaMemcpy(outputMatrix, d_outputMatrix, INPUT_SIZE / TOTAL_POOL * INPUT_SIZE / TOTAL_POOL * sizeof(double), cudaMemcpyDeviceToHost);
    // end total time
    auto end = std::chrono::high_resolution_clock::now();

#ifdef debug
    std::cout << "Output Matrix:" << std::endl;
    printMatrix(outputMatrix, INPUT_SIZE / TOTAL_POOL, INPUT_SIZE / TOTAL_POOL);
#endif

    // print total, calculating, and copying times in microseconds
    std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds" << std::endl;
    std::cout << "Calculating time: " << std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count() << " microseconds" << std::endl;
    std::cout << "Copying time: " << std::chrono::duration_cast<std::chrono::microseconds>((end - start) - (end2 - start2)).count() << " microseconds" << std::endl;
}
