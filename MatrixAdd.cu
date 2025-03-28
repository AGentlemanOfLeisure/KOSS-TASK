#include <iostream>
#include <cuda_runtime.h>

// Matrix size
#define N 32
#define BLOCK_SIZE 16

// CUDA Kernel for matrix addition
__global__ void addMatrices(int *A, int *B, int *C, int width) {
    // Calculate row and column index of the element in the result matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) {
        int index = row * width + col;
        C[index] = A[index] + B[index];
    }
}

int main() {
    // Matrix size
    int width = N;
    int size = width * width * sizeof(int);

    // Allocate memory for matrices A, B, and C
    int *A, *B, *C;
    int *d_A, *d_B, *d_C;
    
    // Allocate memory on the host
    A = (int*)malloc(size);
    B = (int*)malloc(size);
    C = (int*)malloc(size);

    // Initialize matrices A and B
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            A[i * width + j] = i;  // A[i][j] = i
            B[i * width + j] = j;  // B[i][j] = j
        }
    }

    // Allocate memory on the device
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (width + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Create CUDA events to measure time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start time
    cudaEventRecord(start, 0);

    // Launch the kernel
    addMatrices<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    // Record the stop time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy the result matrix C from device to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Print the result matrix C
    std::cout << "Matrix C (A + B):" << std::endl;
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            std::cout << C[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    // Print the time taken to execute the kernel
    std::cout << "Time taken for kernel execution: " << milliseconds << " ms" << std::endl;

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(A);
    free(B);
    free(C);

    return 0;
}
