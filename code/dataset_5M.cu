#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

#define DATA_SIZE 5000000
#define BLOCK_SIZE 256

__global__ void processData(const float *input, float *output, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        output[i] = input[i] * 2.0f;
    }
}

int main()
{
    int N = DATA_SIZE;
    size_t size = N * sizeof(float);

    float *h_input = (float *)malloc(size);
    float *h_output_cpu = (float *)malloc(size);
    float *h_output_gpu = (float *)malloc(size);

    if (h_input == NULL || h_output_cpu == NULL || h_output_gpu == NULL)
    {
        printf("Host memory allocation failed.\n");
        return 1;
    }

    // Generate the numeric dataset
    for (int i = 0; i < N; i++)
    {
        h_input[i] = (float)(i % 1000);
    }

    // CPU processing
    auto cpuStart = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; i++)
    {
        h_output_cpu[i] = h_input[i] * 2.0f;
    }

    auto cpuEnd = std::chrono::high_resolution_clock::now();

    double cpuTime =
        std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

    // GPU memory
    float *d_input;
    float *d_output;

    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);

    cudaEvent_t totalStart, totalStop;
    cudaEvent_t kernelStart, kernelStop;

    cudaEventCreate(&totalStart);
    cudaEventCreate(&totalStop);
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);

    cudaEventRecord(totalStart);

    // CPU -> GPU
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // CUDA kernel
    cudaEventRecord(kernelStart);

    processData<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, N);

    cudaEventRecord(kernelStop);
    cudaEventSynchronize(kernelStop);

    float kernelTime = 0.0f;
    cudaEventElapsedTime(&kernelTime, kernelStart, kernelStop);

    // GPU -> CPU
    cudaMemcpy(h_output_gpu, d_output, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(totalStop);
    cudaEventSynchronize(totalStop);

    float totalCudaTime = 0.0f;
    cudaEventElapsedTime(&totalCudaTime, totalStart, totalStop);

    // Verify CPU and GPU results
    int correct = 1;

    for (int i = 0; i < N; i++)
    {
        if (h_output_cpu[i] != h_output_gpu[i])
        {
            correct = 0;
            break;
        }
    }

    printf("\n========================================\n");
    printf("GPU DATASET PROCESSING USING CUDA\n");
    printf("========================================\n");
    printf("Dataset Size = %d elements\n", N);
    printf("Block Size = %d threads\n", BLOCK_SIZE);
    printf("Grid Size = %d blocks\n", gridSize);

    printf("\nCPU Execution Time = %.6f ms\n", cpuTime);
    printf("CUDA Kernel Time = %.6f ms\n", kernelTime);
    printf("Total CUDA Time = %.6f ms\n", totalCudaTime);

    printf("Verification = %s\n", correct ? "PASSED" : "FAILED");

    printf("\nSample Results:\n");
    for (int i = 0; i < 5; i++)
    {
        printf("Input[%d] = %.2f  Output[%d] = %.2f\n",
               i, h_input[i], i, h_output_gpu[i]);
    }

    if (totalCudaTime > 0)
    {
        double speedup = cpuTime / totalCudaTime;
        printf("\nSpeedup = %.2fx\n", speedup);
    }

    printf("========================================\n");

    cudaFree(d_input);
    cudaFree(d_output);

    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);

    cudaEventDestroy(totalStart);
    cudaEventDestroy(totalStop);
    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelStop);

    return 0;
}
