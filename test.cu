#include <iostream>
#include "common.cuh"

// Device code
__global__ void vec_init_task(double* dst, int max_x, int max_y)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= max_x || y >= max_y) return;
    int index = x + y * max_x;
    dst[index] = (double)index;
}

void vec_init(double* vec, int width, int height)
{
    size_t size = width * height * sizeof(double);

    // capture start time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // allocate
    double *c_vec;
    cudaMalloc((void**)&c_vec, size);

    // kernel invocation code
    int tx = 32;
    int ty = 32;
    dim3 dimBlock(tx, ty);
    dim3 dimGrid(width/tx+1, height/ty+1);
    vec_init_task<<<dimGrid, dimBlock>>>(c_vec, width, height);

    // transfer
    cudaMemcpy(vec, c_vec, size, cudaMemcpyDeviceToHost);

    // stop time and display
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float elapse;
    cudaEventElapsedTime(&elapse, start, stop);
    printf( "Time to generate:  %3.1f ms\n", elapse);

    // free
    cudaFree(c_vec);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}

// Host code
int main()
{
    int width  = 64;
    int height = 64;
    double *vec = (double*)malloc(width * height * sizeof(double));
    vec_init(vec, width, height);

    for (int i  = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int index = i * width + j;
            std::cout << vec[index] << " ";
        }
        std::cout << std::endl;
    }
}