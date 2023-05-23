#ifndef COMMON_CUH
#define COMMON_CUH

#include <curand_kernel.h>

#include "vec3double.cuh"
#include "camera.cuh"
#include "shape.cuh"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

constexpr int IMAGE_WIDTH  = 2048;
constexpr int IMAGE_HEIGHT = 1080;
constexpr int BLOCK_SIZE = 512;
constexpr int SAMPLE_NUM = 100;
constexpr int RAY_TRACING_DEPTH = 10;

constexpr double RAY_T_MIN = 0.0001;
constexpr double RAY_T_MAX = 1.0e30;

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);

// core
__global__ void render(vec3double* pixels, int row, camera** d_camera, shape* shapes, size_t shapes_size);

// random



#endif // COMMON_CUH