#ifndef COMMON_CUH
#define COMMON_CUH

#include <iostream>
#include <curand_kernel.h>
#include "vec3double.cuh"

#define IMAGE_WIDTH  1080
#define IMAGE_HEIGHT 720
#define SAMPLES_PER_PIXEL 100
#define DEPTH 50
#define RAY_T_MIN 0.0001
#define RAY_T_MAX 1.0e30

// for cuda
#define BLOCK_SIZE   512

struct ray
{
    vec3double origin;
    vec3double direction;
    
    __device__ ray();
    __device__ ray(const ray &ray);
    __device__ ~ray();
    __device__ ray(vec3double origin, vec3double direction);
};

class camera
{
private:
    vec3double origin, horizontal, vertical, lower_left_corner;
public:
    __device__ camera(double viewport_height, double focal_length);
    __device__ ray get_ray(double u, double v);
};

enum
{
    MATERIAL_LAMBERTIAN,
    MATERIAL_METAL,
    MATERIAL_DIELECTRIC,
};

class sphere
{
public:
    vec3double color;
    int type;
private:
    vec3double center;
    double radius;
public:
    sphere(const vec3double &center, double radius, const vec3double &color, int type = MATERIAL_LAMBERTIAN);
    __device__ bool hit(ray &in_ray, ray &out_ray, double &t_max);
};

void write_pixels(std::ostream &out, vec3double *pixels);
__global__ void render(vec3double *pixels, int row, camera** camera, sphere* spheres, int sphere_size);

#endif