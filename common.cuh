#ifndef COMMON_CUH
#define COMMON_CUH

#include <iostream>
#include <curand_kernel.h>
#include "vec3double.cuh"

#define IMAGE_WIDTH  3840
#define IMAGE_HEIGHT 2160
// #define IMAGE_WIDTH  1080
// #define IMAGE_HEIGHT 720
#define SAMPLES_PER_PIXEL 50
#define RAY_T_MIN 0.0001
#define RAY_T_MAX 1.0e30

// for cuda
#define BLOCK_SIZE 32


template <class T>
__device__ inline T square(T x) {
    return x * x;
}


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
    vec3double u, v, w;
    double lens_radius;
public:
    // __device__ camera(double viewport_height, double focal_length);
    __device__ camera(vec3double lookfrom, vec3double lookat, vec3double vup, double vfov, double aspect_ratio, double aperture, double focus_dist);
    // __device__ ray get_ray(double u, double v);
    __device__ ray get_ray(double u, double v, curandState &rand_state);
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
    union {
        double fuzz;
        double ir; // Index of Refraction
    };
private:
    vec3double center;
    double radius;
public:
    sphere(const vec3double &center, double radius, const vec3double &color);
    sphere& as_metal(double fuzz);
    sphere& as_dielectric(double ir);
    __device__ bool hit(ray &in_ray, ray &out_ray, bool &into, double &t_max);
};

void write_pixels(std::ostream &out, vec3double *pixels);
__global__ void render(vec3double *pixels, camera** camera, sphere* spheres, int sphere_size, int* tasks_done);

#endif