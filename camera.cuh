#ifndef CAMERA_CUH
#define CAMERA_CUH

#include <curand_kernel.h>

#include "vec3double.cuh"


class camera
{
private:
    vec3double origin, horizontal, vertical, lower_left_corner;
public:
    __device__ camera(double vfov, double aspect_ratio);
    __device__ ~camera();

    __device__ void get_ray(double u, double v, vec3double &origin, vec3double &direction);
};

#endif // CAMERA_CUH