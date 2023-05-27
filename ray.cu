#include "common.cuh"

__device__ ray::ray() {}
__device__ ray::ray(const ray &ray):origin(ray.origin), direction(ray.direction){}
__device__ ray::~ray() {}

__device__ ray::ray(vec3double origin, vec3double direction):
    origin(origin), direction(direction.normalized()){}
