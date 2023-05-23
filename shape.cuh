#ifndef SHAPE_CUH
#define SHAPE_CUH

#include "vec3double.cuh"

enum
{
    SHAPE_TYPE_PLAN,
    SHAPE_TYPE_SPHERE,
    SHAPE_TYPE_TRIANGLE
};

struct shape
{
    // common variables
    int type;
    bool is_metal;
    vec3double color;

    // unique variables
    vec3double center;
    double radius;

    __device__ double get_distance(vec3double &origin, vec3double &direction, vec3double &pos, vec3double &normal, double &min_t, double &max_t);
};

shape new_sphere(vec3double center, double radius, vec3double color, bool is_metal = false);

#endif // SHAPE_CUH