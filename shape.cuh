#ifndef SHAPE_CUH
#define SHAPE_CUH

#include "vec3double.cuh"

enum
{
    SHAPE_TYPE_PLAN,
    SHAPE_TYPE_SPHERE,
    SHAPE_TYPE_TRIANGLE
};

enum
{
    SHAPE_MATERIAL_LAMBERTIAN,
    SHAPE_MATERIAL_METAL,
    SHAPE_MATERIAL_DIELECTRIC,
};

struct shape
{
    // common variables
    int type;
    vec3double color;
    // unique variables
    vec3double center;
    double radius;

    // for material
    int material;
    double fuzz;


    __device__ double get_distance(vec3double &origin, vec3double &direction, vec3double &pos, vec3double &normal, double &min_t, double &max_t);
    shape& set_as_metal(double fuzz = 0.0);
};

shape new_sphere(vec3double center, double radius, vec3double color);

#endif // SHAPE_CUH