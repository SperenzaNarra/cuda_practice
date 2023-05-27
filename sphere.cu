#include "common.cuh"


sphere::sphere(const vec3double &center, double radius, const vec3double &color, int type):
    center(center), radius(radius), color(color), type(type) {}

__device__ bool sphere::hit(ray &in_ray, ray &normal, double &t_max)
{
    vec3double oc = in_ray.origin - center;
    double half_b   = dot(oc, in_ray.direction);
    double c        = oc.length2() - radius * radius;
    double discriminant = half_b*half_b - c;

    if (discriminant < 0)
        return false;

    discriminant = sqrt(discriminant);

    double t = -half_b - discriminant;
    if (t < RAY_T_MIN || t > t_max)
    {
        t = -half_b + discriminant;
        if (t < RAY_T_MIN || t > t_max)
            return false;
    }

    t_max = t;
    normal.origin = in_ray.origin + t * in_ray.direction;
    normal.direction = (normal.origin - center) / radius;
    normal.direction *= std::signbit(dot(in_ray.direction, normal.direction)) * 2.0 - 1.0;
    return true;
}