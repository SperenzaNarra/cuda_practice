#include "common.cuh"


sphere::sphere(const vec3double &center, double radius, const vec3double &color):
    center(center), radius(radius), color(color), type(MATERIAL_LAMBERTIAN) {}

sphere& sphere::as_metal(double fuzz)
{
    type = MATERIAL_METAL;
    this->fuzz = max(0.0, min(1.0, fuzz));
    return *this;
}

sphere& sphere::as_dielectric(double ir)
{
    type = MATERIAL_DIELECTRIC;
    this->ir = max(0.0, ir);
    return *this;
    
}

__device__ bool sphere::hit(ray &in_ray, ray &normal, bool &into, double &t_max)
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
    into = std::signbit(dot(in_ray.direction, normal.direction));
    normal.direction *= into * 2.0 - 1.0;
    return true;
}