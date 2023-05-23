#include "shape.cuh"

shape new_sphere(vec3double center, double radius, vec3double color, bool is_metal)
{
    return (shape)
    {
        .type = SHAPE_TYPE_SPHERE,
        .is_metal = is_metal,
        .color = color,
        .center = center,
        .radius = radius,
    };
}
__device__ double get_dist_from_sphere(
    shape sphere, 
    vec3double &origin, 
    vec3double &direction, 
    vec3double &pos, 
    vec3double &normal, 
    double &min_t, double &max_t)
{
    vec3double oc = origin - sphere.center;
    double a = direction.length2();
    double half_b = dot(oc, direction);
    double c = oc.length2() - sphere.radius * sphere.radius;

    double discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0) return -1.0;
    discriminant = sqrt(discriminant);

    double t1 = (-half_b - discriminant) / a;
    double t2 = (-half_b + discriminant) / a;
    bool has_t1 = min_t <= t1 && t1 <= max_t;
    bool has_t2 = min_t <= t2 && t2 <= max_t;

    // get distance
    double t;
    if (!has_t1 && !has_t2) 
        return -1.0;
    else if (!has_t1) 
        t = t2;
    else if (!has_t2)
        t = t1;
    else 
        t = t1 < t2 ? t1 : t2;

    // init position and normal
    pos = origin + t * direction;
    normal = (pos - sphere.center) / sphere.radius;

    // recorrect normal
    bool is_outside = dot(direction, normal) < 0;
    normal = is_outside ? normal : - normal;

    return t;
}

__device__ double shape::get_distance(
    vec3double &origin, 
    vec3double &direction, 
    vec3double &pos, 
    vec3double &normal, 
    double &min_t, double &max_t)
{
    switch (type)
    {
    case SHAPE_TYPE_SPHERE:
        return get_dist_from_sphere(*this, origin, direction, pos, normal, min_t, max_t);
    default:
        return -1.0;
    }
}
