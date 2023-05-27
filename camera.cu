#include "common.cuh"

__device__ camera::camera(double viewport_height, double focal_length)
{
    double aspect_ratio = (double) IMAGE_WIDTH / IMAGE_HEIGHT;
    viewport_height = 2.0;
    double viewport_width = aspect_ratio * viewport_height;
    focal_length = 1.0;

    origin = vec3double(0);
    horizontal = vec3double(viewport_width, 0, 0);
    vertical = vec3double(0, viewport_height, 0);
    lower_left_corner = origin - horizontal / 2 - vertical / 2 - vec3double(0, 0, focal_length);
}

__device__ ray camera::get_ray(double u, double v)
{
    return ray(origin, (lower_left_corner + u * horizontal + v * vertical - origin).normalized());
}