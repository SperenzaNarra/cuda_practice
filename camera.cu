#include "camera.cuh"

__device__ camera::camera(double vfov, double aspect_ratio)
{
    double theta = vfov * M_PI / 180.0;
    double viewport_height = 2.0 * tan(theta / 2.0);
    double viewport_width = viewport_height * aspect_ratio;
    double focal_length = 1.0;

    origin = vec3double(0);
    horizontal = vec3double(viewport_width, 0, 0);
    vertical = vec3double(0, viewport_height, 0);
    lower_left_corner = origin - horizontal / 2 - vertical / 2 - vec3double(0, 0, focal_length);
}

__device__ camera::~camera(){}

__device__ void camera::get_ray(double u, double v, vec3double &origin, vec3double &direction)
{
    origin = this->origin;
    direction = lower_left_corner + u * horizontal + v * vertical - this->origin;
}