#include "common.cuh"

// __device__ camera::camera(double viewport_height, double focal_length)
// {
//     double aspect_ratio = (double) IMAGE_WIDTH / IMAGE_HEIGHT;
//     viewport_height = 2.0;
//     double viewport_width = aspect_ratio * viewport_height;
//     focal_length = 1.0;

//     origin = vec3double(0);
//     horizontal = vec3double(viewport_width, 0, 0);
//     vertical = vec3double(0, viewport_height, 0);
//     lower_left_corner = origin - horizontal / 2 - vertical / 2 - vec3double(0, 0, focal_length);
// }

__device__ camera::camera(
    vec3double lookfrom, 
    vec3double lookat, 
    vec3double vup, 
    double vfov, 
    double aspect_ratio,
    double aperture,
    double focus_dist
){

    double theta = vfov * M_PI / 180.0;
    double h = tan(theta / 2.0);
    double viewport_height = 2.0 * h;
    double viewport_width = viewport_height * aspect_ratio;

    w = (lookfrom - lookat).normalized();
    u = cross(vup, w);
    v = cross(w, u);

    origin = lookfrom;
    horizontal = focus_dist * viewport_width * u;
    vertical = focus_dist * viewport_height * v;
    lower_left_corner = origin - horizontal / 2 - vertical / 2 - focus_dist * w;

    lens_radius = aperture / 2;
}

// __device__ ray camera::get_ray(double u, double v)
// {
//     return ray(origin, (lower_left_corner + u * horizontal + v * vertical - origin).normalized());
// }

__device__ ray camera::get_ray(double u, double v, curandState &rand_state)
{
    vec3double rd = lens_radius * 
        vec3double(curand_uniform_double(&rand_state),curand_uniform_double(&rand_state),curand_uniform_double(&rand_state))
        .normalized();
    vec3double offset = u * rd.x + v * rd.y;
    return ray(origin+offset, (lower_left_corner + u * horizontal + v * vertical - origin - offset).normalized());
}