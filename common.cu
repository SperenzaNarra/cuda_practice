#include "common.cuh"

void write_pixels(std::ostream &out, vec3double *pixels)
{
    for (int i = 0; i < IMAGE_WIDTH; i++)
    {
        vec3double *pixel = &pixels[i];
        int r = static_cast<int>(255.999 * std::sqrt(pixel->r));
        int g = static_cast<int>(255.999 * std::sqrt(pixel->g));
        int b = static_cast<int>(255.999 * std::sqrt(pixel->b));
        out << r << ' ' << g << ' ' << b << std::endl;
    }
}

__device__ double random_double(double min, double max, curandState &rand_state)
{
    return min + (max-min) * curand_uniform_double(&rand_state);
}

__device__ vec3double random_double_vector(double min, double max, curandState &rand_state)
{
    return vec3double(random_double(min, max, rand_state), random_double(min, max, rand_state), random_double(min, max, rand_state));
}

__device__ vec3double random_in_unit_sphere(curandState &rand_state)
{
    vec3double vec = random_double_vector(-1.0, 1.0, rand_state);
    while (vec.length2() >= 0.5)
        vec = random_double_vector(-1.0, 1.0, rand_state);
    return vec;
}

__device__ vec3double get_default_color(vec3double &direction)
{
    float t = 0.5 * (direction.y + 1.0);
    return (1.0 - t) * vec3double(1.0) + t * vec3double(0.5, 0.7, 1.0);
}


__device__ vec3double get_color(ray &in_ray, sphere* spheres, int sphere_size, curandState &rand_state)
{
    double t = RAY_T_MAX;
    sphere* target_sphere;
    ray normal;
    vec3double attenuation = vec3double(1);
    vec3double rand_vec;

    for (int depth = 0; depth < DEPTH; depth++)
    {
        target_sphere = NULL;
        for (int i = 0; i < sphere_size; i++)
        {
            if (spheres[i].hit(in_ray, normal, t))
            {
                target_sphere = &spheres[i];
            }
        }

        if (!target_sphere) return attenuation * get_default_color(in_ray.direction);

        attenuation *= target_sphere->color;
        in_ray.origin = normal.origin;

        switch (target_sphere->type)
        {
        case MATERIAL_LAMBERTIAN:
            rand_vec = random_in_unit_sphere(rand_state);
            rand_vec *= std::signbit(dot(rand_vec, normal.direction)) * 2 - 1;
            in_ray.direction = normal.direction + rand_vec;
            break;
        case MATERIAL_METAL:
            in_ray.direction = in_ray.direction - 2 * dot(in_ray.direction, normal.direction) * normal.direction;
            break;
        default:
            return vec3double(0);
        }
        in_ray.direction = in_ray.direction.normalized();
    }
    
    // out of depth
    return vec3double(0);
}

__global__ void render(vec3double *pixels, int row, camera** camera, sphere* spheres, int sphere_size)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (col >= IMAGE_WIDTH) return;

    curandState rand_state;
    curand_init((unsigned long long)clock64() + col, 0, 0, &rand_state);

    for (int s = 0; s < SAMPLES_PER_PIXEL; s++)
    {
        double u = ((double) col + curand_uniform_double(&rand_state)) / (IMAGE_WIDTH - 1);
        double v = ((double) row + curand_uniform_double(&rand_state)) / (IMAGE_HEIGHT - 1);
        ray in_ray = (*camera)->get_ray(u, v);
        pixels[col] += get_color(in_ray, spheres, sphere_size, rand_state);
    }
    pixels[col] /= SAMPLES_PER_PIXEL;
}