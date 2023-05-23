#include "common.cuh"

#include <iostream>
#include <curand_kernel.h>

#include "vec3double.cuh"

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) 
{
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ vec3double random_in_unit_sphere(curandState &rand_state)
{
    vec3double vec;
    do 
    {
        vec = vec3double(curand_uniform_double(&rand_state), curand_uniform_double(&rand_state), curand_uniform_double(&rand_state));
        vec = 2.0 * vec - vec3double(1);
    }while (vec.length2() >= 1);
    return vec;
}

__device__ bool near_zero(vec3double &vec)
{
    if (vec.x > RAY_T_MIN || vec.x < -RAY_T_MIN)
        return false;
    if (vec.y > RAY_T_MIN || vec.y < -RAY_T_MIN)
        return false;
    if (vec.z > RAY_T_MIN || vec.z < -RAY_T_MIN)
        return false;

    return true;
}

__device__ bool get_vectors_from_shapes(
    vec3double &origin, 
    vec3double &direction, 
    vec3double &pos, 
    vec3double &normal,
    shape** target,
    shape *shapes, 
    size_t shapes_size)
{
    double t = RAY_T_MAX;
    double t_min = RAY_T_MIN;
    *target = NULL;
    for (int i = 0; i < shapes_size; i++)
    {
        double res = shapes[i].get_distance(origin, direction, pos, normal, t_min, t);
        if (res != -1.0 && res < t)
        {
            t = res;
            *target = &shapes[i];
        }
    }
    return !!*target;
}

__device__ vec3double get_default_color(vec3double &direction)
{
    double t = 0.5 * (direction.y + 1.0);
    return (1.0 - t) * vec3double(1) + t * vec3double(0.5, 0.7, 1.0);
}

__device__ bool get_reflected_vector(vec3double &direction, vec3double &normal, vec3double &reflected)
{
    reflected = direction - 2 * dot(direction, normal) * normal;
    return dot(reflected, normal) > 0;
}

__device__ vec3double get_color(vec3double &origin, vec3double &direction, shape *shapes, size_t shapes_size, curandState &rand_state)
{
    vec3double attenuation = vec3double(1);
    vec3double res, normal, rand_vec;
    shape* target;
    // for (int depth = 0; depth < RAY_TRACING_DEPTH; depth++)
    while (!near_zero(attenuation))
    {
        if (!get_vectors_from_shapes(origin, direction, res, normal, &target, shapes, shapes_size))
            return attenuation * get_default_color(direction);
        attenuation *= target->color;
        rand_vec = random_in_unit_sphere(rand_state);

        switch (target->material)
        {
        case SHAPE_MATERIAL_LAMBERTIAN:
            origin = res;
            direction = normal;
            if (!near_zero(rand_vec))
                direction += dot(rand_vec, normal) > 0 ? rand_vec : -rand_vec;
            break;
        case SHAPE_MATERIAL_METAL:
            origin = res;
            get_reflected_vector(direction, normal, res);
            direction = res;
            if (!near_zero(rand_vec))
                direction += dot(rand_vec, normal) > 0 ? (target->fuzz * rand_vec) : -(target->fuzz * rand_vec);
            break;
        default:
            return vec3double(0);
        }

    }
    return vec3double(0);
}

__global__ void render(vec3double* pixels, int row, camera** d_camera, shape* shapes, size_t shapes_size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= IMAGE_WIDTH) return;

    // set random seed
    curandState rand_state;
    curand_init((unsigned long long)clock64() + i, 0, 0, &rand_state);

    // render processing
    vec3double origin, direction;
    vec3double *target = &pixels[i];
    double u, v;
    for (int s = 0; s < SAMPLE_NUM; s++)
    {
        u = ((double)i + curand_uniform_double(&rand_state)) / (IMAGE_WIDTH - 1);
        v = ((double)row + curand_uniform_double(&rand_state)) / (IMAGE_HEIGHT - 1);
        (*d_camera)->get_ray(u, v, origin, direction);
        *target += get_color(origin, direction, shapes, shapes_size, rand_state);
    }
    
    *target /= SAMPLE_NUM;
    target->r = sqrt(target->r);
    target->g = sqrt(target->g);
    target->b = sqrt(target->b);
}