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

enum
{
    STATE_DEFAULT,
    STATE_RIGID,
    STATE_REFLECTION,
    STATE_REFRACTION,
    STATE_TERMINATE
};

__device__ vec3double get_color(vec3double &origin, vec3double &direction, shape *shapes, size_t shapes_size, curandState &rand_state)
{
    // since recursive does not worked on cuda, I have to build a dp
    vec3double color[RAY_TRACING_DEPTH+1];
    vec3double P [RAY_TRACING_DEPTH+1]; // position
    vec3double N [RAY_TRACING_DEPTH+1]; // normal
    vec3double D [RAY_TRACING_DEPTH+1]; // direction
    shape* target[RAY_TRACING_DEPTH+1];
    vec3double rand_unit_sphere;
    size_t states[RAY_TRACING_DEPTH+1];

    D[0] = direction;
    P[0] = origin;

    if (!get_vectors_from_shapes(origin, direction, P[1], N[1], &target[1], shapes, shapes_size)) 
        return get_default_color(direction);

    for (int i = 0; i < RAY_TRACING_DEPTH; i++)
    {
        color[i] = vec3double(0);
        states[i] = STATE_DEFAULT;
    }

    size_t depth = 1;
    while (true)
    {
        states[depth] += 1;
        if (states[0] != STATE_DEFAULT) break;
        switch (states[depth])
        {
        case STATE_RIGID:
            // get rigid direction
            color[depth] = vec3double(1);
            D[depth] = N[depth];
            rand_unit_sphere = random_in_unit_sphere(rand_state);
            if (!near_zero(rand_unit_sphere))
                D[depth] += dot(rand_unit_sphere, N[depth]) > 0 ? random_in_unit_sphere(rand_state) :  - random_in_unit_sphere(rand_state);
            
            // shoot the array and get the shape, shape color, hit position, and the normal vector from the position
            if (!get_vectors_from_shapes(P[depth], D[depth], P[depth+1], N[depth+1], &target[depth+1], shapes, shapes_size))
                color[depth] = get_default_color(D[depth]);
            else if (depth < RAY_TRACING_DEPTH)
                depth += 1;
            break;
        case STATE_REFLECTION:
            if (target[depth]->is_metal && get_reflected_vector(D[depth-1], N[depth], D[depth]))
            {
                if (!get_vectors_from_shapes(P[depth], D[depth], P[depth+1], N[depth+1], &target[depth+1], shapes, shapes_size))
                    color[depth] = get_default_color(D[depth]);
                else if (depth < RAY_TRACING_DEPTH)
                    depth += 1;
            }
            break;
        case STATE_REFRACTION:
        case STATE_TERMINATE:
        default:
            color[depth-1] += color[depth] * target[depth]->color;
            if (color[depth-1].r > 1) color[depth-1].r = 1.0;
            if (color[depth-1].g > 1) color[depth-1].g = 1.0;
            if (color[depth-1].b > 1) color[depth-1].b = 1.0;
            states[depth] = STATE_DEFAULT;
            depth -= 1;
            break;
        }
    }
    return color[0];

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