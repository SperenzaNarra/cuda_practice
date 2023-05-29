#include "common.cuh"
#include <math.h>

void write_pixels(std::ostream &out, vec3double *pixels)
{
    for (int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; i++)
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

__device__ vec3double get_default_color(vec3double &direction)
{
    float t = 0.5 * (direction.y + 1.0);
    return (1.0 - t) * vec3double(1.0) + t * vec3double(0.5, 0.7, 1.0);
}

__device__ vec3double get_rand_hemisphere(vec3double &normal, curandState &rand_state)
{
    double angle = 2 * M_PI * curand_uniform_double(&rand_state);
    double rad_2 = curand_uniform_double(&rand_state);
    double rad = sqrt(rad_2);
    vec3double u = (cross(abs(normal.x) > .1 ? vec3double(0, 1, 0) : vec3double(1, 0, 0), normal)).normalized(),
               v = cross(normal, u);
    // sample unit hemisphere
    vec3double d = (u * cos(angle) * rad + v * sin(angle) * rad + normal * sqrt(1 - rad_2)).normalized();
    return d;
}

__device__ vec3double get_color(ray &in_ray, sphere* spheres, int sphere_size, curandState &rand_state)
{
    vec3double attenuation = vec3double(1);
    int depth = 0;
    while (true)
    // while (depth < 10)
    {
        sphere* target_sphere = NULL;
        double t = RAY_T_MAX;
        ray normal;
        bool into;

        for (int i = 0; i < sphere_size; i++)
        {
            if (spheres[i].hit(in_ray, normal, into, t))
            {
                target_sphere = &spheres[i];
            }
        }

        if (!target_sphere) return attenuation * get_default_color(in_ray.direction);

        double p = max(attenuation.r, max(attenuation.g, attenuation.b));
        if (++depth > 5)
            if (curand_uniform_double(&rand_state) < p) //R.R.
                attenuation = attenuation * (1 / p);
            else {
                break;
            }

        // update
        attenuation *= target_sphere->color;
        in_ray.origin = normal.origin;

        switch (target_sphere->type)
        {
        case MATERIAL_LAMBERTIAN:
            in_ray.direction = get_rand_hemisphere(normal.direction, rand_state);
            break;
        case MATERIAL_METAL:
            in_ray.direction = in_ray.direction - 2 * dot(in_ray.direction, normal.direction) * normal.direction;
            in_ray.direction += get_rand_hemisphere(in_ray.direction, rand_state) * target_sphere->fuzz;
            break;
        case MATERIAL_DIELECTRIC:{
            double n_ratio = into ? 1.0 / target_sphere->ir : target_sphere->ir;
            double d_dot_n = dot(in_ray.direction, normal.direction),
                   cos2t = 1 - square(n_ratio) * (1 - square(d_dot_n));
            if (cos2t < 0) {   // Total internal reflection
                in_ray.direction = in_ray.direction - 2 * dot(in_ray.direction, normal.direction) * normal.direction;
                break;
            }

            vec3double tdir = (in_ray.direction * n_ratio - normal.direction * (d_dot_n * n_ratio + sqrt(cos2t)));

            double refl_norm = square(target_sphere->ir - 1.0) / square(target_sphere->ir + 1.0),
                   c = 1 - (into ? -d_dot_n : -dot(tdir, normal.direction));
            double refl_fresnel = refl_norm + (1 - refl_norm) * c * c * c * c * c,
                   trans_fresnel = 1 - refl_fresnel,
                   prob_refl = .25 + .5 * refl_fresnel;

            if (curand_uniform_double(&rand_state) < prob_refl) { // Russian roulette
                attenuation = attenuation * (refl_fresnel / prob_refl);
                in_ray.direction = in_ray.direction - 2 * dot(in_ray.direction, normal.direction) * normal.direction;
            } else {
                attenuation = attenuation * (trans_fresnel / (1 - prob_refl));
                in_ray.direction = tdir;
            }
            break;
        }

        default:
            return vec3double(0);
        }
        in_ray.direction = in_ray.direction.normalized();
    }
    
    // out of depth
    return vec3double(0);
}

__global__ void render(vec3double *pixels, camera** camera, sphere* spheres, int sphere_size, int* tasks_done)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if (col >= IMAGE_WIDTH || row >= IMAGE_HEIGHT) return;

    int index = col + row * IMAGE_WIDTH;
    row = IMAGE_HEIGHT - row - 1;

    curandState rand_state;
    curand_init((unsigned long long)clock64() + index, 0, 0, &rand_state);

    for (int s = 0; s < SAMPLES_PER_PIXEL; s++)
    {
        double u = ((double) col + curand_uniform_double(&rand_state)) / (IMAGE_WIDTH - 1);
        double v = ((double) row + curand_uniform_double(&rand_state)) / (IMAGE_HEIGHT - 1);
        ray in_ray = (*camera)->get_ray(u, v, rand_state);
        pixels[index] += get_color(in_ray, spheres, sphere_size, rand_state);
    }
    pixels[index] /= SAMPLES_PER_PIXEL;
    atomicAdd(tasks_done, 1);
    if (*tasks_done%(BLOCK_SIZE*BLOCK_SIZE) == 0)
        printf("\r%.6f%%", (double)*tasks_done/(IMAGE_WIDTH * IMAGE_HEIGHT) * 100);
}