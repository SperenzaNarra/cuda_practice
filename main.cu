#include "common.cuh"
#include <vector>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
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

__global__ void render_init(camera** d_camera)
{
    // *d_camera = new camera(2.0, 1.0);
    vec3double  lookfrom(13, 2, 3), lookat(0, 0, 0), vup(0, 1, 0);
    double dist_to_focus = 10.0;
    double aperture = 0.1;
    *d_camera = new camera(lookfrom, lookat, vup, 20, (double)IMAGE_WIDTH/IMAGE_HEIGHT, aperture, dist_to_focus);
}

__global__ void render_free(camera** d_camera)
{
    delete *d_camera;
}


inline double random_double()
{
    return rand() / (RAND_MAX + 1.0);
}

inline double random_double(double min, double max)
{
    return min + (max - min) * random_double();
}

inline vec3double random_vector()
{
    return vec3double(random_double(), random_double(), random_double());
}

inline vec3double random_vector(double min, double max)
{
    return vec3double(random_double(min, max), random_double(min, max), random_double(min, max));
}


int main()
{
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(IMAGE_WIDTH/BLOCK_SIZE+1, IMAGE_HEIGHT/BLOCK_SIZE+1);

    // buffer
    int pixel_size = sizeof(vec3double) * IMAGE_WIDTH * IMAGE_HEIGHT;
    vec3double  *h_pixels = (vec3double*)malloc(pixel_size);
    vec3double *d_pixels;
    checkCudaErrors(cudaMallocManaged((void**)&d_pixels, pixel_size));

    // cameras
    camera** d_camera;
    checkCudaErrors(cudaMallocManaged((void**)&d_camera, sizeof(camera*)));

    // spheres
    std::vector<sphere> h_spheres;
    
    // ground
    // srand(time(NULL));
    h_spheres.push_back(sphere(vec3double(0, -1000, 0), 1000, vec3double(0.5)));
    for (int a = -11; a < 11; a++)
    for (int b = -11; b < 11; b++){
        double choose_mat = random_double();
        vec3double center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());
        if ((center - vec3double(4, 0.2, 0)).length() > 0.9){
            if (choose_mat < 0.8)   
                h_spheres.push_back(sphere(center, 0.2, random_vector()*random_vector()));
            else if (choose_mat < 0.95)
                h_spheres.push_back(sphere(center, 0.2, random_vector(0.5, 1)).as_metal(random_double(0, 0.5)));
            else
                h_spheres.push_back(sphere(center, 0.2, random_vector()*random_vector()).as_dielectric(1.5));
        }
    }
    h_spheres.push_back(sphere(vec3double(0, 1, 0), 1, vec3double(1)).as_dielectric(1.5));
    h_spheres.push_back(sphere(vec3double(-4, 1, 0), 1, vec3double(0.4, 0.2, 0.1)));
    h_spheres.push_back(sphere(vec3double(4, 1, 0), 1, vec3double(0.7, 0.6, 0.5)).as_metal(0.0));

    sphere* d_spheres;
    if (h_spheres.size()){
        checkCudaErrors(cudaMallocManaged((void**)&d_spheres, sizeof(sphere) * h_spheres.size()));
        checkCudaErrors(cudaMemcpy(d_spheres, &h_spheres[0], sizeof(sphere) * h_spheres.size(), cudaMemcpyHostToDevice));
    }

    // init
    render_init<<<1, 1>>>(d_camera);

    // core
    std::cout << "P3\n" << IMAGE_WIDTH << " " << IMAGE_HEIGHT << "\n255\n";
    render<<<dimGrid, dimBlock>>>(d_pixels, d_camera, d_spheres, h_spheres.size());

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(h_pixels, d_pixels, pixel_size, cudaMemcpyDeviceToHost));
    write_pixels(std::cout, h_pixels);

    render_free<<<1, 1>>>(d_camera);
    checkCudaErrors(cudaFree(d_pixels));
    checkCudaErrors(cudaFree(d_camera));
    if (h_spheres.size())
        checkCudaErrors(cudaFree(d_spheres));
}