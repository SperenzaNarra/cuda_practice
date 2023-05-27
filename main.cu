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
    *d_camera = new camera(2.0, 1.0);
}

__global__ void render_free(camera** d_camera)
{
    delete *d_camera;
}

int main()
{
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(IMAGE_WIDTH/BLOCK_SIZE+1);

    // buffer
    int pixel_size = sizeof(vec3double) * IMAGE_WIDTH;
    vec3double  h_pixels[IMAGE_WIDTH];
    vec3double *d_pixels;
    checkCudaErrors(cudaMallocManaged((void**)&d_pixels, pixel_size));

    // camera
    camera** d_camera;
    checkCudaErrors(cudaMallocManaged((void**)&d_camera, sizeof(camera*)));

    // spheres
    std::vector<sphere> h_spheres;
    // h_spheres.push_back(sphere(vec3double(0, -100.5, -1), 100, vec3double(0.5)));
    // h_spheres.push_back(sphere(vec3double(0, 0, -1), 0.5, vec3double(0.5)));

    h_spheres.push_back(sphere(vec3double(0, -100.5, -1), 100, vec3double(0.8, 0.8, 0)));
    h_spheres.push_back(sphere(vec3double(0, 0, -1), 0.5, vec3double(0.7, 0.3, 0.3)));
    h_spheres.push_back(sphere(vec3double(-1, 0, -1), 0.5, vec3double(0.8), MATERIAL_METAL));
    h_spheres.push_back(sphere(vec3double(1, 0, -1), 0.5, vec3double(0.8, 0.6, 0.2), MATERIAL_METAL));

    sphere* d_spheres;
    if (h_spheres.size())
    {
        checkCudaErrors(cudaMallocManaged((void**)&d_spheres, sizeof(sphere) * h_spheres.size()));
        checkCudaErrors(cudaMemcpy(d_spheres, &h_spheres[0], sizeof(sphere) * h_spheres.size(), cudaMemcpyHostToDevice));
    }

    // init
    render_init<<<1, 1>>>(d_camera);

    // core
    std::cout << "P3\n" << IMAGE_WIDTH << ' ' << IMAGE_HEIGHT << "\n255\n";
    for (int row = IMAGE_HEIGHT - 1; row >= 0; row--)
    {
        std::cerr << "\rScanlines remaining: " << row << ' ' << std::flush;
        render<<<dimGrid, dimBlock>>>(d_pixels, row, d_camera, d_spheres, h_spheres.size());

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaMemcpy(h_pixels, d_pixels, pixel_size, cudaMemcpyDeviceToHost));
        write_pixels(std::cout, h_pixels);
    }
    std::cerr << "\nDone.\n";

    render_free<<<1, 1>>>(d_camera);
    checkCudaErrors(cudaFree(d_pixels));
    checkCudaErrors(cudaFree(d_camera));
    if (h_spheres.size())
        checkCudaErrors(cudaFree(d_spheres));
}