// cuda_practice.cpp: 定义应用程序的入口点。
//

#include "common.cuh"

#include <iostream>
#include <fstream>
#include <curand_kernel.h>
#include <vector>

#include "vec3double.cuh"
#include "camera.cuh"
#include "shape.cuh"

__global__ void create_prefabs(camera** d_camera)
{
    *d_camera = new camera(90.0, (double)IMAGE_WIDTH/IMAGE_HEIGHT);
}

__global__ void delete_prefabs(camera** d_camera)
{
    delete *d_camera;
}

void render_handler(vec3double *h_pixels, int row, size_t image_size, vec3double **d_pixels, camera ***d_camera, shape** d_shapes, int d_shape_size)
{
    // kernel invocation code
    int tx = BLOCK_SIZE;
    dim3 dimBlock(tx);
    dim3 dimGrid(IMAGE_WIDTH/tx+1);

    render<<<dimGrid, dimBlock>>>(*d_pixels, row, *d_camera, *d_shapes, d_shape_size);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // download data from device
    checkCudaErrors(cudaMemcpy(h_pixels, *d_pixels, image_size, cudaMemcpyDeviceToHost));

}

int main()
{
    // preparation
    size_t pixel_num = IMAGE_WIDTH;
    size_t image_size = pixel_num * sizeof(vec3double);

    // buffers
    vec3double *h_pixels = (struct vec3double*)malloc(image_size);
    vec3double *d_pixels;
    checkCudaErrors(cudaMallocManaged((void**)&d_pixels, image_size));

    // camera
    camera **d_camera;
    checkCudaErrors(cudaMallocManaged((void**)&d_camera, sizeof(camera*)));
    create_prefabs<<<1, 1>>>(d_camera);

    // shapes
    std::vector<shape> h_shapes;
    h_shapes.push_back(new_sphere(vec3double(0, -100.5, -1), 100.0, vec3double(0.8, 0.8, 0)));
    h_shapes.push_back(new_sphere(vec3double(0, 0, -1), 0.5, vec3double(0.7, 0.3, 0.3)));
    h_shapes.push_back(new_sphere(vec3double(1, 0, -1), 0.5, vec3double(0.8, 0.6, 0.2)).set_as_metal(1.0));
    h_shapes.push_back(new_sphere(vec3double(-1, 0, -1), 0.5, vec3double(0.8, 0.8, 0.8)).set_as_metal(0.3));

    shape* d_shapes;
    checkCudaErrors(cudaMallocManaged((void**)&d_shapes, sizeof(shape) * h_shapes.size()));
    cudaMemcpy(d_shapes, &h_shapes[0], sizeof(shape) * h_shapes.size(), cudaMemcpyHostToDevice);


    //result
	std::cout << "P3\n" << IMAGE_WIDTH << ' ' << IMAGE_HEIGHT << "\n255\n";
	
	for (int j = IMAGE_HEIGHT-1; j >= 0; --j) 
    {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        render_handler(h_pixels, j, image_size, &d_pixels, &d_camera, &d_shapes, h_shapes.size());
        for (int i = 0; i < IMAGE_WIDTH; ++i) 
        {
            struct vec3double *pixel = &h_pixels[i];
            int ir = int(255.999 * pixel->r);
            int ig = int(255.999 * pixel->g);
            int ib = int(255.999 * pixel->b);

            std::cout << ir << ' ' << ig << ' ' << ib << std::endl;
        }
    }
    std::cerr << "\nDone.\n";
    
    // free device variables
    free(h_pixels);
    delete_prefabs<<<1, 1>>>(d_camera);
    checkCudaErrors(cudaFree(d_pixels));
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_shapes));

	return 0;
}
