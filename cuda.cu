#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "include/util.h"

// Error checking macro for CUDA calls
#define CHECK_CUDA_ERROR(call)                                                                                 \
    do                                                                                                         \
    {                                                                                                          \
        cudaError_t err = call;                                                                                \
        if (err != cudaSuccess)                                                                                \
        {                                                                                                      \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                                \
        }                                                                                                      \
    } while (0)

// Create Gaussian kernel and return it as a device pointer
__host__ float *create_gaussian_kernel(int radius, float sigma, float **d_kernel)
{
    int kernel_size = 2 * radius + 1;
    float *kernel = (float *)malloc(kernel_size * sizeof(float));
    float sum = 0.0f;

    // Fill kernel with Gaussian values
    for (int i = 0; i < kernel_size; i++)
    {
        int x = i - radius;
        kernel[i] = expf(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }

    // Normalize kernel
    for (int i = 0; i < kernel_size; i++)
    {
        kernel[i] /= sum;
    }

    // Allocate and copy kernel to device
    CHECK_CUDA_ERROR(cudaMalloc(d_kernel, kernel_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(*d_kernel, kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    return kernel;
}

// CUDA kernel for horizontal blur pass
__global__ void horizontal_blur_kernel(
    unsigned char *d_input,
    unsigned char *d_output,
    int width,
    int height,
    int radius,
    float *d_kernel)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        float r = 0.0f, g = 0.0f, b = 0.0f, a = 0.0f;
        const int kernel_size = 2 * radius + 1;

        for (int i = -radius; i <= radius; i++)
        {
            int ix = x + i;
            // Handle boundary conditions
            if (ix < 0)
                ix = 0;
            if (ix >= width)
                ix = width - 1;

            const int input_idx = (y * width + ix) * 4;
            const float weight = d_kernel[i + radius];

            r += d_input[input_idx + 0] * weight;
            g += d_input[input_idx + 1] * weight;
            b += d_input[input_idx + 2] * weight;
            a += d_input[input_idx + 3] * weight;
        }

        const int output_idx = (y * width + x) * 4;
        d_output[output_idx + 0] = (unsigned char)r;
        d_output[output_idx + 1] = (unsigned char)g;
        d_output[output_idx + 2] = (unsigned char)b;
        d_output[output_idx + 3] = (unsigned char)a;
    }
}

// CUDA kernel for vertical blur pass
__global__ void vertical_blur_kernel(
    unsigned char *d_input,
    unsigned char *d_output,
    int width,
    int height,
    int radius,
    float *d_kernel)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        float r = 0.0f, g = 0.0f, b = 0.0f, a = 0.0f;
        const int kernel_size = 2 * radius + 1;

        for (int i = -radius; i <= radius; i++)
        {
            int iy = y + i;
            // Handle boundary conditions
            if (iy < 0)
                iy = 0;
            if (iy >= height)
                iy = height - 1;

            const int input_idx = (iy * width + x) * 4;
            const float weight = d_kernel[i + radius];

            r += d_input[input_idx + 0] * weight;
            g += d_input[input_idx + 1] * weight;
            b += d_input[input_idx + 2] * weight;
            a += d_input[input_idx + 3] * weight;
        }

        const int output_idx = (y * width + x) * 4;
        d_output[output_idx + 0] = (unsigned char)r;
        d_output[output_idx + 1] = (unsigned char)g;
        d_output[output_idx + 2] = (unsigned char)b;
        d_output[output_idx + 3] = (unsigned char)a;
    }
}

// Host function to apply Gaussian blur using CUDA
void apply_gaussian_blur_cuda(png_bytep *row_pointers, int width, int height, int radius)
{
    const size_t image_size = width * height * 4 * sizeof(unsigned char);
    unsigned char *h_image = (unsigned char *)malloc(image_size);

    // Copy row_pointers to a contiguous memory block
    for (int y = 0; y < height; y++)
    {
        memcpy(&h_image[y * width * 4], row_pointers[y], width * 4);
    }

    // Calculate sigma based on radius
    float sigma = radius / 2.0f;

    // Create and copy Gaussian kernel to device
    float *d_kernel;
    float *h_kernel = create_gaussian_kernel(radius, sigma, &d_kernel);

    // Allocate device memory
    unsigned char *d_input, *d_output, *d_temp;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, image_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, image_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_temp, image_size));

    // Copy image data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_image, image_size, cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    // Apply horizontal blur
    horizontal_blur_kernel<<<gridDim, blockDim>>>(d_input, d_temp, width, height, radius, d_kernel);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Apply vertical blur
    vertical_blur_kernel<<<gridDim, blockDim>>>(d_temp, d_output, width, height, radius, d_kernel);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_image, d_output, image_size, cudaMemcpyDeviceToHost));

    // Copy back to row_pointers format
    for (int y = 0; y < height; y++)
    {
        memcpy(row_pointers[y], &h_image[y * width * 4], width * 4);
    }

    // Clean up
    free(h_image);
    free(h_kernel);
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_temp));
    CHECK_CUDA_ERROR(cudaFree(d_kernel));
}

int main(int argc, char *argv[])
{
    const char *input_file = "image.png";
    const char *output_file = "cuda_out.png";
    int blur_radius = 10; // Default value

    if (argc > 1)
    {
        blur_radius = atoi(argv[1]);
        if (blur_radius <= 0)
        {
            printf("Invalid blur radius. Using default value: 10\n");
            blur_radius = 10;
        }
        input_file = argv[2];
    }
    else
    {
        printf("No blur radius specified. Using default value: 10\n");
    }
    printf("Using blur radius: %d\n", blur_radius);

    png_bytep *row_pointers;
    int width, height;
    clock_t start, end, read_start, read_end, write_start, write_end;
    double cpu_time_used, read_time_used, write_time_used;

    // Start measuring read time
    printf("Reading image from %s\n", input_file);
    read_start = clock();
    read_png_file(input_file, &row_pointers, &width, &height);
    read_end = clock();
    read_time_used = ((double)(read_end - read_start)) / CLOCKS_PER_SEC;
    printf("Image read successfully\nImage dimensions: %d x %d\n\n", width, height);

    // Start measuring processing time
    printf("Starting CUDA Blurring Process\n");
    start = clock();
    apply_gaussian_blur_cuda(row_pointers, width, height, blur_radius);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("CUDA Blurring Process Completed\n\n");

    // Start measuring write time
    printf("Writing image to %s\n", output_file);
    write_start = clock();
    write_png_file(output_file, row_pointers, width, height);
    write_end = clock();
    write_time_used = ((double)(write_end - write_start)) / CLOCKS_PER_SEC;
    printf("Image written successfully\n\n");

    printf("Freeing memory\n");
    for (int y = 0; y < height; y++)
    {
        free(row_pointers[y]);
    }
    free(row_pointers);
    printf("Memory freed\n\n");

    printf("Execution Summary:\n");
    printf("Time taken for reading: %f seconds\n", read_time_used);
    printf("Time taken for CUDA Gaussian blur with %d radius: %f seconds\n", blur_radius, cpu_time_used);
    printf("Time taken for writing: %f seconds\n", write_time_used);

    return 0;
}
