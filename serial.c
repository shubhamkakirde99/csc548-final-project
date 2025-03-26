#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdint.h> // Include this header for uint8_t
#include "include/util.h"

// Apply Gaussian blur with a configurable kernel size
void apply_gaussian_blur(png_bytep *row_pointers, int width, int height, int radius)
{
    // Create a copy of the image to read from while writing to the original
    png_bytep *temp_rows = (png_bytep *)malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++)
    {
        temp_rows[y] = (png_byte *)malloc(width * 4);
        memcpy(temp_rows[y], row_pointers[y], width * 4);
    }

    // Calculate sigma based on radius
    float sigma = radius / 2.0;

    // Create Gaussian kernel
    int kernel_size = 2 * radius + 1;
    float *kernel = (float *)malloc(kernel_size * sizeof(float));
    float sum = 0.0;

    // Fill kernel with Gaussian values
    for (int i = 0; i < kernel_size; i++)
    {
        int x = i - radius;
        kernel[i] = exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }

    // Normalize kernel
    for (int i = 0; i < kernel_size; i++)
    {
        kernel[i] /= sum;
    }

    // Apply horizontal blur first (into a temporary buffer)
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            float r = 0, g = 0, b = 0, a = 0;

            for (int i = -radius; i <= radius; i++)
            {
                int ix = x + i;
                // Handle boundary conditions
                if (ix < 0)
                    ix = 0;
                if (ix >= width)
                    ix = width - 1;

                png_bytep px = &(temp_rows[y][ix * 4]);
                float weight = kernel[i + radius];

                r += px[0] * weight;
                g += px[1] * weight;
                b += px[2] * weight;
                a += px[3] * weight;
            }

            // Write result back to original image
            png_bytep out_px = &(row_pointers[y][x * 4]);
            out_px[0] = (uint8_t)r;
            out_px[1] = (uint8_t)g;
            out_px[2] = (uint8_t)b;
            out_px[3] = (uint8_t)a;
        }
    }

    // Copy the current result back to temp for the vertical pass
    for (int y = 0; y < height; y++)
    {
        memcpy(temp_rows[y], row_pointers[y], width * 4);
    }

    // Apply vertical blur
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            float r = 0, g = 0, b = 0, a = 0;

            for (int i = -radius; i <= radius; i++)
            {
                int iy = y + i;
                // Handle boundary conditions
                if (iy < 0)
                    iy = 0;
                if (iy >= height)
                    iy = height - 1;

                png_bytep px = &(temp_rows[iy][x * 4]);
                float weight = kernel[i + radius];

                r += px[0] * weight;
                g += px[1] * weight;
                b += px[2] * weight;
                a += px[3] * weight;
            }

            // Write result back to original image
            png_bytep out_px = &(row_pointers[y][x * 4]);
            out_px[0] = (uint8_t)r;
            out_px[1] = (uint8_t)g;
            out_px[2] = (uint8_t)b;
            out_px[3] = (uint8_t)a;
        }
    }

    // Free temporary image
    for (int y = 0; y < height; y++)
    {
        free(temp_rows[y]);
    }
    free(temp_rows);
    free(kernel);
}

int main(int argc, char *argv[])
{

    const char *input_file = "spidey.png";
    const char *output_file = "out_serial.png";
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
    printf("Starting Blurring Process\n");
    start = clock();
    apply_gaussian_blur(row_pointers, width, height, 50);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Blurring Process Completed\n\n");

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
    printf("Time taken for Gaussian blur with %d radius: %f seconds\n", blur_radius, cpu_time_used);
    printf("Time taken for writing: %f seconds\n", write_time_used);
    return 0;
}
