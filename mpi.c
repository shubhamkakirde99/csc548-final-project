#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdint.h> // Include this header for uint8_t
#include <mpi.h>    // Add MPI header
#include "include/util.h"

// Apply Gaussian blur with a configurable kernel size
void apply_gaussian_blur(png_bytep *row_pointers, int width, int real_height, int radius, int start_row, int height, int rank)
{
    height = fmin(real_height - start_row, height); // Ensure height does not exceed real height
    // Create a copy of the image to read from while writing to the original
    png_bytep *temp_rows = (png_bytep *)malloc(sizeof(png_bytep) * real_height);
    for (int y = 0; y < real_height; y++)
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
    for (int y = start_row; y < start_row + height; y++)
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
    for (int y = 0; y < real_height; y++)
    {
        memcpy(temp_rows[y], row_pointers[y], width * 4);
    }

    int temp_iy_debug[real_height];
    for (int y = 0; y < real_height; y++)
    {
        temp_iy_debug[y] = 0; // Initialize the debug array
    }
    // Apply vertical blur
    for (int y = start_row; y < start_row + height; y++)
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
                if (iy >= real_height)
                    iy = real_height - 1;
                temp_iy_debug[iy] += 1; // Debugging line
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
    // Print the debug array values
    // printf("%d---------------------------startrow: %d:\n", rank, start_row);
    // for (int y = 0; y < real_height; y++)
    // {
    //     printf("%d Row %d:  %d \n", rank, y, temp_iy_debug[y]);
    // }

    // Free temporary image
    for (int y = 0; y < real_height; y++)
    {
        free(temp_rows[y]);
    }
    free(temp_rows);
    free(kernel);
}

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char *input_file = "spidey.png";
    const char *output_file = "out_mpi.png"; // Changed output filename
    int blur_radius = 10;                    // Default value

    if (argc > 1)
    {
        blur_radius = atoi(argv[1]);
        if (blur_radius <= 0)
        {
            if (rank == 0)
                printf("Invalid blur radius. Using default value: 10\n");
            blur_radius = 10;
        }
        if (argc > 2)
            input_file = argv[2];
    }
    else
    {
        if (rank == 0)
            printf("No blur radius specified. Using default value: 10\n");
    }

    if (rank == 0)
        printf("Using blur radius: %d\n", blur_radius);

    png_bytep *row_pointers = NULL;
    int width, height;
    clock_t start, end, read_start, read_end, write_start, write_end;
    double cpu_time_used, read_time_used, write_time_used;

    // Only root process reads the file
    if (rank == 0)
    {
        // Start measuring read time
        printf("Reading image from %s\n", input_file);
        read_start = clock();
        read_png_file(input_file, &row_pointers, &width, &height);
        read_end = clock();
        read_time_used = ((double)(read_end - read_start)) / CLOCKS_PER_SEC;
        printf("Image read successfully\nImage dimensions: %d x %d\n\n", width, height);
    }

    // Broadcast image dimensions to all processes
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate rows per process for work distribution
    int rows_per_proc = height / size;
    int remainder = height % size;

    // Each process calculates its own portion of work
    int start_row = rank * rows_per_proc + (rank < remainder ? rank : remainder);
    int num_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    int end_row = start_row + num_rows;

    // Create a buffer for broadcasting the entire image
    unsigned char *buffer = NULL;
    size_t buffer_size = width * height * 4;
    buffer = (unsigned char *)malloc(buffer_size);

    // Root process prepares the buffer
    if (rank == 0)
    {
        for (int y = 0; y < height; y++)
        {
            memcpy(buffer + y * width * 4, row_pointers[y], width * 4);
        }
    }

    // Broadcast the entire image to all processes
    MPI_Bcast(buffer, buffer_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Each process creates its own complete copy of the image
    png_bytep *local_row_pointers = (png_bytep *)malloc(height * sizeof(png_bytep));
    for (int y = 0; y < height; y++)
    {
        local_row_pointers[y] = (png_byte *)malloc(width * 4);
        memcpy(local_row_pointers[y], buffer + y * width * 4, width * 4);
    }

    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);

    // Start measuring processing time
    if (rank == 0)
        printf("Starting Blurring Process\n");
    start = clock();

    // Apply the gaussian blur only to the assigned portion of the image
    apply_gaussian_blur(local_row_pointers, width, height, blur_radius, fmax(start_row - blur_radius, 0), num_rows * 2, rank);

    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Create a buffer for the results from each process
    unsigned char *result_buffer = (unsigned char *)malloc(width * num_rows * 4);
    for (int y = 0; y < num_rows; y++)
    {
        memcpy(result_buffer + y * width * 4, local_row_pointers[start_row + y], width * 4);
    }

    // Root process will receive the results
    unsigned char **recv_buffers = NULL;
    int *recv_counts = NULL;
    int *displacements = NULL;

    if (rank == 0)
    {
        recv_buffers = (unsigned char **)malloc(size * sizeof(unsigned char *));
        recv_counts = (int *)malloc(size * sizeof(int));
        displacements = (int *)malloc(size * sizeof(int));

        int offset = 0;
        for (int i = 0; i < size; i++)
        {
            int rows = rows_per_proc + (i < remainder ? 1 : 0);
            recv_counts[i] = rows * width * 4;
            displacements[i] = offset;
            offset += recv_counts[i];

            if (i > 0)
            { // Skip allocation for rank 0 (we'll use result_buffer directly)
                recv_buffers[i] = (unsigned char *)malloc(recv_counts[i]);
            }
            else
            {
                recv_buffers[i] = result_buffer;
            }
        }
    }

    // Gather the processed rows from each process
    MPI_Gatherv(result_buffer, width * num_rows * 4, MPI_UNSIGNED_CHAR,
                buffer, recv_counts, displacements, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Root process writes the output file
    if (rank == 0)
    {
        printf("Blurring Process Completed\n\n");

        // Convert buffer back to row_pointers
        for (int y = 0; y < height; y++)
        {
            memcpy(row_pointers[y], buffer + y * width * 4, width * 4);
        }

        // Start measuring write time
        printf("Writing image to %s\n", output_file);
        write_start = clock();
        write_png_file(output_file, row_pointers, width, height);
        write_end = clock();
        write_time_used = ((double)(write_end - write_start)) / CLOCKS_PER_SEC;
        printf("Image written successfully\n\n");

        printf("Execution Summary:\n");
        printf("Time taken for reading: %f seconds\n", read_time_used);
        printf("Time taken for Gaussian blur with %d radius: %f seconds\n", blur_radius, cpu_time_used);
        printf("Time taken for writing: %f seconds\n", write_time_used);

        // Free root-only resources
        for (int i = 1; i < size; i++)
        {
            free(recv_buffers[i]);
        }
        free(recv_buffers);
        free(recv_counts);
        free(displacements);

        // Free image data
        for (int y = 0; y < height; y++)
        {
            free(row_pointers[y]);
        }
        free(row_pointers);
    }

    // Free local resources
    for (int y = 0; y < height; y++)
    {
        free(local_row_pointers[y]);
    }
    free(local_row_pointers);
    free(buffer);
    free(result_buffer);

    MPI_Finalize();
    return 0;
}
