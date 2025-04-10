#include <stdio.h>
#include <stdlib.h>
#include <png.h>

void read_png_file(const char *filename, png_bytep **row_pointers, int *width, int *height)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        perror("File could not be opened for reading");
        exit(EXIT_FAILURE);
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png)
    {
        perror("png_create_read_struct failed");
        exit(EXIT_FAILURE);
    }

    png_infop info = png_create_info_struct(png);
    if (!info)
    {
        perror("png_create_info_struct failed");
        exit(EXIT_FAILURE);
    }

    if (setjmp(png_jmpbuf(png)))
    {
        perror("Error during init_io");
        exit(EXIT_FAILURE);
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    *width = png_get_image_width(png, info);
    *height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if (bit_depth == 16)
        png_set_strip_16(png);

    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);

    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);

    if (png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);

    if (color_type == PNG_COLOR_TYPE_RGB ||
        color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

    if (color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * (*height));
    for (int y = 0; y < *height; y++)
    {
        (*row_pointers)[y] = (png_byte *)malloc(png_get_rowbytes(png, info));
    }

    png_read_image(png, *row_pointers);

    fclose(fp);
    png_destroy_read_struct(&png, &info, NULL);
}

void write_png_file(const char *filename, png_bytep *row_pointers, int width, int height)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp)
    {
        perror("File could not be opened for writing");
        exit(EXIT_FAILURE);
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png)
    {
        perror("png_create_write_struct failed");
        exit(EXIT_FAILURE);
    }

    png_infop info = png_create_info_struct(png);
    if (!info)
    {
        perror("png_create_info_struct failed");
        exit(EXIT_FAILURE);
    }

    if (setjmp(png_jmpbuf(png)))
    {
        perror("Error during init_io");
        exit(EXIT_FAILURE);
    }

    png_init_io(png, fp);

    if (setjmp(png_jmpbuf(png)))
    {
        perror("Error during writing header");
        exit(EXIT_FAILURE);
    }

    png_set_IHDR(
        png,
        info,
        width, height,
        8,
        PNG_COLOR_TYPE_RGB_ALPHA,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    if (setjmp(png_jmpbuf(png)))
    {
        perror("Error during writing bytes");
        exit(EXIT_FAILURE);
    }

    png_write_image(png, row_pointers);

    if (setjmp(png_jmpbuf(png)))
    {
        perror("Error during end of write");
        exit(EXIT_FAILURE);
    }

    png_write_end(png, NULL);

    fclose(fp);
    png_destroy_write_struct(&png, &info);
}
