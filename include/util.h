#ifndef UTIL_H
#define UTIL_H

#include <png.h>

/**
 * Reads a PNG file and loads it into memory
 *
 * @param filename Path to the PNG file
 * @param row_pointers Pointer to array of row pointers to store image data
 * @param width Pointer to store image width
 * @param height Pointer to store image height
 */
void read_png_file(const char *filename, png_bytep **row_pointers, int *width, int *height);

/**
 * Writes PNG data to a file
 *
 * @param filename Path to the output PNG file
 * @param row_pointers Array of row pointers containing image data
 * @param width Image width
 * @param height Image height
 */
void write_png_file(const char *filename, png_bytep *row_pointers, int width, int height);

#endif /* UTIL_H */