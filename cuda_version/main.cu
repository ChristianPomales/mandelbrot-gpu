/******************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ******************************************************************************/

#include "kernel.cu"
#include <stdio.h>
#include <stdlib.h>

#define SIZE 1024
#define WIDTH SIZE
#define HEIGHT SIZE
#define size_x SIZE
#define size_y SIZE

typedef enum
{
  RGB_RED = 0,
  RGB_GREEN,
  RGB_BLUE,
} rgb_t;

void
write_image(const char* filename, unsigned int* mandelbrot_picture_h)
{
  unsigned char pixel_black[3] = { 0, 0, 0 };
  unsigned char pixel_white[3] = { 255, 255, 255 };
  unsigned int esc_time;
  FILE* f = fopen(filename, "w");

  fprintf(f, "P6\n");
  fprintf(f, "#The comment string\n");
  fprintf(f, "%d %d\n", size_x, size_y);
  fprintf(f, "%d\n", 255);

  for (int i = 0; i < size_y; i++) {
    for (int j = 0; j < size_x; j++) {
      esc_time = mandelbrot_picture_h[i * size_x + j];
      if (esc_time) {
        unsigned char pixel[3] = { esc_time, 256 - esc_time, 255 };
        fwrite(pixel, sizeof(pixel_white), 1, f);
      } else {
        fwrite(pixel_black, sizeof(pixel_black), 1, f);
      }
    }
  }

  fclose(f);
}

int
main(int argc, char* argv[])
{

  cudaError_t cuda_ret;

  // Initialize host variables ----------------------------------------------
  unsigned int* mandelbrot_picture_h;
  unsigned int* mandelbrot_picture_d;

  mandelbrot_picture_h =
    (unsigned int*)malloc(size_x * size_y * sizeof(unsigned int));

  // Allocate device variables ----------------------------------------------
  cudaMalloc((void**)&mandelbrot_picture_d,
             size_x * size_y * sizeof(unsigned int));

  cudaDeviceSynchronize();

  // Copy host variables to device ------------------------------------------
  cudaMemcpy(mandelbrot_picture_d,
             mandelbrot_picture_h,
             size_x * size_y * sizeof(unsigned int),
             cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();

  // Launch kernel using standard sgemm interface ---------------------------
  generate_mandelbrot(mandelbrot_picture_d);

  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess) {
    printf("Unable to launch kernel");
    exit(1);
  }

  // Copy device variables from host ----------------------------------------

  cudaMemcpy(mandelbrot_picture_h,
             mandelbrot_picture_d,
             size_x * size_y * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  // Write host array to image ----------------------------------------------

  const char* filename = argc > 1 ? argv[1] : "out.ppm";
  write_image(filename, mandelbrot_picture_h);

  // Free memory ------------------------------------------------------------

  free(mandelbrot_picture_h);

  cudaFree(mandelbrot_picture_d);

  return 0;
}
