/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <cuComplex.h>
#include <stdio.h>

#define TILE_SIZE 16
#define MAX_ITER 10000
#define SIZE 1024
#define WIDTH SIZE
#define HEIGHT SIZE

__global__ void
generate_mandelbrot_kern(unsigned int* mandelbrot_picture)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y; // WIDTH
  int col = blockIdx.x * blockDim.x + threadIdx.x; // HEIGHT
  int idx = row * WIDTH + col;

  if (col >= WIDTH || row >= HEIGHT) {
    return;
  }

  float x0 = ((float)col / WIDTH) * 3.5f - 2.5f;
  float y0 = ((float)row / HEIGHT) * 3.5f - 1.75f;

  float x = 0.0f;
  float y = 0.0f;
  int iter = 0;
  float xtemp;
  while ((x * x + y * y <= 4.0f) && (iter < MAX_ITER)) {
    xtemp = x * x - y * y + x0;
    y = 2.0f * x * y + y0;
    x = xtemp;
    iter++;
  }

  int color = iter * 5;
  if (color >= 256) {
    color = 0;
  }
  mandelbrot_picture[idx] = color;
}

void
generate_mandelbrot(unsigned int* mandelbrot_picture)
{
  // Initialize thread block and kernel grid dimensions ---------------------

  const unsigned int BLOCK_SIZE = TILE_SIZE;

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(WIDTH / dimBlock.x, HEIGHT / dimBlock.y);

  // Invoke CUDA kernel -----------------------------------------------------

  generate_mandelbrot_kern<<<dimGrid, dimBlock>>>(mandelbrot_picture);
}
