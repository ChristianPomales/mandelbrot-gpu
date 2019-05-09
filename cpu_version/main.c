#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

#define MAX_ITER 10000
#define SIZE 1024
#define WIDTH SIZE
#define HEIGHT SIZE

typedef enum
{
  RGB_RED = 0,
  RGB_GREEN,
  RGB_BLUE,
} rgb_t;

// performs mandelbrot calculation for every pixel
void generate_mandelbrot(unsigned int *mandelbrot_picture)
{
  for (int i = 0; i < HEIGHT; i++)
  {
    for (int j = 0; j < WIDTH; j++)
    {
      int row = i;
      int col = j;
      int idx = row * WIDTH + col;

      if (col >= WIDTH || row >= HEIGHT)
      {
        return;
      }

      float x0 = ((float)col / WIDTH) * 3.5f - 2.5f;
      float y0 = ((float)row / HEIGHT) * 3.5f - 1.75f;

      float x = 0.0f;
      float y = 0.0f;
      int iter = 0;
      float xtemp;
      while ((x * x + y * y <= 4.0f) && (iter < MAX_ITER))
      {
        xtemp = x * x - y * y + x0;
        y = 2.0f * x * y + y0;
        x = xtemp;
        iter++;
      }

      int color = iter * 5;
      if (color >= 256)
      {
        color = 0;
      }
      mandelbrot_picture[idx] = color;
    }
  }
}

// writes mandelbrot array to file as a .ppm image
void write_image(char *filename, unsigned int *mandelbrot_picture)
{
  unsigned char pixel_black[3] = {0, 0, 0};
  unsigned char pixel_white[3] = {255, 255, 255};
  unsigned int esc_time;
  FILE *f = fopen(filename, "w");

  fprintf(f, "P6\n");
  fprintf(f, "#The comment string\n");
  fprintf(f, "%d %d\n", WIDTH, HEIGHT);
  fprintf(f, "%d\n", 255);

  for (int i = 0; i < HEIGHT; i++)
  {
    for (int j = 0; j < WIDTH; j++)
    {
      esc_time = mandelbrot_picture[i * WIDTH + j];
      if (esc_time)
      {
        unsigned char pixel[3] = {esc_time, 256 - esc_time, 255};
        fwrite(pixel, sizeof(pixel_white), 1, f);
      }
      else
      {
        fwrite(pixel_black, sizeof(pixel_black), 1, f);
      }
    }
  }

  fclose(f);
}

int main(int argc, char *argv[])
{
  unsigned int *mandelbrot_picture = (unsigned int *)malloc(WIDTH * HEIGHT * sizeof(unsigned int));

  generate_mandelbrot(mandelbrot_picture);

  char *filename = argc > 1 ? argv[1] : "out.ppm";
  write_image(filename, mandelbrot_picture);

  return 0;
}
