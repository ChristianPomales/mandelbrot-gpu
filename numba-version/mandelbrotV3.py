from __future__ import print_function, division, absolute_import

from timeit import default_timer as timer
from matplotlib.pylab import imshow, jet, savefig, ion
import numpy as np
from numba import njit

@njit(parallel=True)
def get_color(row, col, max_iter):
    color = 0
    z = 0.0j
    for color in range(max_iter):
        z = z*z + complex(row, col)
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return color
    return 255

@njit(parallel=True)
def generate_mandelbrot(image):
    px = ((3.0) / WIDTH)
    py = ((2.0) / HEIGHT) 

    for row in range(WIDTH):
        i = -1.75 + row * px
        for col in range(HEIGHT):
            j = -1.0 + col * py
            color = get_color(i, j, MAX_ITER)
            image[col, row] = color
    return image

SIZE = 4096
WIDTH = SIZE
HEIGHT = SIZE
MAX_ITER = 10000

def main():   
  
    image = np.zeros((WIDTH, HEIGHT), dtype=np.uint8)
    s = timer()
    generate_mandelbrot(image)
    e = timer()
    imshow(image)
    savefig('mandelbrot.png')

if __name__== "__main__":
  main()