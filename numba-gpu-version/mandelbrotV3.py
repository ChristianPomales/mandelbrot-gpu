from __future__ import print_function, division, absolute_import

from timeit import default_timer as timer
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.pylab import imshow, jet, savefig, ion
import numpy as np
from numba import cuda

@cuda.jit(device=True)
def get_color(row, col, max_iter):
    color = 0
    z = 0.0j
    for color in range(max_iter):
        z = z*z + complex(row, col)
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return color
    return max_iter

@cuda.jit
def generate_mandelbrot(image):
    px = ((3.0) / WIDTH)
    py = ((2.0) / HEIGHT) 

    startX = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    startY = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y

    for row in range(startX, WIDTH, gridX):
        i = -1.75 + row * px
        for col in range(startY, HEIGHT, gridY):
            j = -1.0 + col * py
            color = get_color(i, j, MAX_ITER)
            image[col, row] = color
    #return image

SIZE = 4096
WIDTH = SIZE
HEIGHT = SIZE
MAX_ITER = 20

def main():   

    blockdim = (32, 8)
    griddim = (32,16)

    image = np.zeros((WIDTH, HEIGHT), dtype=np.uint8)
    #s = timer()
    d_image = cuda.to_device(image)
    generate_mandelbrot[griddim, blockdim](d_image) 
    d_image.to_host()
    #generate_mandelbrot(image)
    # e = timer() - s
    imshow(image)
    savefig('mandelbrot.png')

if __name__== "__main__":
  main()