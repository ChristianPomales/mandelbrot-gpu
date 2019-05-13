CC=gcc-9
CFLAGS= -std=c99 -Ofast -fopenmp
BIN=mandelbrot

${BIN}: main.c
	${CC} ${CFLAGS} $^ -o $@

clean:
	rm -f ${BIN} *.ppm

.PHONY: clean
