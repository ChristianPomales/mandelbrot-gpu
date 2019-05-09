CFLAGS=-g -std=c99 -Ofast
BIN=mandelbrot

${BIN}: main.c
	${CC} ${CFLAGS} $^ -o $@

clean:
	rm -f ${BIN} *.ppm

.PHONY: clean
