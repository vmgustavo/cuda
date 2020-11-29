build:
	R CMD SHLIB distmat_c.c
clean:
	-rm distmat_c.o
	-rm distmat_c.so
