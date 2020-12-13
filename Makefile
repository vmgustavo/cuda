build_c:
	R CMD SHLIB distmat_c.c
build_cuda:
	nvcc --compiler-options '-fPIC' --shared -I/usr/share/R/include -o distmat_cuda.so distmat_cuda.cu
build: build_c build_cuda

clean:
	-rm *.o
	-rm *.so
