build_c:
	R CMD SHLIB distmat_c.c
build_cu:
	R CMD SHLIB distmat_cuda.cu
build_cuda:
	nvcc --compiler-options '-fPIC' --shared -I/usr/share/R/include -o distmat_cuda.so distmat_cuda.cu

clean:
	-rm *.o
	-rm *.so
