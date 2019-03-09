NVCC=nvcc
CUDAFLAGS= --std=c++11 -O3 -arch=sm_60

dot.exe: dot.cu
	$(NVCC) $(CUDAFLAGS) dot.cu -o dot.exe
clean:
	rm -rf *.o *.exe