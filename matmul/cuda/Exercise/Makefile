NVCC=nvcc
CUDAFLAGS= --std=c++11 -O3 -arch=sm_60 -I../../../include/

matmul.exe: matmul.cu
	$(NVCC) $(CUDAFLAGS) matmul.cu -o matmul.exe
clean:
	rm -rf *.o *.exe