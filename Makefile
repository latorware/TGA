CUDA_HOME   = /Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.2

NVCC        = $(CUDA_HOME)/bin/nvcc.exe
NVCC_FLAGS  = -O3 -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include -arch=native --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib
EXE	    = puzzle1D.exe
OBJ	    = puzzle1D.o

default: $(EXE)

puzzle1D.o: puzzle1D.cu	
	$(NVCC) -c -o $@ puzzle1D.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)