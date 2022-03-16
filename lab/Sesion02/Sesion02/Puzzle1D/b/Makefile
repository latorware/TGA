CUDA_HOME = ${CUDA_PATH}

NVCC = $(CUDA_HOME)\bin\nvcc.exe
NVCC_FLAGS = -std=c++11 -O3 -I$(CUDA_HOME)\include -I$(CUDA_HOME)\src I$(CUDA_HOME)\lib\x64 I$(CUDA_HOME)\lib\Win32 -I$(CUDA_HOME)\include\cuda -arch=compute_86 --ptxas-options=-v
LD_FLAGS = -lcudart -Xlinker -rpath,$(CUDA_HOME)\lib\x64
EXE = puzzle1D.exe
OBJ = puzzle1D.o


default: $(EXE)

puzzle1D.o: puzzle1D.cu	
	$(NVCC) -c -o $@ puzzle1D.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
