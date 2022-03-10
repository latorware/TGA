#!/bin/bash

### Directivas para el gestor de colas
#SBATCH --job-name=sAXPYp
#SBATCH -D .
#SBATCH --output=submit-sAXPYp.o%j
#SBATCH --error=submit-sAXPYp.e%j
#SBATCH -A cuda
#SBATCH -p cuda
#SBATCH --gres=gpu:1

export PATH=/Soft/cuda/11.2.1/bin:$PATH

N=$((1024*1024*16))
nThreads=$((1024))
echo""
echo "./SaxpyP.exe $N $nThreads"
./SaxpyP.exe $N $nThreads

echo""
echo "nvprof -V"
nvprof -V

echo""
echo "nvprof ./SaxpyP.exe $N $nThreads"
nvprof ./SaxpyP.exe $N $nThreads

echo""
echo "nvprof --print-gpu-summary ./SaxpyP.exe $N $nThreads"
nvprof --print-gpu-summary ./SaxpyP.exe $N $nThreads

