#!/bin/bash

### Directivas para el gestor de colas
#SBATCH --job-name=PUZZLE1D
#SBATCH -D .
#SBATCH --output=submit-PUZZLE1D.o%j
#SBATCH --error=submit-PUZZLE1D.e%j
#SBATCH -A cuda
#SBATCH -p cuda
#SBATCH --gres=gpu:4

export PATH=/Soft/cuda/11.2.1/bin:$PATH

echo""
echo "./puzzle1D.exe 30000 Y"
./puzzle1D.exe 30000 Y

echo""
echo "nvprof ./puzzle1D.exe 10000 N"
nvprof ./puzzle1D.exe 10000 N

echo""
echo "nvprof ./puzzle1D.exe 60000 N"
nvprof ./puzzle1D.exe 60000 N



