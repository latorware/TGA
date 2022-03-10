#!/bin/bash

### Directivas para el gestor de colas
#SBATCH --job-name=PUZZLE3D
#SBATCH -D .
#SBATCH --output=submit-PUZZLE3D.o%j
#SBATCH --error=submit-PUZZLE3D.e%j
#SBATCH -A cuda
#SBATCH -p cuda
#SBATCH --gres=gpu:4

export PATH=/Soft/cuda/11.2.1/bin:$PATH

./puzzle3D.exe 128 64 128 Y

#nvprof ./puzzle3D.exe 32 64 128 Y
#nvprof ./puzzle3D.exe 128 64 128 Y
#nvprof ./puzzle3D.exe 64 64 64 Y

