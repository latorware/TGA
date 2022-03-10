#!/bin/bash

### Directivas para el gestor de colas
#SBATCH --job-name=deviceQuery
#SBATCH -D .
#SBATCH --output=submit-deviceQuery.o%j
#SBATCH --error=submit-deviceQuery.e%j
#SBATCH -A cuda
#SBATCH -p cuda
#SBATCH --gres=gpu:1

export PATH=/Soft/cuda/11.2.1/bin:$PATH

./deviceQuery.exe
