#!/bin/bash

### Directivas para el gestor de colas
#SBATCH --job-name=PUZZLE2D
#SBATCH -D .
#SBATCH --output=submit-PUZZLE2D.o%j
#SBATCH --error=submit-PUZZLE2D.e%j
#SBATCH -A cuda
#SBATCH -p cuda
#SBATCH --gres=gpu:4

export PATH=/Soft/cuda/11.2.1/bin:$PATH

./puzzle2D.exe 512 1024 Y

#nvprof ./puzzle2D.exe 1024 512 Y
#nvprof ./puzzle2D.exe 1024 1024 Y

#nvprof --print-gpu-trace  ./puzzle2D.exe 1024 1024 Y
#nvprof --print-gpu-summary ./puzzle2D.exe 1024 1024 Y
#nvprof --metrics all ./puzzle2D.exe 1024 1024 Y
#nvprof --metrics sm_efficiency,achieved_occupancy,gld_requested_throughput,gst_requested_throughput,dram_utilization ./puzzle2D.exe 1024 1024 Y


