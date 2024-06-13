#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --reservation=psistemi
#SBATCH -G1
#SBATCH --output=out.txt


module load CUDA
nvcc discover-device.cu -o discover-device

srun discover-device
# --partition=gpu --reservation=psistemi -G1

# srun --partition=gpu --reservation=psistemi -G1 sample valve.png output.png