#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --reservation=psistemi
#SBATCH -G1
#SBATCH --output=outGPU.txt


module load CUDA
nvcc sobel.cu -o sobelGPU --expt-relaxed-constexpr

srun sobelGPU valve.png outputGPU.png

# srun --partition=gpu --reservation=psistemi -G1 sobelGPU valve.png outputGPU.png