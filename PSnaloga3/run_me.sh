#!/bin/bash
#SBATCH --nodes=1
#SBATCH --array=0-4
#SBATCH --reservation=psistemi
#SBATCH --output=agent-%a.txt

# path=./telefonUDP

module load Go
go build naloga3.go
srun naloga3 -p 15000 -id $SLURM_ARRAY_TASK_ID -N $SLURM_ARRAY_TASK_COUNT -M 7 -K 2