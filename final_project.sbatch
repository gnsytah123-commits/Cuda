#!/bin/bash
#SBATCH --job-name=cuda_proximity
#SBATCH --partition=rtx3070
#SBATCH --time=0-00:10:00
#SBATCH --output=job-%J.out
#SBATCH --error=job-%J.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

module load anaconda
source activate aharoni

nvcc -O3 final_project.cu -o final_project -L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -lcudart
./final_project
