#!/usr/bin/env bash

#SBATCH --job-name=feature
#SBATCH--partition=wacc
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --time=0-00:01:30
#SBATCH --output=feature.out -e feature.err
#SBATCH --gres=gpu:1 -c 1

cd $SLURM_SUBMIT_DIR

module purge
module load gcc/0_cuda/7.1.0
module load cuda/10.0

g++ -o feature feature.cpp `pkg-config opencv --cflags --libs`
./feature --left ./altera.jpg --right ./altera_in_scene.jpg

