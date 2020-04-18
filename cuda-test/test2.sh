#!/usr/bin/env bash

#SBATCH --job-name=test2
#SBATCH--partition=wacc
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --time=0-00:00:30
#SBATCH --output=test2.out -e test2.err
#SBATCH --gres=gpu:1 -c 1

cd $SLURM_SUBMIT_DIR
./test2

