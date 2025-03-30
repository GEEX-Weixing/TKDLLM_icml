#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a800:2
#SBATCH --output=%j.out
#SBATCH --error=%j.err

python tete.py
