#!/bin/bash
#SBATCH -p gpu-week 
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A_%a.out  # Output log file (%x: job name, %A: job ID, %a: array task ID)
#SBATCH --error=logs/%x_%A_%a.err   # Error log file

module load anaconda/2023-06-19
source /opt/psi/TOMCAT/anaconda/2023-06-19/conda/etc/profile.d/conda.sh
conda activate ptycho

n_fluences=20
n_modes=5

echo "Running simulation for n_modes=$n_modes and n_fluences=$n_fluences"

python -u sim_bandlim_corrections.py $n_modes $n_fluences

echo "Simulation for n_modes=$n_modes and n_fluences=$n_fluences completed!"
