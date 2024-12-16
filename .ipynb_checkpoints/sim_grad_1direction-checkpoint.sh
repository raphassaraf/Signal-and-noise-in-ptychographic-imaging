#!/bin/bash
#SBATCH -p gpu-week 
#SBATCH --gres=gpu:1
#SBATCH --array=1-6
#SBATCH --output=logs/%x_%A_%a.out  # Output log file (%x: job name, %A: job ID, %a: array task ID)
#SBATCH --error=logs/%x_%A_%a.err   # Error log file

module load anaconda/2023-06-19
source /opt/psi/TOMCAT/anaconda/2023-06-19/conda/etc/profile.d/conda.sh
conda activate ptycho

n_fluences=20
principal_mode_weight=$SLURM_ARRAY_TASK_ID

echo "Running simulation for principal_mode_weight=$principal_mode_weight and n_fluences=$n_fluences"

python -u sim_grad_1direction.py $principal_mode_weight $n_fluences

echo "Simulation for principal_mode_weight=$principal_mode_weight and n_fluences=$n_fluences completed!"
