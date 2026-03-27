#!/bin/bash
# SLURM submission script to run train.py on the HPC cluster

#SBATCH --job-name=eyewave_train
#SBATCH --output=eyewave_train_%j.log
#SBATCH --error=eyewave_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1          # Request 1 GPU
#SBATCH --mem=32G
#SBATCH --time=48:00:00

echo "=========================================================="
echo "Starting EyeWave HPC Training Job"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "=========================================================="

# Load required modules if necessary on your HPC (uncomment and adjust if needed)
# module load python/3.9 cuda/11.8

# Run the Python orchestrator script globally
python3 /home/hutlab_int/Hegde_netravaad/train.py

echo "=========================================================="
echo "Job completed at $(date)"
echo "=========================================================="
