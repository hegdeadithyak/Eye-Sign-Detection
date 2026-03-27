#!/bin/bash
# HPC Training Script for EyeWave YOLO
# Usage: sbatch train_hpc.sh  (if using SLURM) or bash train_hpc.sh

# Uncomment and adjust these SLURM directives if you are using SLURM:
# #SBATCH --job-name=eyewave_train
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=8
# #SBATCH --gres=gpu:1          # Request 1 GPU
# #SBATCH --mem=32G
# #SBATCH --time=48:00:00

# Stop on first error
set -e

echo "=========================================================="
echo "Starting EyeWave HPC Training Job"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "=========================================================="

# 1. Load your Python/CUDA modules if required on your HPC (e.g., module load python/3.10 cuda/11.8)
# module load python/3.10

# 2. Set up Virtual Environment
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "=> Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

echo "=> Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# 3. Install dependencies
echo "=> Upgrading pip..."
pip install --upgrade pip

echo "=> Installing local ultralytics and dependencies (editable mode)..."
# In a venv, -e works cleanly without permission errors
pip install -e ./ultralytics

# Optionally, ensure torch with CUDA is installed if the default is CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Run the training script
echo "=> Starting YOLO training..."
# Using PYTHONPATH just to be absolutely sure the local ultralytics is prioritized
PYTHONPATH=./ultralytics python3 train_yolo.py \
    --data data.yaml \
    --model ultralytics/ultralytics/cfg/models/11/eyewave_transformer.yaml \
    --epochs 50 \
    --batch 32 \
    --imgsz 224

echo "=========================================================="
echo "Training completed: $(date)"
echo "=========================================================="
