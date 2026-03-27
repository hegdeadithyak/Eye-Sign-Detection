#!/bin/bash
# SLURM submission script to run YOLOv11 on the HPC cluster

#SBATCH --job-name=eyewave_train
#SBATCH --output=eyewave_train_%j.log
#SBATCH --error=eyewave_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1          
#SBATCH --mem=32G
#SBATCH --time=48:00:00

# Cluster managers execute from strange temporary directories. We robustly 
# capture the original submission directory for both SLURM and PBS.
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    BASE_DIR="$SLURM_SUBMIT_DIR"
elif [ -n "$PBS_O_WORKDIR" ]; then
    BASE_DIR="$PBS_O_WORKDIR"
else
    BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
fi

echo "=========================================================="
echo "Starting EyeWave HPC Training Job (Isolated Python 3.10)"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "=========================================================="

# 1. Install Micromamba locally if it doesn't exist (Zero root required)
export MAMBA_ROOT_PREFIX="$BASE_DIR/micromamba"
if [ ! -f "$MAMBA_ROOT_PREFIX/bin/micromamba" ]; then
    echo "=> Downloading isolated Micromamba..."
    mkdir -p "$BASE_DIR/tmp_mamba"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "$BASE_DIR/tmp_mamba" bin/micromamba
    mkdir -p "$MAMBA_ROOT_PREFIX/bin"
    mv "$BASE_DIR/tmp_mamba/bin/micromamba" "$MAMBA_ROOT_PREFIX/bin/"
    rm -rf "$BASE_DIR/tmp_mamba"
fi

export PATH="$MAMBA_ROOT_PREFIX/bin:$PATH"
eval "$(micromamba shell hook -s posix)"

# 2. Create Python 3.10 environment
ENV_DIR="$BASE_DIR/yolo_env"
if [ ! -d "$ENV_DIR" ]; then
    echo "=> Creating Python 3.10 environment..."
    micromamba create -y -p "$ENV_DIR" python=3.10
fi

# 3. Activate environment
micromamba activate "$ENV_DIR"

# 4. Install dependencies
echo "=> Upgrading pip & installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -e "$BASE_DIR/ultralytics"

# 5. Run the training script directly
echo "=> Starting YOLO training..."
python3 "$BASE_DIR/train_yolo.py" \
    --data "$BASE_DIR/data.yaml" \
    --model "$BASE_DIR/ultralytics/ultralytics/cfg/models/11/eyewave_transformer.yaml" \
    --epochs 50 \
    --batch 32 \
    --imgsz 224

echo "=========================================================="
echo "Job completed at $(date)"
echo "=========================================================="
