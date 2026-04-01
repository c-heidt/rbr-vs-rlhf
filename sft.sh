#!/bin/bash
#SBATCH --job-name=llm-sft
#SBATCH --output=logs/cluster-%j.out
#SBATCH --error=logs/cluster-%j_error.out
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --mail-type=ALL


PROJECT_DIR=$(dirname "$(realpath "$0")")
CONDA_ENV=${CONDA_ENV:-"base"}

echo "Activating conda environment: $CONDA_ENV"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

echo "Load cuda and install flash attention if not already installed"
module load devel/cuda/12.8
pip install flash-attn --no-build-isolation

echo "Running the SFT script"
python "$PROJECT_DIR/sft.py" --project_dir "$PROJECT_DIR"
