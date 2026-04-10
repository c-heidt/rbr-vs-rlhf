#!/bin/bash
#SBATCH --job-name=llm-sft
#SBATCH --output=logs/lm_sft-%j.out
#SBATCH --error=logs/lm_sft-%j_error.out
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=ALL


PROJECT_DIR="$SLURM_SUBMIT_DIR"
OUTPUT_DIR="${WORKSPACE:-$SCRATCH}"
CONDA_ENV=${CONDA_ENV:-"base"}

echo "Activating conda environment: $CONDA_ENV"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

echo "Load cuda module and install flash attention"
module load devel/cuda/12.8
pip install flash-attn@https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

mkdir -p "$OUTPUT_DIR"

echo "Running the SFT script"
echo "  project_dir : $PROJECT_DIR"
echo "  output_dir  : $OUTPUT_DIR"
python "$PROJECT_DIR/sft.py" --output_dir "$OUTPUT_DIR"
