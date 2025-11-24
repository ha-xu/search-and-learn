#!/bin/bash
#SBATCH --account=csci_ga_3033_szhang-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G  
#SBATCH --job-name=myjob
#SBATCH --output=/scratch/zx1875/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/zx1875/slurm_logs/%x-%j.err
#SBATCH --chdir=/home/zx1875/efficientai/search-and-learn 

# 1. 创建日志目录 (如果不存在)
mkdir -p /scratch/zx1875/slurm_logs

# 2. 打印任务信息
echo "Job starting on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

nvidia-smi

if [ ! -d "/home/zx1875/efficientai/search-and-learn" ]; then
  echo "ERROR: workdir /home/zx1875/efficientai/search-and-learn not found. Exiting."
  exit 2
fi

source /home/zx1875/efficientai/miniconda3/etc/profile.d/conda.sh || true
conda activate sal || { echo "activate conda env failed"; exit 3; }

git fetch --all --prune
git reset --hard origin/main

huggingface-cli login --token $(cat /home/zx1875/efficientai/huggingface.txt)
# run your script
export CONFIG=recipes/Llama-3.2-1B-Instruct/best_of_n.yaml

python scripts/test_time_compute.py $CONFIG 

echo "Job finished."