#!/bin/bash
#SBATCH --account=csci_ga_3033_szhang-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G  
#SBATCH --job-name=search_and_learn
#SBATCH --output=/scratch/zx1875/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/zx1875/slurm_logs/%x-%j.err
#SBATCH --chdir=/home/zx1875/efficientai/search-and-learn

# 1. 创建日志目录 (如果不存在)
mkdir -p /scratch/zx1875/slurm_logs

# 2. 打印任务信息
echo "Job starting on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

nvidia-smi

export SEARCHANDLEARN=/home/zx1875/efficientai/search-and-learn
# export RESULTDIR=/home/zx1875/efficientai/search-and-learn/data/meta-llama/$MODEL/
# export EVALDIR=/home/zx1875/efficientai/Qwen2.5-Math/evaluation/


if [ ! -d "$SEARCHANDLEARN" ]; then
  echo "ERROR: workdir $SEARCHANDLEARN not found. Exiting."
  exit 2
fi

cd $SEARCHANDLEARN
# update the code
git fetch --all --prune
git reset --hard origin/main

bash run.sh

echo "job finished."