#!/bin/bash
#SBATCH --account=csci_ga_3033_szhang-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G  
#SBATCH --job-name=myjob
#SBATCH --output=/scratch/zx1875/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/zx1875/slurm_logs/%x-%j.err

# 1. 创建日志目录 (如果不存在)
mkdir -p /scratch/zx1875/slurm_logs

# 2. 打印任务信息
echo "Job starting on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

nvidia-smi

# 下面写你的计算或启动命令
echo "Job started"
cd /home/zx1875/EfficientAI/search_and_learn
git pull 
# activate conda environment
conda activate sal

# run your script
export CONFIG=recipes/Llama-3.2-1B-Instruct/best_of_n.yaml

python scripts/test_time_compute.py $CONFIG --push_to_hub=true
