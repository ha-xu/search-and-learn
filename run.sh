#!/bin/bash
#SBATCH --account=csci_ga_3033_szhang-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=myjob
#SBATCH --output=log.out

# 下面写你的计算或启动命令
echo "Job started"
bash