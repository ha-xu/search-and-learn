#!/bin/bash
#SBATCH --account=csci_ga_3033_szhang-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G  
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
export SEED=0 

export SEARCHANDLEARN=/home/zx1875/efficientai/search-and-learn
export RESULTDIR=/home/zx1875/efficientai/search-and-learn/data/meta-llama/Llama-3.2-1B-Instruct/
export EVALDIR=/home/zx1875/efficientai/Qwen2.5-Math/evaluation/
for n in 4 16 64 256; do
    cd $SEARCHANDLEARN
    python scripts/test_time_compute.py $CONFIG \
        --n=$n \
        --num_samples=500 \
        --seed=$SEED
    
    echo "Evaluation results for CONFIG=$CONFIG, n=$n, seed=$SEED" >> $RESULTDIR/results_Llama-3.2-1B-Instruct_best_of_n.txt

    # echo $RESULTDIR/best_of_n_completions.jsonl

    # Evaluation of the accuracy
    cd $EVALDIR
    conda create -n qwen-math python=3.11 && conda activate qwen-math
    cd latex2sympy
    pip install -e .
    cd ..
    pip install -r requirements.txt 
    python evaluate.py --file_path $RESULTDIR/best_of_n_completions_${n}.jsonl >> $RESULTDIR/results_Llama-3.2-1B-Instruct_best_of_n.txt
    conda deactivate
    # print time
    python $SEARCHANDLEARN/staticalprint.py $RESULTDIR/best_of_n_completions_${n}.jsonl >> $RESULTDIR/results_Llama-3.2-1B-Instruct_best_of_n.txt

done


echo "job finished."