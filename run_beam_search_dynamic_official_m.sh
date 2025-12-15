export SEARCHANDLEARN=/home/zx1875/efficientai/search-and-learn
export EVALDIR=/home/zx1875/efficientai/Qwen2.5-Math/evaluation/

# create and activate conda env
source /home/zx1875/efficientai/miniconda3/etc/profile.d/conda.sh || true
conda activate sal || { echo "activate conda env failed"; exit 3; }

# login to huggingface
huggingface-cli login --token $(cat /home/zx1875/efficientai/huggingface.txt)
# run your script


export MODEL=Llama-3.2-3B-Instruct
export APPROACH=beam_search_dynamic_official

export CONFIG=recipes/$MODEL/$APPROACH.yaml
export SEED=0 
export SAMPLES=500
export RESULTDIR=/home/zx1875/efficientai/search-and-learn/data/meta-llama/$MODEL

export RESULTCOLLECTIONFILE=$RESULTDIR/results_collection_${MODEL}_${APPROACH}_samples_${SAMPLES}.txt


echo "Running with MODEL=$MODEL, APPROACH=$APPROACH, CONFIG=$CONFIG, SEED=$SEED, SAMPLES=$SAMPLES"

# Clear previous results file
# echo > $RESULTCOLLECTIONFILE
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export VLLM_USE_CUDA_GRAPH=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

IMAGE_PATH=/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif

for n in 4 16; do
    for beam_decay_temp in 1; do
        cd $SEARCHANDLEARN
        singularity exec --nv $IMAGE_PATH python scripts/test_time_compute.py $CONFIG \
                --n=$n \
                --num_samples=$SAMPLES \
                --seed=$SEED \
                --beam_decay_temperature=$beam_decay_temp
            
            echo "Evaluation results for CONFIG=$CONFIG, n=$n, beam_decay_temperature=$beam_decay_temp, seed=$SEED, samples=$SAMPLES" >> $RESULTCOLLECTIONFILE

            # echo $RESULTDIR/beam_search_completions.jsonl

            # Evaluation of the accuracy
            cd $EVALDIR
            conda create -n qwen-math python=3.11 && conda activate qwen-math
            cd latex2sympy
            pip install -e .
            cd ..
            pip install -r requirements.txt 
            python evaluate.py --file_path $RESULTDIR/${APPROACH}_completions_${n}_${MODEL}_${SAMPLES}.jsonl >> $RESULTCOLLECTIONFILE
            conda deactivate
            # print time
            python $SEARCHANDLEARN/staticalprint.py $RESULTDIR/${APPROACH}_completions_${n}_${MODEL}_${SAMPLES}.jsonl >> $RESULTCOLLECTIONFILE

        done
        # cd $SEARCHANDLEARN
        # python scripts/test_time_compute.py $CONFIG \
        #     --n=$n \
        #     --num_samples=$SAMPLES \
        #     --seed=$SEED
        
        # echo "Evaluation results for CONFIG=$CONFIG, n=$n, seed=$SEED, samples=$SAMPLES" >> $RESULTCOLLECTIONFILE

        # # echo $RESULTDIR/beam_search_completions.jsonl

        # # Evaluation of the accuracy
        # cd $EVALDIR
        # conda create -n qwen-math python=3.11 && conda activate qwen-math
        # cd latex2sympy
        # pip install -e .
        # cd ..
        # pip install -r requirements.txt 
        # python evaluate.py --file_path $RESULTDIR/${APPROACH}_completions_${n}.jsonl >> $RESULTCOLLECTIONFILE
        # conda deactivate
        # # print time
        # python $SEARCHANDLEARN/staticalprint.py $RESULTDIR/${APPROACH}_completions_${n}.jsonl >> $RESULTCOLLECTIONFILE

    done


echo "job finished."