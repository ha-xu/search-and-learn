
export SEARCHANDLEARN=/home/zx1875/efficientai/search-and-learn
export EVALDIR=/home/zx1875/efficientai/Qwen2.5-Math/evaluation/


# if [ ! -d "$SEARCHANDLEARN" ]; then
#   echo "ERROR: workdir $SEARCHANDLEARN not found. Exiting."
#   exit 2
# fi

# cd $SEARCHANDLEARN
# # update the code
# git fetch --all --prune
# git reset --hard origin/main

# create and activate conda env
source /home/zx1875/efficientai/miniconda3/etc/profile.d/conda.sh || true
conda activate sal || { echo "activate conda env failed"; exit 3; }

# login to huggingface
huggingface-cli login --token $(cat /home/zx1875/efficientai/huggingface.txt)
# run your script

export MODEL=Llama-3.2-1B-Instruct
export APPROACH=dvts

export CONFIG=recipes/$MODEL/$APPROACH.yaml
export SEED=0 
export SAMPLES=100

export RESULTDIR=/home/zx1875/efficientai/search-and-learn/data/meta-llama/$MODEL
export RESULTCOLLECTIONFILE=$RESULTDIR/results_collection_${MODEL}_${APPROACH}_samples_${SAMPLES}.txt


echo "Running with MODEL=$MODEL, APPROACH=$APPROACH, CONFIG=$CONFIG, SEED=$SEED, SAMPLES=$SAMPLES"

# Clear previous results file
echo > $RESULTCOLLECTIONFILE

for n in 64; do
    cd $SEARCHANDLEARN
    python scripts/test_time_compute.py $CONFIG \
        --n=$n \
        --num_samples=$SAMPLES \
        --seed=$SEED \
        --prm_batch_size=1 \
        --search_batch_size=1
    
    echo "Evaluation results for CONFIG=$CONFIG, n=$n, seed=$SEED, samples=$SAMPLES" >> $RESULTCOLLECTIONFILE

    # echo $RESULTDIR/beam_search_completions.jsonl

    # Evaluation of the accuracy
    cd $EVALDIR
    conda create -n qwen-math python=3.11 && conda activate qwen-math
    cd latex2sympy
    pip install -e .
    cd ..
    pip install -r requirements.txt 
    python evaluate.py --file_path $RESULTDIR/${APPROACH}_completions_${n}.jsonl >> $RESULTCOLLECTIONFILE
    conda deactivate
    # print time
    python $SEARCHANDLEARN/staticalprint.py $RESULTDIR/${APPROACH}_completions_${n}.jsonl >> $RESULTCOLLECTIONFILE

done


