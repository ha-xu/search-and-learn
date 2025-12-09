
export SEARCHANDLEARN=/home/zx1875/efficientai/search-and-learn
export EVALDIR=/home/zx1875/efficientai/Qwen2.5-Math/evaluation/

# create and activate conda env
source /home/zx1875/efficientai/miniconda3/etc/profile.d/conda.sh || true
conda activate sal || { echo "activate conda env failed"; exit 3; }

# login to huggingface
huggingface-cli login --token $(cat /home/zx1875/efficientai/huggingface.txt)
# run your script


export MODEL=Llama-3.2-1B-Instruct
export APPROACH=beam_search

export CONFIG=recipes/$MODEL/$APPROACH.yaml
export SEED=0 
export SAMPLES=300

export RESULTDIR=/home/zx1875/efficientai/search-and-learn/data/meta-llama/$MODEL/

export RESULTCOLLECTIONFILE=$RESULTDIR/results_collection_${MODEL}_${APPROACH}_samples_${SAMPLES}.txt

echo "Running with MODEL=$MODEL, APPROACH=$APPROACH, CONFIG=$CONFIG, SEED=$SEED, SAMPLES=$SAMPLES"

# Clear previous results file
echo > $RESULTCOLLECTIONFILE

for n in 4 16 64; do
    cd $SEARCHANDLEARN
    python scripts/test_time_compute.py $CONFIG \
        --n=$n \
        --num_samples=$SAMPLES \
        --seed=$SEED
    
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


echo "job finished."