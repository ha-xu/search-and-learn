#!/bin/bash

export SEARCHANDLEARN=/home/zx1875/efficientai/search-and-learn

# create and activate conda env
source /home/zx1875/efficientai/miniconda3/etc/profile.d/conda.sh || true
conda activate sal || { echo "activate conda env failed"; exit 3; }

export MODEL=Llama-3.2-1B-Instruct
export APPROACH=beam_search

# Adjust this path if your data is stored elsewhere
export RESULTDIR=/home/zx1875/efficientai/search-and-learn/data/meta-llama/$MODEL/

echo "Running score analysis for MODEL=$MODEL, APPROACH=$APPROACH"

# Iterate over the same n values as in run_beam_search.sh
for n in 4 16 ; do
    INPUT_FILE=$RESULTDIR/${APPROACH}_completions_${n}.jsonl
    OUTPUT_FILE=$RESULTDIR/${APPROACH}_scores_stats_${n}.json
    
    echo "Processing n=$n..."
    if [ -f "$INPUT_FILE" ]; then
        python $SEARCHANDLEARN/print_all_level_score.py "$INPUT_FILE" --output "$OUTPUT_FILE"
    else
        echo "Warning: File not found: $INPUT_FILE"
    fi
done

echo "Job finished."
