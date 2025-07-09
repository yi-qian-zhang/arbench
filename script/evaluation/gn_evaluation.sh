#!/bin/bash

# Set default values
DATA_PATH="data/gn/test.json"
MODEL="Qwen2.5-32B-Instruct"
# zero_shot, few_shot, few_shot_inst, greedy
METHOD="greedy"
MAX_TURN=25

OUTPUT_PATH="logs/gn/log_${METHOD}_${MODEL}.json"
# Create output directory if not exists
mkdir -p "$(dirname "$OUTPUT_PATH")"

python3 -m arbench.reasoner.gn.gn_evaluator \
    --model="$MODEL" \
    --method="$METHOD" \
    --data_path="$DATA_PATH" \
    --output_path="$OUTPUT_PATH" \
    --max_turn="$MAX_TURN"
