#!/bin/bash

# zero_shot, few_shot, few_shot_inst, greedy
METHOD="zero_shot"
MAX_TURN=25
BRANCH=3
POLICY_MODEL="Qwen2.5-32B-Instruct"
RESPONSE_MODEL="Qwen2.5-32B-Instruct"
POLICY_TEMPERATURE=0.7
POLICY_TOP_P=0.7
RESPONSE_TEMPERATURE=0.7
RESPONSE_TOP_P=0.7
DATA_PATH="data/dc/test.json"
# Set default values

OUTPUT_PATH="logs/dc/log_${METHOD}_${POLICY_MODEL}.json"
mkdir -p "$(dirname "$OUTPUT_PATH")"
# Run the crime detection evaluator
python3 -m arbench.reasoner.dc.dc_evaluator \
    --method="$METHOD" \
    --data_path="$DATA_PATH" \
    --output_path="$OUTPUT_PATH" \
    --policy_model="$POLICY_MODEL" \
    --response_model="$RESPONSE_MODEL" \
    --branch="$BRANCH" \
    --max_turn="$MAX_TURN" \
    --policy_temperature="$POLICY_TEMPERATURE" \
    --policy_top_p="$POLICY_TOP_P" \
    --response_temperature="$RESPONSE_TEMPERATURE" \
    --response_top_p="$RESPONSE_TOP_P"