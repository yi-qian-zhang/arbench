CUDA_VISIBLE_DEVICES=0 vllm serve /data2/chenxin/model_sft_results/distill-qwen-7b/full/distill_r1_coing_neo_cleaned_uncertainty_threshold_40_sft_conversation_train_dataset \
    --api-key 123456  \
    --port 8722 \
    --gpu-memory-utilization 0.9 \
    --served-model-name distill_r1_coing_neo_cleaned_uncertainty_threshold_40_sft_conversation_train_dataset \
    --trust-remote-code

CUDA_VISIBLE_DEVICES=7 vllm serve /data1/HF-Models/Qwen/Qwen2.5-7B-Instruct \
    --api-key 123456 \
    --port 8725 \
    --gpu-memory-utilization 0.9 \
    --served-model-name Qwen2.5-7B-Instruct \
    --trust-remote-code


python3 -m arbench.reasoner.dc.dc_evaluator \
    --method="zero_shot" \
    --data_path="data/dc/test.json" \
    --output_path="./results/distill_r1_coing_neo_cleaned_uncertainty_threshold_40_sft_conversation_train_dataset.json" \
    --policy_model="distill_r1_coing_neo_cleaned_uncertainty_threshold_40_sft_conversation_train_dataset" \
    --response_model="gpt-4o" \
    --branch=3 \
    --max_turn=25 \
    --policy_temperature=0.7 \
    --policy_top_p=0.7 \
    --response_temperature=0.7 \
    --response_top_p=0.7

python3 -m arbench.reasoner.dc.dc_evaluator \
    --method="zero_shot" \
    --data_path="data/dc/test.json" \
    --output_path="./results/Qwen2.5-7B-Instruct.json" \
    --policy_model="Qwen2.5-7B-Instruct" \
    --response_model="gpt-4o" \
    --branch=3 \
    --max_turn=25 \
    --policy_temperature=0.7 \
    --policy_top_p=0.7 \
    --response_temperature=0.7 \
    --response_top_p=0.7

python -m arbench.reasoner.dc.dc_evaluator \
    --method "zero_shot" \
    --data_path data/dc/test.json \
    --output_path ./results/distill_zero_shot.json \
    --policy_model distill_r1_coing_neo_cleaned_uncertainty_threshold_40_sft_conversation_train_dataset \
    --response_model gpt-4o \
    --branch=3 \
    --max_turn=25 \
    --policy_temperature=0.7 \
    --policy_top_p=0.7 \
    --response_temperature=0.7 \
    --response_top_p=0.7

python -m arbench.reasoner.dc.dc_evaluator \
    --method zero_shot \
    --data_path data/dc/test.json \
    --output_path ./results/dc_interactive_zero_shot.json \
    --policy_model "distill_r1_coing_neo_cleaned_uncertainty_threshold_40_sft_conversation_train_dataset" \
    --response_model "gpt-4o" \
    --max_turn 25 \
    --policy_temperature 0.6 \
    --policy_top_p 0.95 \
    --response_temperature 0.7 \
    --response_top_p 0.7 \
    --enable_interaction true
