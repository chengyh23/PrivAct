#!/bin/bash


# Multi-agent
if [ $# -lt 1 ]; then
    echo "Usage: $0 <ROLE>"
    exit 1
fi

PARAM_ROLE="$1"
if [[ "${PARAM_ROLE}" == "verifier" || "${PARAM_ROLE}" == "refiner" ]]; then
    ROLES=("${PARAM_ROLE}")
    echo "Training for ROLE=${PARAM_ROLE}"
elif [[ "${PARAM_ROLE}" == "both" ]]; then
    ROLES=("verifier" "refiner")
    echo "Training for both roles: verifier and refiner"
else
    echo "Error: ROLE must be either 'verifier' / 'refiner' / 'both'."
    exit 1
fi

REWARD_MODEL='lcars'

for ROLE in "${ROLES[@]}"; do

    MODEL_NAME_OR_PATH=mistralai/Mistral-7B-Instruct-v0.2   # meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen3-4B-Instruct-2507
    DATA_PATH=data_pipeline_MA/predictions/Mistral-7B-Instruct-v0.2-branch_4-pref_pairs_${ROLE}.${REWARD_MODEL}.json    
    MODEL_ID=Mistral-7B-Instruct-v0.2-dpo-Mistral-7B-Instruct-v0.2-branch_4-pref_pairs_${ROLE}.${REWARD_MODEL}

    OUTPUT_DIR_ROOT=outputs
    echo "=================================================="
    echo "Starting training for ROLE=${ROLE} with REWARD_MODEL=${REWARD_MODEL}"
    echo "Model: ${MODEL_NAME_OR_PATH}"
    echo "Data path: ${DATA_PATH}"
    echo "Output directory: ${OUTPUT_DIR_ROOT}/${MODEL_ID}"
    echo "=================================================="
    if [[ "${ROLE}" == "verifier" ]]; then
        NUM_TRAIN_EPOCHS=12
    elif [[ "${ROLE}" == "refiner" ]]; then
        NUM_TRAIN_EPOCHS=4
    fi
    ACCELERATE_LOG_LEVEL=info accelerate launch --gpu_ids 4 --config_file configs/deepspeed/zero3_cpu.yaml --mixed_precision bf16 \
        --num_processes 1 \
        src/train.py \
        --do_train \
        --eval_strategy 'steps' \
        --eval_steps 50 \
        --config configs/trl/config_full.yaml \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --data_path ${DATA_PATH} \
        --per_device_train_batch_size=8 \
        --gradient_accumulation_steps=4 \
        --torch_dtype=bfloat16 \
        --bf16=True \
        --beta=0.4 \
        --num_train_epochs=${NUM_TRAIN_EPOCHS} \
        --dataloader_num_workers 1 \
        --save_strategy='steps' \
        --save_steps=50 \
        --metric_for_best_model eval_loss \
        --save_total_limit=1 \
        --output_dir=${OUTPUT_DIR_ROOT}/${MODEL_ID} \
        --hub_model_id=${MODEL_ID} \
        # --prompt=qwen2-boxed
        # --do_eval \
        # --num_train_epochs=4 \
done
