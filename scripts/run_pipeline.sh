#!/bin/bash
set -e  # Exit immediately if any command exits with a non-zero status

# Multi-Agent Preference Construction Pipeline

START_STEP=1
MA_MODE=${1:-'GVR'}  # Default to 'GVR', or use command-line argument, e.g., '123check'


MODEL=mistralai/Mistral-7B-Instruct-v0.2    # meta-llama/Llama-3.1-8B-Instruct

PRED_MODEL_ID=Mistral-7B-Instruct-v0.2
if [ "$MA_MODE" = 'GVR' ]; then
    MODEL_GENERATOR=${MODEL}
    MODEL_VERIFIER=${MODEL}
    MODEL_REFINER=${MODEL}
elif [ "$MA_MODE" = '123check' ]; then
    MODEL_EXTRACTOR=${MODEL}
    MODEL_CHECKER=${MODEL}
    MODEL_EXECUTOR=${MODEL}
else
    echo "❌ Invalid MA_MODE: $MA_MODE. Please choose 'GVR' or '123check'."
    exit 1
fi

N_BRANCHING=4
EVAL_STEP="judge_leakage helpfulness"    # "judge_leakage helpfulness"
EVAL_MODEL=mistralai/Mistral-7B-Instruct-v0.2

REWARD_MODEL='lcars'

OUTPUT_ROOT=data_pipeline_MA/predictions
if [ "$MA_MODE" = '123check' ]; then
    OUTPUT_ROOT=data_pipeline_MA/predictions.123check
fi
NAME=${PRED_MODEL_ID}-branch_${N_BRANCHING}
ACTION_PATH=${OUTPUT_ROOT}/${NAME}.json
JUDGMENT_PATH=${OUTPUT_ROOT}/${NAME}-judgment.json
JUDGMENT_VALUE_PATH="${OUTPUT_ROOT}/${NAME}-judgment-with-V.${REWARD_MODEL}.json"
PREF_DATA_PATH_VERIFIER="${OUTPUT_ROOT}/${NAME}-pref_pairs_verifier.${REWARD_MODEL}.json"
PREF_DATA_PATH_REFINER="${OUTPUT_ROOT}/${NAME}-pref_pairs_refiner.${REWARD_MODEL}.json"


DATASET_PATH='./data/main_data_train.json'


PROMPT_TYPE='naive' # or 'privacy_enhanced' or 'naive'
CUDA_VISIBLE_DEVICES=4

# ----- Step 1: Generate Multi-Agent Branching Tree -----
if [ $START_STEP -le 1 ]; then

    echo "🟦 Step 1: Generating multi-agent branching data..."
    echo ACTION_PATH: $ACTION_PATH
    if [ "$MA_MODE" = 'GVR' ]; then
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
        python src/get_final_action_MA.py \
            --input-path $DATASET_PATH \
            --output-path $ACTION_PATH \
            --prompt-type $PROMPT_TYPE \
            --start-index 0 \
            --num-case -1 \
            --n $N_BRANCHING \
            --model-generator ${MODEL_GENERATOR} \
            --model-verifier ${MODEL_VERIFIER} \
            --model-refiner ${MODEL_REFINER} \
            --gpu-num 1 \
            --gpu-memory-utilization 0.6 \
            --output-tree-format nested
    elif [ "$MA_MODE" = '123check' ]; then
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
        python evaluation_MA/get_final_action_MA_123check.py \
            --input-path $DATASET_PATH \
            --output-path $ACTION_PATH \
            --prompt-type $PROMPT_TYPE \
            --start-index 0 \
            --num-case -1 \
            --n $N_BRANCHING \
            --model-extractor ${MODEL_EXTRACTOR} \
            --model-checker ${MODEL_CHECKER} \
            --model-executor ${MODEL_EXECUTOR} \
            --gpu-num 1 \
            --gpu-memory-utilization 0.5 \
            --output-tree-format nested
    else
        echo "❌ Invalid MA_MODE: $MA_MODE. Please choose 'GVR' or '123check'."
        exit 1
    fi
    echo "🟦 Step 1 completed."
    echo ""
fi

# ----- Step 2: Evaluate and Judge Leakage -----
if [ $START_STEP -le 2 ]; then

    echo "🟩 Step 2: Evaluating generated action tree..."
    echo EVAL_STEP: ${EVAL_STEP}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python3 src/evaluate_final_action_MA.py \
      --output-tree-format nested \
      --data-path $DATASET_PATH \
      --action-path ${ACTION_PATH} \
      --eval-step ${EVAL_STEP} \
      --model ${EVAL_MODEL} \
      --output-path ${JUDGMENT_PATH} \
      --gpu-memory-utilization 0.5
    #   --hf-cache-dir $HF_CACHE_DIR
    echo "🟩 Step 2 completed."
    echo ""
fi

# ----- Step 3: Reward Shaping -----
if [ $START_STEP -le 3 ]; then

    echo "🟨 Step 3: Reward assignment and Value iteration..."
    # mkdir -p "$(dirname "$OUTPUT_PATH")"
    python src/value_iteration_MA.py \
        --action-path "$ACTION_PATH" \
        --flat-judgment "$JUDGMENT_PATH" \
        --output-path "$JUDGMENT_VALUE_PATH" \
        --reward-model "$REWARD_MODEL"
    echo "🟨 Step 3 completed."
    echo ""
fi

# ----- Step 4: Generate Preference Pairs -----
if [ $START_STEP -le 4 ]; then

    echo "🟫 Step 4: Generating preference pairs for verifier and refiner..."

    python src/preference_pairs_MA.py \
        --input_path "$JUDGMENT_VALUE_PATH" \
        --reward_model "$REWARD_MODEL" \
        --output_path_V "$PREF_DATA_PATH_VERIFIER" \
        --output_path_R "$PREF_DATA_PATH_REFINER"

    echo "🟫 Step 4 completed."
    echo ""
fi


