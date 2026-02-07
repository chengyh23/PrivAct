#!/bin/bash
set -e  # Exit immediately if any command exits with a non-zero status

TEST_DATA_PATH=data/main_data_test.json

# --------------------------
# Model configs
# --------------------------
version="privacy_then_helpfulness_v8"

MODEL_GENERATOR="mistralai/Mistral-7B-Instruct-v0.2"
MODEL_VERIFIER="outputs/Mistral-7B-Instruct-v0.2-dpo-Mistral-7B-Instruct-v0.2-branch_4-pref_pairs_verifier.$version"
MODEL_REFINER="outputs/Mistral-7B-Instruct-v0.2-dpo-Mistral-7B-Instruct-v0.2-branch_4-pref_pairs_refiner.$version"
MODE="customized"
NAME="auto"


PROMPT_TYPE=naive   # naive or privacy_enhanced

# --------------------------
# Eval settings
# --------------------------
EVAL_MODEL=mistralai/Mistral-7B-Instruct-v0.2
EVAL_STEP=("judge_leakage" "helpfulness")
EVAL_NUM_SAMPLES=10



CUDA_VISIBLE_DEVICES=4
# --------------------------
# Run experiments
# --------------------------
echo "=================================================="
echo "Running test_MA.py #$i:"
echo "GENERATOR: $MODEL_GENERATOR"
echo "VERIFIER:  $MODEL_VERIFIER"
echo "REFINER:   $MODEL_REFINER"
echo "EVAL MODEL: $EVAL_MODEL"
echo "PROMPT TYPE: $PROMPT_TYPE"
echo "=================================================="

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python src/test_MA.py \
    --input-path ${TEST_DATA_PATH} \
    --num 2 \
    --prompt-type ${PROMPT_TYPE} \
    --eval-step "${EVAL_STEP[@]}" \
    --eval-num-samples ${EVAL_NUM_SAMPLES} \
    --model-generator ${MODEL_GENERATOR} \
    --model-verifier ${MODEL_VERIFIER} \
    --model-refiner ${MODEL_REFINER} \
    --custom-output-dirname ${NAME} \
    --eval-model ${EVAL_MODEL} \
    --gpu-memory-utilization 0.6 \
    --gpu-num 1

STATUS=$?
if [[ $STATUS -ne 0 ]]; then
    echo "❌ Experiment #$i FAILED with exit code $STATUS"
else
    echo "✅ Experiment #$i completed successfully"
fi
