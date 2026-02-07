"""Evaluate the final actions of the LM agents by checking if the final actions fulfill the user instructions and
if the final actions leak sensitive information."""
import argparse
import json
import os
import random
import sys
from typing import List

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer

from langchain_community.llms import VLLM

from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from evaluate_final_action import evaluate_final_actions_batch, prepare_prompt_to_identify_sensitive_information, prepare_prompt_to_identify_leakage, prepare_prompt_to_evaluate_helpfulness
from evaluate_final_action import post_process_extracted_secrets, parse_leakage_judgment, parse_helpfulness_score
from evaluate_final_action import evaluate_final_actions
from evaluate_final_action import calc_eval_metrics, check_action_validity_for_predictions
from tree_utils import Forest



def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-tree-format', type=str, choices=['nested','flat'])
    parser.add_argument('--data-path', type=str, help='Path of the evaluation data in json format.')
    parser.add_argument('--action-path', type=str, help='Path of the LM agent final actions.')
    parser.add_argument('--eval-step', nargs='+', type=str,
                        choices=['extract_secret', 'judge_leakage', 'helpfulness'])
    parser.add_argument('--output-path', type=str, required=True, help='Path to save the judgments.')
    parser.add_argument('--helpfulness-score-path', type=str,
                        help='Path that saves the helpfulness scores. If provided, can compute the adjusted leakage '
                             'rate when "step" is "judge_leakage".')
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--gpu-num', type=int, default=1,
                        help='Number of GPUs to use.')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                        help='The ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache.')
    parser.add_argument('--hf-cache-dir', type=str,
                        help='The cache directory for the Hugging Face model.')

    return parser.parse_args()

def main():
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = prepare_args()
    load_dotenv()

    if os.path.exists(args.output_path):
        print(f"⏩ Skipping evaluation: {args.output_path} already exists. ")
        calc_eval_metrics(args.output_path, eval_step=args.eval_step, verbose=False)
        return

    # Load actions from JSON file
    if args.output_tree_format == 'nested':
        # forest = load_forest_from_json(args.action_path)
        forest = Forest.from_dict(args.action_path)
    elif args.output_tree_format == 'flat':
        with open(args.action_path, 'r') as f:
            actions = json.load(f)
    else:
        raise ValueError(f"Unsupported output_tree_format: {args.output_tree_format}")
    
    actions = []
    for i_case in tqdm(range(len(forest)), desc="loading actions of all cases", leave=False):
        # assert forest[i_case].name == f'main{i_case+1}', f"{forest[i_case].name} != main{i_case+1}"
        leaves = forest[i_case].get_all_leaves()
        for i in range(len(leaves)):
            _action = {
                'name': forest[i_case].name, 
                # 'pred_model': args.model, 
                'final_action': leaves[i].output,
                'tree_index': leaves[i].tree_index,
            }
            actions.append(_action)
    
    # Eval actions
    name_and_result = evaluate_final_actions_batch(
    # name_and_result = evaluate_final_actions(
        args, 
        args.model, 
        args.data_path, 
        actions, 
        args.eval_step
    )
    
    # Save
    with open(args.output_path, 'w') as f:
        json.dump(name_and_result, f, indent=4)
        print("Evaluation complete. Results written to", args.output_path)

    calc_eval_metrics(args.output_path, eval_step=args.eval_step, verbose=False)

if __name__ == '__main__':
    main()

