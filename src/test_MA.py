import logging
import argparse
from tqdm import tqdm
import os
import json
import yaml
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import model_info

from tree_utils import Forest
from get_final_action_MA import collect_trajectories_batch, collect_trajectories_batch_123check
from evaluate_final_action import evaluate_final_actions_batch, calc_eval_metrics, check_action_validity_for_predictions

logger = logging.getLogger(__name__)


def find_latest_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory {output_dir} does not exist!")
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
    if not checkpoints:
        return output_dir
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
    return os.path.join(output_dir, checkpoints[-1])


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True,
                        help='Path of the TEST data in json/txt format.')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Start index of the cases to evaluate.')
    parser.add_argument('--num', type=int, default=1,
                        help='Number of cases to evaluate. If -1, evaluate all remaining cases.')
    parser.add_argument('--specific-case-name', type=str, default=None,
                        help='If not None, only evaluate the case with the given name.')
    parser.add_argument('--prompt-type', type=str,
                        choices=['naive', 'privacy_enhanced', 'conservative', 'reckless'],
                        help='The type of the prompt to use for the agent.')
    parser.add_argument('--gpu-num', type=int, default=1,
                        help='Number of GPUs to use for VLLM.')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                        help='The ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache.')
    parser.add_argument('--hf-cache-dir', type=str,
                        help='The cache directory for the Hugging Face model.')
    parser.add_argument('--mas-mode', type=str, default="GVR",
                        choices=['GVR', '123check'],
                        help='The mode of the multi-agent system to use.')
    parser.add_argument('--model-generator', type=str, required=False,
                        help='The model to use for generating action of Generator in the multi-agent system GVR.')
    parser.add_argument('--model-verifier', type=str, required=False,
                        help='The model to use for generating action of Verifier in the multi-agent system GVR.')
    parser.add_argument('--model-refiner', type=str, required=False,
                        help='The model to use for generating action of Refiner in the multi-agent system GVR.')
    parser.add_argument('--model-extractor', type=str, required=False,
                        help='123check')
    parser.add_argument('--model-checker', type=str, required=False,
                        help='123check')
    parser.add_argument('--model-executor', type=str, required=False,
                        help='123check')
    parser.add_argument("--custom-output-dirname", type=str, default="auto",
                        help="The dir name under outputs_test/.")
    parser.add_argument("--checkpoint_dir", type=str,  
                        help="Path to model outputs/checkpoints")
    parser.add_argument('--eval-model', type=str, required=True,
                        help='The model to use for evaluating action.')
    parser.add_argument("--eval_result", type=str, default=None, help="Path to eval leakage json file (if already evaluated)")
    parser.add_argument("--eval_final_action_path", type=str, default="evaluation/evaluate_final_action.py", help="Evaluation script")
    parser.add_argument("--eval-step", type=str, nargs='+',
                        choices=['extract_secret', 'judge_leakage', 'helpfulness'])
    parser.add_argument("--eval-num-samples", type=int, default=10, help="Number of samples (num branches for the spider graph)")
    parser.add_argument("--pref_data_path", type=str, default="data_pipeline/pref_pairs", help="Eval data path (dir or file)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run evaluation (unused if not re-running eval)")
    return parser.parse_args()


def main():
    args = prepare_args()
    load_dotenv()
    
    log_level = logging.INFO
    logger.setLevel(log_level)

    ourputs_test_dir = "outputs_test/"
        
    if args.mas_mode == "GVR":
        # determine models
        if not os.path.exists(args.model_generator):
            model_name_or_path_G = args.model_generator
        else:
            model_name_or_path_G = find_latest_checkpoint(args.model_generator)

        if not os.path.exists(args.model_verifier):
            model_name_or_path_V = args.model_verifier
        else:
            model_name_or_path_V = find_latest_checkpoint(args.model_verifier)

        if not os.path.exists(args.model_refiner):
            model_name_or_path_R = args.model_refiner
            outputs_test_dir = ourputs_test_dir + args.model_refiner.split("/", 1)[-1]
            if args.prompt_type != "naive":
                outputs_test_dir += f".{args.prompt_type}"
        else:
            model_name_or_path_R = find_latest_checkpoint(args.model_refiner)
            outputs_test_dir = model_name_or_path_R.replace("outputs", "outputs_test")
        if args.custom_output_dirname != "auto":
            outputs_test_dir = "outputs_test/" + args.custom_output_dirname
    
        print(f"Pred using:\n\tG: {model_name_or_path_G},\n\tV: {model_name_or_path_V},\n\tR: {model_name_or_path_R}.")
    elif args.mas_mode.startswith("123check"):
        if not os.path.exists(args.model_extractor):
            model_name_or_path_E = args.model_extractor
        else:
            model_name_or_path_E = find_latest_checkpoint(args.model_extractor)
        
        if not os.path.exists(args.model_checker):
            model_name_or_path_C = args.model_checker
        else:
            model_name_or_path_C = find_latest_checkpoint(args.model_checker)

        if not os.path.exists(args.model_executor):
            model_name_or_path_X = args.model_executor
            outputs_test_dir = f"outputs_test/predictions.{args.mas_mode}/" + args.model_executor.split("/", 1)[-1]
            if args.prompt_type != "naive":
                outputs_test_dir += f".{args.prompt_type}"
        else:
            model_name_or_path_X = find_latest_checkpoint(args.model_executor)
            outputs_test_dir = model_name_or_path_X.replace("outputs", "outputs_test")
        if args.custom_output_dirname != "auto":
            outputs_test_dir = f"outputs_test/predictions.{args.mas_mode}/" + args.custom_output_dirname
            
        print(f"Pred using:\n\tE: {model_name_or_path_E},\n\tC: {model_name_or_path_C},\n\tX: {model_name_or_path_X}.")
    else:
        raise NotImplementedError(f"Unsupported MAS mode: {args.mas_mode}.")
    
    print(f"Output to {outputs_test_dir}")
    print(f"Eval using: {args.eval_model}")
    
    os.makedirs(outputs_test_dir, exist_ok=True)
    output_dir = os.path.join(outputs_test_dir, "predictions")
    os.makedirs(output_dir, exist_ok=True)
    # Save config to outputs_test/
    if args.mas_mode == "GVR":
        config = {
            "G": model_name_or_path_G,
            "V": model_name_or_path_V,
            "R": model_name_or_path_R,
        }
    elif args.mas_mode.startswith("123check"):
        config = {
            "E": model_name_or_path_E,
            "C": model_name_or_path_C,
            "X": model_name_or_path_X,
        }
    else:
        raise NotImplementedError(f"Unsupported MAS mode: {args.mas_mode}.")
    config["prompt_type"] = args.prompt_type
    config["eval_model"] = args.eval_model
    with open(os.path.join(output_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    #########################
    # Get final action
    #########################
    output_path_actions = os.path.join(output_dir, f"actions.n{args.eval_num_samples}.json")    
    if not os.path.exists(output_path_actions):
        if args.mas_mode == "GVR":
            result = collect_trajectories_batch(
                args, 
                model_generator=model_name_or_path_G,
                model_verifier=model_name_or_path_V,
                model_refiner=model_name_or_path_R,
                num_case=args.num, 
                n_branching=[args.eval_num_samples, 1, 1]
            )
        elif args.mas_mode.startswith("123check"):
            result = collect_trajectories_batch_123check(
                args, 
                model_extractor=model_name_or_path_E,
                model_checker=model_name_or_path_C,
                model_executor=model_name_or_path_X,
                num_case=args.num, 
                n_branching=[args.eval_num_samples, 1, 1],
            )
        else:
            raise NotImplementedError(f"Unsupported MAS mode: {args.mas_mode}.")
        os.makedirs(output_dir, exist_ok=True)
        result.to_dict(output_path_actions)
        print(f"Successfully wrote actions to {output_path_actions}")
    else:
        logger.info(f"{output_path_actions} exists, skipping collecting trajectories and loading existing results.")
        print(f"⏩ {output_path_actions} exists, skipping collecting trajectories and loading existing results.")
    
    #########################
    # Evaluate final action
    #########################
    output_path_eval = os.path.join(output_dir, f"evaluations.n{args.eval_num_samples}.json")
    
    if not os.path.exists(output_path_eval):
        forest = Forest.from_dict(output_path_actions)
        actions = []
        for i_case in tqdm(range(len(forest)), desc="loading actions of all cases", leave=False):
            leaves = forest[i_case].get_all_leaves()
            for i in range(len(leaves)):
                _action = {
                    'name': forest[i_case].name, 
                    'final_action': leaves[i].output,
                    'tree_index': leaves[i].tree_index,
                }
                actions.append(_action)
                
        name_to_result = evaluate_final_actions_batch(
            args, 
            model_name_or_path=args.eval_model, 
            data_path=args.input_path, 
            actions=actions,
            step=args.eval_step
        )

        with open(output_path_eval, 'w') as f:
            json.dump(name_to_result, f, indent=4)
    else:
        print(f"⏩ {output_path_eval} exists, skipping evaluate_final_actions and loading existing results.")
    
    calc_eval_metrics(output_path_eval, args.eval_step, verbose=False, filter_invalid=True)
    check_action_validity_for_predictions(output_path_eval)

if __name__ == "__main__":
    main()