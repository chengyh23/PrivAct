import json
import os
import numpy as np
import argparse
from torch.utils.data import dataset
from typing import List
from datasets import load_dataset

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from tree_utils import TreeNode, MultiAgentTree, build_tree_from_flat_data, Forest
from get_final_action_MA import VERIFIER_PROMPT_TMPL, REFINEMENT_PROMPT_TMPL

def cartesian_product(pos_set: List[str], neg_set: List[str], label: str, prompt: str = None):
    r"""
    Return:
        prod (List[dict]):
            "name": str
            "chosen": str
            "rejected": str
    """
    prod = []
    for e1 in pos_set:
        for e2 in neg_set:
            pair = {
                "name": label,
                "chosen": e1,
                "rejected": e2,
                **({"prompt": prompt} if prompt is not None else {})
            }
            prod.append(pair)
    return prod

def collect_preference_pairs_from_tree(tree: MultiAgentTree, threshold=0.5, verbose=True):
    """
    Collect preference pairs for both verifier and refiner roles.

    Returns:
        - verifier_preference_pairs: list of dicts.
        - refiner_preference_pairs: list of dicts.
    """
    
    preference_pairs_V = []
    preference_pairs_R = []

    name, root = tree.name, tree.root
    context = root.prompt
    # VERIFIER PAIRS: for each generator node, consider verifier children
    for nodeG in root.children:
    # for i, nodeG in enumerate(root.children):
        verifiers = [(j, nodeV) for j, nodeV in enumerate(nodeG.children)]
        v_prompt = (
            f"{context}"
            f"{VERIFIER_PROMPT_TMPL.format(GENERATOR_ANSWER=nodeG.output)}" 
        )

        verifier_thresh = threshold

        positive_verifiers_response = [node_j.output for _, node_j in verifiers if node_j.V >= verifier_thresh]
        negative_verifiers_response = [node_j.output for _, node_j in verifiers if node_j.V < verifier_thresh]
        case_pref_pairs_verifier = cartesian_product(positive_verifiers_response, negative_verifiers_response, name, v_prompt)
        preference_pairs_V += case_pref_pairs_verifier
        if verbose:
            print(f'{name}:{nodeG.tree_index}: {len(positive_verifiers_response):<2} x {len(negative_verifiers_response):<2} = {len(case_pref_pairs_verifier):<4}')

        
        # REFINER PAIRS: for each verifier node, consider refiner children
        for nodeV in nodeG.children:
            refiners = [(k, child) for k, child in enumerate(nodeV.children)]
            r_prompt = (
                f"{context}"
                f"{REFINEMENT_PROMPT_TMPL.format(GENERATOR_ANSWER=nodeG.output, VERIFIER_ANSWER=nodeV.output)}"
            )

            refiner_thresh = threshold

            positive_refiner_response = [child.output for _, child in refiners if child.V >= refiner_thresh]
            negative_refiner_response = [child.output for _, child in refiners if child.V < refiner_thresh]
            case_pref_pairs_refiner = cartesian_product(positive_refiner_response, negative_refiner_response, name, r_prompt)
            preference_pairs_R += case_pref_pairs_refiner
            if verbose:
                print(f'{name}:{nodeV.tree_index}: {len(positive_refiner_response):<2} x {len(negative_refiner_response):<2} = {len(case_pref_pairs_refiner):<4}')

            
    return preference_pairs_V, preference_pairs_R


def parse_args():
    parser = argparse.ArgumentParser(description='Generate preference pairs for verifier and refiner training')
    parser.add_argument('--input_path', 
                       required=True,
                       help='Path to the input nested tree JSON file')
    parser.add_argument('--reward_model',
                       type=str,
                       help='Reward model used in value iteration (default: privacy_then_helpfulness)')
    parser.add_argument('--output_path_V', 
                       default='data_pipeline_MA/pref_pairs_verifier.json',
                       help='Path to save verifier preference pairs')
    parser.add_argument('--output_path_R', 
                       default='data_pipeline_MA/pref_pairs_refiner.json',
                       help='Path to save refiner preference pairs')
    parser.add_argument('--threshold', 
                       type=float, 
                       default=0.5,
                       help='Threshold for determining positive vs negative examples')
    
    return parser.parse_args()

def main(verbose: bool = False):
    args = parse_args()
    input_path = args.input_path
    output_path_V = args.output_path_V
    output_path_R = args.output_path_R
    threshold = args.threshold
    
    print(f"Loading forest from: {input_path}")
    forest = Forest.from_dict(input_path)

    datasetV, datasetR = [], []
    test_case_names = []
    print("threshold: ", threshold)
    for tree in forest.trees:
        verifier_pairs, refiner_pairs = collect_preference_pairs_from_tree(tree, threshold, verbose=verbose)
        datasetV += verifier_pairs
        datasetR += refiner_pairs
        if len(verifier_pairs)==0 and len(refiner_pairs)==0:
            test_case_names.append(tree.name)
        if verbose:
            print(f"{tree.name}: {len(verifier_pairs)} verifier & {len(refiner_pairs)} refiner pairs added.")
    
    # Save preference pairs
    if os.path.exists(output_path_V):
        with open(output_path_V, "r", encoding="utf-8") as fin:
            pref_pairs_existing = json.load(fin)
        print(f"⏩ Already exists: {output_path_V}\n(Total # preference pairs: {len(pref_pairs_existing)})")
    else:
        print(f"Saving verifier pairs to: {output_path_V}")
        with open(output_path_V, 'w') as f:
            json.dump(datasetV, f, indent=2)
        print(f"Generated #pairs: verifier {len(datasetV)} pairs; refiner {len(datasetR)} pairs")
    
    if os.path.exists(output_path_R):
        with open(output_path_R, "r", encoding="utf-8") as fin:
            pref_pairs_existing = json.load(fin)
        print(f"⏩ Already exists: {output_path_R}\n(Total # preference pairs: {len(pref_pairs_existing)})")
    else:
        print(f"Saving refiner pairs to: {output_path_R}")
        with open(output_path_R, 'w') as f:
            json.dump(datasetR, f, indent=2)
    

def split_pref_dataset(pref_pairs_path):
    # Split preference pairs dataset. Tried to use load_dataset with streaming=True, but IterableDatasetDict is incompatible with DPOTrainer.
    output_path = pref_pairs_path.strip('.json')
    if os.path.exists(output_path):
        print(f"[split_pref_dataset] Already exists: {output_path}")
    else:
        datasetV = load_dataset('json', data_files=pref_pairs_path, split='train')
        datasetV_split = datasetV.train_test_split(test_size=0.1, seed=42)
        datasetV_split.save_to_disk(output_path)
        print(f"[split_pref_dataset] Saved split dataset to: {output_path}")

if __name__ == '__main__':
    main(verbose=False)
