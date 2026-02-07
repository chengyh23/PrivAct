import os
import json
from typing import List, Dict
import numpy as np
import math

from tree_utils import TreeNode, MultiAgentTree, Forest



def reward_lcars(eval_result, b1=0.0, b2=0.45, leak_power=0.5, utility_power=2.0):
    """ Leakage-Conditioned Asymmetric Reward Shaping
    """
    U = eval_result['helpfulness_score'] / 3.0 # helpfulness_score in [0,1,2,3]
    
    if eval_result['leak_info'] is True:
        secret_judgment = eval_result['secret_judgment']
        n_leaked = sum([1 for item in secret_judgment if item[1]==True])
        leaked_frac = n_leaked / len(secret_judgment)
        leak_penalty = - leaked_frac ** leak_power
        utility_penalty = - U ** utility_power
        R = b1 + leak_penalty + utility_penalty
        return max(R, -1.0)
        
    # --- at this point, privacy is preserved ---
    bonus = U ** utility_power  # [0,1]
    R = b2 + (1 - b2) * bonus
    return R


def assign_reward_to_leaves(flat_judgments: List[Dict], forest: Forest, reward_model_name: str):
    """
    in-place modification
    """

    for judgment in flat_judgments:
        name = judgment['name']
        pos = judgment['tree_index']
        tree = forest.get_tree_by_name(name)
        node = tree.get_node_by_index(pos)
        node.V = reward_lcars(judgment['eval_result']) # TODO need to encapsulate or not?
        assert node.V is not None, f"Failed to assign reward to leaf node {name} at position {pos}"


def propagate_expected_value(node: TreeNode):
    """ Reward Propagation
    Recursively computes the expected value V for all nodes in the tree.
    Leaf nodes take their reward as V; non-leaf nodes take mean of children's V.
    """
    if not node.children:
        # Leaf node: already has V set (should be reward from leaf trajectory)
        return node.V
    child_vs = [propagate_expected_value(child) for child in node.children]
    if not child_vs:
        raise ValueError(f"Tree node '{node.type}' has no children; cannot compute mean.")
    # print(f"Child vs: {child_vs} node: {node.type}")
    node.V = float(np.mean(child_vs))
    return node.V


def main(args):
    if os.path.exists(args.output_path):
        print(f"⏩ Output file {args.output_path} already exists; skipping value iteration.")
        return
    # Reward assignment at leaves
    forest = Forest.from_dict(args.action_path)
    with open(args.flat_judgment, "r") as f:
        flat_judgments = json.load(f)
    assign_reward_to_leaves(flat_judgments, forest, reward_model_name=args.reward_model)
    
    # Value Iteration
    for tree in forest.trees:
        propagate_expected_value(tree.root)
    
    forest.to_dict(args.output_path, verbose=True)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add reward values to nested tree structure")
    parser.add_argument("--action-path", required=True, help="Path to action (nested tree) JSON file")
    parser.add_argument("--flat-judgment", required=True, help="Path to flat judgment JSON file")
    parser.add_argument("--output-path", required=True, help="Path to output JSON file")
    parser.add_argument("--reward-model", type=str, default="lcars", help="Reward model to use for computing rewards (default: lcars)")
    args = parser.parse_args()

    main(args)
