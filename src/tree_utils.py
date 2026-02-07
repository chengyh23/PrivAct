"""
tree_utils.py

This module provides utilities for representing and processing multi-agent decision trees
in evaluation workflows (such as generator-verifier-refiner trees). It defines tree node
structures, building and traversing utilities, and conversion to/from JSON for storing and
loading such trees. The utilities facilitate converting flat prediction outputs to hierarchical
tree formats, obtaining leaf outputs or node paths for analysis, and working with nested
multi-agent reasoning results.

Key classes and functions:
- TreeNode: Represents individual nodes (generator, verifier, refiner).
- MultiAgentTree: Represents an entire case tree (with methods for leaf and path utilities).
- build_tree_from_flat_data: Builds a tree from a flat list of outputs.
- save_tree_to_json / load_tree_from_json: Store/load trees as JSON files.
- convert_flat_to_nested_tree: Tool for converting flat outputs to nested tree structure.
- analyze_tree_structure: Utility for printing tree statistics (leaf counts, depths, etc).

Typical use cases:
- Transforming step-by-step model outputs into a navigable tree for structured analysis.
- Inspecting and traversing multi-agent reasoning branches.
- Converting or analyzing batch evaluation outputs across multiple problem instances.
"""


import json
from typing import Dict, List, Any, Optional

class TreeNode:
    """Represents a node in the multi-agent tree structure."""
    
    def __init__(self, node_type: str, output: str, tree_index: List[int], V: float = None, prompt: str = None):
        self.type = node_type  # 'generator', 'verifier', 'refiner'; 'extractor', 'checker', 'executor'
        self.output = output
        self.tree_index = tree_index  # [i, j, k] for generator, verifier, refiner
        self.V = V
        self.prompt = prompt
        self.children: List['TreeNode'] = []
        self.parent: Optional['TreeNode'] = None
    
    def __repr__(self):
        return f"TreeNode(type={self.type}, tree_index={self.tree_index}, V={self.V})"

    def add_child(self, child: 'TreeNode'):
        """Add a child node and set parent relationship."""
        child.parent = self
        self.children.append(child)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        result = {
            'type': self.type,
            'prompt': self.prompt,
            'output': self.output,
            'tree_index': self.tree_index,
            'V': self.V
        }
        if self.children:
            result['children'] = [child.to_dict() for child in self.children]
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TreeNode':
        """Recursively create TreeNode from dictionary."""
        node = cls(
            node_type=data['type'],
            output=data['output'],
            tree_index=data['tree_index'],
            V=data['V'] if 'V' in data else None,
            prompt=data['prompt'] if 'prompt' in data else None,
        )
        if 'children' in data:
            for child_data in data['children']:
                child_node = cls.from_dict(child_data)
                node.add_child(child_node)
        return node
    
    def get_path_to_root(self) -> List['TreeNode']:
        """Get the path from this node to the root."""
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return path[::-1]  # Reverse to get root -> leaf order
    
    def get_all_leaves(self) -> List['TreeNode']:
        """Get all leaf nodes in the subtree rooted at this node."""
        if not self.children:
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_all_leaves())
        return leaves
    
    def get_nodes_at_level(self, level: int) -> List['TreeNode']:
        """Get all nodes at a specific level (0=root, 1=children, 2=grandchildren, etc)."""
        if level == 0:
            return [self]
        if level < 0:
            return []
        nodes_at_level = []
        for child in self.children:
            nodes_at_level.extend(child.get_nodes_at_level(level - 1))
        return nodes_at_level
    
    def get_node_by_index(self, tree_index: List[int]) -> Optional['TreeNode']:
        """Find a node by its tree index."""
        if self.tree_index == tree_index:
            return self
        for child in self.children:
            result = child.get_node_by_index(tree_index)
            if result is not None:
                return result
        return None


class MultiAgentTree:
    """Represents the complete multi-agent tree structure."""
    
    def __init__(self, name: str, root: TreeNode):
        self.name = name
        self.root = root
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tree to dictionary representation."""
        return {
            'name': self.name,
            'tree': self.root.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MultiAgentTree':
        """Create MultiAgentTree from dictionary."""
        return cls(
            name=data['name'],
            root=TreeNode.from_dict(data['tree'])
        )
    
    def get_all_leaves(self) -> List[TreeNode]:
        """Get all leaf nodes in the tree."""
        return self.root.get_all_leaves()
    
    def get_node_by_index(self, index: List[int]) -> TreeNode:
        # TODO redundant with TreeNode.get_node_by_index
        return self.root.get_node_by_index(index)

    def get_path_to_leaf(self, leaf_index: List[int]) -> List[TreeNode]:
        """Get the path from root to a specific leaf node."""
        leaf_node = self.root.get_node_by_index(leaf_index)
        if leaf_node is None:
            raise ValueError(f"Leaf node with index {leaf_index} not found")
        return leaf_node.get_path_to_root()
    
    def get_leaf_outputs(self) -> List[Dict[str, Any]]:
        """Get all leaf node outputs with their full paths."""
        leaves = self.get_all_leaves()
        results = []
        
        for leaf in leaves:
            path = leaf.get_path_to_root()
            result = {
                'tree_index': leaf.tree_index,
                'path': [
                    {
                        'type': node.type,
                        'output': node.output,
                        'tree_index': node.tree_index
                    }
                    for node in path
                ],
                'generator_output': path[0].output if len(path) > 0 else None,
                'verifier_output': path[1].output if len(path) > 1 else None,
                'refiner_output': path[2].output if len(path) > 2 else None
            }
            results.append(result)
        
        return results

class Forest:
    def __init__(self, trees: List[MultiAgentTree] = None):
        """
        Params:
            filepath (str):
                List of Tree
        """
        self.trees = trees

    @classmethod
    def from_dict(cls, filepath: str) -> 'Forest':
        """Load forest from JSON file."""
        assert filepath is not None
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(
            trees=[MultiAgentTree.from_dict(tree_data) for tree_data in data]
        )
    def __len__(self):
        """Return the number of trees in the forest."""
        return len(self.trees) if self.trees is not None else 0
        
    def __getitem__(self, idx):
        """Allows indexing into the Forest to get a tree by integer index."""
        if self.trees is None:
            raise IndexError("Forest is empty.")
        return self.trees[idx]

    def to_dict(self, filepath: str, verbose: bool =False):
    # def save_forest_to_json(forest: List[MultiAgentTree], filepath: str):
        """Save a forest (list of MultiAgentTree) to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([tree.to_dict() for tree in self.trees], f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"Saved forest with {len(self.trees)} trees to {filepath}")

    def get_tree_by_name(self, name: str):        
        for tree in self.trees:
            if tree.name == name:
                return tree
        raise ValueError(f"No tree named {name} in the given forest")
        return None



def build_tree_from_flat_data(flat_data: List[Dict[str, Any]]) -> MultiAgentTree:
    """Build tree structure from flat data format (current implementation).
    used to be: construct_depth_3_tree_from_trajectories
    """
    if not flat_data:
        raise ValueError("No data provided")
    
    name = flat_data[0]['name']
    
    # Group by generator index (first level)
    generator_groups = {}
    for item in flat_data:
        gen_idx = item['tree_position'][0]
        if gen_idx not in generator_groups:
            generator_groups[gen_idx] = []
        generator_groups[gen_idx].append(item)
    
    # Create root node (dummy root to hold multiple generator branches)
    root = TreeNode("root", "", [])
    
    for gen_idx, gen_items in generator_groups.items():
        # Get generator output (should be same for all items in this group)
        generator_output = gen_items[0]['generator_output']
        generator_node = TreeNode("generator", generator_output, [gen_idx])
        root.add_child(generator_node)
        
        # Group by verifier index (second level)
        verifier_groups = {}
        for item in gen_items:
            ver_idx = item['tree_position'][1]
            if ver_idx not in verifier_groups:
                verifier_groups[ver_idx] = []
            verifier_groups[ver_idx].append(item)
        
        for ver_idx, ver_items in verifier_groups.items():
            # Get verifier output (should be same for all items in this group)
            verifier_output = ver_items[0]['verifier_output']
            verifier_node = TreeNode("verifier", verifier_output, [gen_idx, ver_idx])
            generator_node.add_child(verifier_node)
            
            # Add refiner nodes (third level)
            for item in ver_items:
                ref_idx = item['tree_position'][2]
                refiner_output = item['refiner_output']
                refiner_node = TreeNode("refiner", refiner_output, [gen_idx, ver_idx, ref_idx])
                verifier_node.add_child(refiner_node)
    
    return MultiAgentTree(name, root)


def save_tree_to_json(tree: MultiAgentTree, filepath: str):
    """Save tree to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(tree.to_dict(), f, indent=2, ensure_ascii=False)


def load_tree_from_json(filepath: str) -> MultiAgentTree:
    """Load tree from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return MultiAgentTree.from_dict(data)

# Example usage and testing functions
def example_usage():
    """Example of how to use the tree utilities."""
    
    # Example flat data (current format)
    flat_data = [
        {
            'name': 'test_case',
            'tree_position': [0, 0, 0],
            'generator_output': 'gen_0_output',
            'verifier_output': 'ver_0_0_output', 
            'refiner_output': 'ref_0_0_0_output'
        },
        {
            'name': 'test_case',
            'tree_position': [0, 0, 1],
            'generator_output': 'gen_0_output',
            'verifier_output': 'ver_0_0_output',
            'refiner_output': 'ref_0_0_1_output'
        },
        {
            'name': 'test_case', 
            'tree_position': [0, 1, 0],
            'generator_output': 'gen_0_output',
            'verifier_output': 'ver_0_1_output',
            'refiner_output': 'ref_0_1_0_output'
        }
    ]
    
    # Build tree from flat data
    tree = build_tree_from_flat_data(flat_data)
    
    # Save to JSON
    save_tree_to_json(tree, 'example_tree.json')
    
    # Load from JSON
    loaded_tree = load_tree_from_json('example_tree.json')
    
    # Get all leaf outputs (equivalent to current format)
    leaf_outputs = loaded_tree.get_leaf_outputs()
    print("Leaf outputs:")
    for output in leaf_outputs:
        print(f"Index {output['tree_index']}: {output['refiner_output']}")
    
    # Get specific path
    path = loaded_tree.get_path_to_leaf([0, 1, 0])
    print(f"\nPath to [0,1,0]:")
    for node in path:
        print(f"  {node.type}: {node.output}")


def convert_flat_to_nested_tree(flat_json_path: str, tree_json_path: str):
    """
    Convert existing flat format to tree format.
    """
    with open(flat_json_path, 'r', encoding='utf-8') as f:
        flat_data = json.load(f)
    
    # Group by case name
    cases = {}
    for item in flat_data:
        name = item['name']
        if name not in cases:
            cases[name] = []
        cases[name].append(item)
    
    trees = []
    for name, case_data in cases.items():
        tree = build_tree_from_flat_data(case_data)
        trees.append(tree)
    
    # Save trees
    tree_data = [tree.to_dict() for tree in trees]
    with open(tree_json_path, 'w', encoding='utf-8') as f:
        json.dump(tree_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(trees)} cases from flat to tree format")
    print(f"Saved to: {tree_json_path}")


def analyze_tree_structure(tree_json_path: str):
    """
    Analyze the tree structure and print statistics.
    """
    with open(tree_json_path, 'r', encoding='utf-8') as f:
        tree_data = json.load(f)
    
    print(f"Loaded {len(tree_data)} trees")
    
    for i, case in enumerate(tree_data):
        tree = MultiAgentTree.from_dict(case)
        leaves = tree.get_all_leaves()
        print(f"\nCase {i+1}: {tree.name}")
        print(f"  Total leaves: {len(leaves)}")
        print(f"  Tree depth: {max(len(leaf.get_path_to_root()) for leaf in leaves)}")
        
        # Show some example paths
        for j, leaf in enumerate(leaves[:3]):  # Show first 3 paths
            path = leaf.get_path_to_root()
            print(f"  Path {j+1}: {' -> '.join(f'{node.type}[{node.tree_index}]' for node in path)}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input flat JSON file")
    parser.add_argument("--output", help="Output tree JSON file")
    parser.add_argument("--analyze", help="Analyze tree JSON file")
    args = parser.parse_args()
    
    if args.input and args.output:
        convert_flat_to_nested_tree(args.input, args.output)
        analyze_tree_structure(args.output)
    elif args.analyze:
        analyze_tree_structure(args.analyze)
    else:
        example_usage()
