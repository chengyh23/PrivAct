import os
import json
import random
import numpy as np

# Refined logic to support both 2-way (train/test) and 3-way (train/val/test) splits more robustly.

def split_privacylens(json_path, out_prefix="main_data", split_ratio=[0.8, 0.1, 0.1]):
    """
    Splits main_data.json into train/validation/test sets for future ML work.

    Args:
        json_path (str): Path to the main_data.json file.
        out_prefix (str): Prefix for output files.
        split_ratio (list): [train, val, test] or [train, test] as proportions.
    """

    split_ratio = list(split_ratio)
    n = len(split_ratio)
    assert n in [2, 3], "split_ratio must be of length 2 (train/test) or 3 (train/val/test)"
    total = sum(split_ratio)
    split_ratio = [r/total for r in split_ratio]  # Normalize if sum != 1

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    random.seed(42)
    indices = list(range(len(data)))
    random.shuffle(indices)

    num_total = len(data)
    if n == 3:
        train_ratio, val_ratio, test_ratio = split_ratio
        num_train = int(num_total * train_ratio)
        num_val = int(num_total * val_ratio)
        num_test = num_total - num_train - num_val  # Remainder goes to test
        train_indices = indices[:num_train]
        val_indices = indices[num_train:num_train + num_val]
        test_indices = indices[num_train + num_val:]
    elif n == 2:
        train_ratio, test_ratio = split_ratio
        num_train = int(num_total * train_ratio)
        num_test = num_total - num_train
        train_indices = indices[:num_train]
        test_indices = indices[num_train:]
        val_indices = []

    train = [data[i] for i in train_indices]
    test = [data[i] for i in test_indices]
    val = [data[i] for i in val_indices] if val_indices else []

    out_root = os.path.dirname(json_path)
    with open(f"{out_root}/{out_prefix}_train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    if n == 3:
        with open(f"{out_root}/{out_prefix}_val.json", "w", encoding="utf-8") as f:
            json.dump(val, f, ensure_ascii=False, indent=2)
    with open(f"{out_root}/{out_prefix}_test.json", "w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=2)

    print(f"Total: {num_total} | Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"Wrote: {out_prefix}_train.json"
          + (f", {out_prefix}_val.json" if n == 3 else "")
          + f", {out_prefix}_test.json")

def split_pref_pairs(pairs_json_path, train_json_path = "data/main_data_train.json"):
    # Get all case names 'mainX' in main_data_train.json
    with open(train_json_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    train_case_names = [case["name"] for case in train_data]
    print(f"Case names in main_data_train.json: count: {len(train_case_names)}")

    # Filter entries from @Mistral-7B-Instruct-v0.2-branch_2-pref_pairs_refiner.json whose 'name' is in train_case_names
    with open(pairs_json_path, "r", encoding="utf-8") as f:
        pairs_data = json.load(f)
    filtered_pairs = [entry for entry in pairs_data if entry["name"] in train_case_names]

    # Write filtered entries to a new file
    out_filtered_path = pairs_json_path.replace(".json", ".train.json")
    if os.path.exists(out_filtered_path):
        print(f"{out_filtered_path} already exists, skipping writing filtered pairs.")
    else:
        with open(out_filtered_path, "w", encoding="utf-8") as f:
            json.dump(filtered_pairs, f, ensure_ascii=False, indent=2)
    print(f"Wrote filtered pairs to {out_filtered_path}, count: {len(filtered_pairs)}")


if __name__ == "__main__":
    # split_privacylens("data/main_data.json", split_ratio=[0.8,0.2])

    for pref_pairs_fpath in [
        "data_pipeline_MA/predictions/Llama-3.1-8B-Instruct-branch_4-pref_pairs_refiner.json", 
        "data_pipeline_MA/predictions/Llama-3.1-8B-Instruct-branch_4-pref_pairs_verifier.json"]:
        split_pref_pairs(pref_pairs_fpath)
