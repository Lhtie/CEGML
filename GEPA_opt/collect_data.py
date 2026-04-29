import json
import os
import random
import argparse

def iter_epochs(data):
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        if key.startswith("epoch-"):
            yield value
            continue
        if key.startswith("run-"):
            for nested_key, nested_value in value.items():
                if nested_key.startswith("epoch-") and isinstance(nested_value, dict):
                    yield nested_value


def build_example(log):
    prompt = log.get("Prompt", "")
    if "Training Data" not in prompt:
        return None
    training_data_idx = prompt.rfind("Training Data")

    reasoning = log.get("Reasoning") or log.get("Response")
    prediction = log.get("Prediction")
    if not reasoning or prediction is None:
        return None

    return {
        "input": prompt[training_data_idx:],
        "additional_context": {
            "reasoning": reasoning,
        },
        "answer": prediction,
    }


def split_data(
    data: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be smaller than 1.")

    shuffled = list(data)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def collect_data(
    input_dir: str,
    output_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    all_data = []
    for dirpath, dirnames, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.lower().endswith(".json"):
                file_path = os.path.join(dirpath, filename)
                print(f"Found JSON file: {file_path}")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for epoch in iter_epochs(data):
                            for log in epoch.get("Logs", []):
                                if log.get("Equivalent") is not True:
                                    continue
                                example = build_example(log)
                                if example is not None:
                                    all_data.append(example)
                                        
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    split_dataset = split_data(
        all_data,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    with open(output_path, "w", encoding="utf-8") as out_file:
        json.dump(split_dataset, out_file, indent=4)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect data from JSON logs and split into train/val/test sets.")
    parser.add_argument("--input_dir", type=str, default="logs/opt_prompt/icl_gen_extrx", help="Directory to search for JSON log files.")
    parser.add_argument("--output_path", type=str, default="prompting/gepa_icl_gen_extrx.json", help="Path to save the collected dataset in JSON format.")
    args = parser.parse_args()
    
    # icl_gen: data collection
    collect_data(args.input_dir, args.output_path)
