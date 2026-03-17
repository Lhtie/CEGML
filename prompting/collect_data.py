import json
import os

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


def collect_data(input_dir: str, output_path: str):
    # Collect training data from JSON files in the input directory and save to output path
    
    train_data = []
    seen_answers = set()
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
                                if example is not None and example["answer"] not in seen_answers:
                                    seen_answers.add(example["answer"])
                                    train_data.append(example)
                                        
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    with open(output_path, "w", encoding="utf-8") as out_file:
        json.dump({"train": train_data, "val": []}, out_file, indent=4)
        
if __name__ == "__main__":
    # icl_gen: data collection
    collect_data("logs/opt_prompt/icl_gen_simplyrx", "prompting/gepa_icl_gen_simplyrx.json")
