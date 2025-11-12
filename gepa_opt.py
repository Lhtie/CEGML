import gepa
import json
import os
import keysecrets

def collect_data(input_dir: str, output_path: str):
    train_data = []
    for dirpath, dirnames, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.lower().endswith(".json"):
                file_path = os.path.join(dirpath, filename)
                print(f"Found JSON file: {file_path}")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for epoch, res in data.items():
                            if float(res["Accuracy"]) == 1.0:
                                for log in res["Logs"]:
                                    if "Equivalent" in log and bool(log["Equivalent"]) == True:
                                        train_data.append({
                                            "input": "Training Data" + log["Prompt"].split("Training Data")[1],
                                            "additional_context": {
                                                "solution": log["Response"]
                                            },
                                            "answer": log["Prediction"]
                                        })
                                        
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    with open(output_path, "w", encoding="utf-8") as out_file:
        json.dump({"train": train_data, "val": []}, out_file, indent=4)

prompt_template = """Task: Infer a single regular language (unknown but fixed) from labeled examples, then directly output the infered regex string that is valid for pyformlang.regular_expression.Regex.
Syntax rules:
- Union is +; Concatenation is space-separated tokens (we do not need multi-char tokens); Kleene star is *;
- Do not use |, ., ?, character classes [], {{m,n}}, lookaheads, or anchors.
Premises:
- Prefer simpler regexes with fewer operators and literals while still consistent with the datapoints.
- Concretely, the total lengths (ignore spaces) <= 50 characters
- the depths of klene star nesting <= 3

You could think step by step, and finally output the regex. (Please briefly explain your reasoning before the final answer)
Please wrap your final answer in <ans> and </ans> tags, for example: ... <ans>(a+b)*c</ans>
"""

os.environ["OPENAI_API_KEY"] = keysecrets.api_key

if __name__ == "__main__":
    # icl_gen
    # collect_data("logs", "datasets/gepa_icl_gen.json")

    with open("datasets/gepa_icl_gen.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Data size:", len(data["train"]))
    
    gepa_result = gepa.optimize(
        seed_candidate={
            "system_prompt": prompt_template
        },
        trainset=data["train"],
        task_lm="openai/gpt-5",
        max_metric_calls=150,
        reflection_lm="openai/gpt-5"
    )

    print("GEPA Optimized Prompt:", gepa_result.best_candidate['system_prompt'])