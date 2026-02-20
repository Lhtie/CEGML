import torch
import argparse
import random
import os
import json
import re
import numpy as np
from tqdm import tqdm

from modeling.RNN import RNN
from modeling.llm import load_model_and_tokenizer, run_model
from tasks.rl import SimplyRegularLanguage
from learner import Learner
from teacher import Teacher
from curve import plot_loss_curve, plot_accuracy_curve
from dataset import generate_dataset
from keysecrets import api_key

device_map = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_map)

prompt_template = """Task: Infer a single regular language (unknown but fixed) from labeled examples, then classify new strings against that same rule.
{0}
Please answer 0/1 to each line of the evaluating data.
You could think step by step, and finally output a list containing all the answers in order. (Please briefly explain your reasoning before the final answer)
Please wrap your final answer in <ans> and </ans> tags, for example: ... <ans>[1, 0, ...]</ans>
Training Data (Each line has one input-output pair separated by comma):
{0}

Evaluating Data (Each line has one input string):
{1}
"""
regularization = """Premises:
- Prefer simpler regexes with fewer operators and literals while still consistent with the datapoints.
- Concretely, the total lengths (ignore spaces) <= 50 characters
- the depths of klene star nesting <= 3

"""


train_data_template = "{0}, {1}"
eval_data_template = "{0}"

def tokens_of_text(enc, text) -> int:
    return len(enc.encode(text, disallowed_special=()))

def extract_ans(res):
    match = re.search(r"<ans>\s*(\[.*?\])\s*</ans>", res, re.DOTALL)
    if match:
        ans_str = match.group(1)
        try:
            ans = [int(x.strip()) for x in ans_str[1:-1].split(",")]
            return ans
        except ValueError:
            return None
    else:
        return None
    
def log_scaling(total, start, scale_factor):
    sizes = []
    current = start
    while current < total:
        sizes.append(current)
        current = int(current * scale_factor)
    sizes.append(total)
    return sizes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regex", type=str, default="(a(b+c)(a+b)c(a+c)b(a+b+c)(a+b+c))*")           # (a(a)*b)* or (a b + b a) (a + b b + c)* (a c + b a)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--eval_max_length", type=int, default=32)
    parser.add_argument("--mkey", type=str, default="gpt5")
    parser.add_argument("--tot_train_size", type=int, default=384)
    parser.add_argument("--start_size", type=int, default=3)
    parser.add_argument("--scale_factor", type=float, default=2.0)
    parser.add_argument("--eval_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--use_reg", default=False, action="store_true")
    parser.add_argument("--use_ce", default=False, action="store_true")
    parser.add_argument("--indir", type=str, default="datasets/")
    parser.add_argument("--outdir", type=str, default="logs/icl/")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    task = SimplyRegularLanguage(args.regex, args.max_length)
    model, tokenizer = load_model_and_tokenizer(args.mkey, api_key)

    config_name = os.path.join(args.outdir, f"model={args.mkey}/")
    config_name += "ce/" if args.use_ce else "std/"
    config_name += "reg/" if args.use_reg else "noreg/"
    config_name += f"msgdict_regex={args.regex}_totTrain={args.tot_train_size}_startSize={args.start_size}_scaleFactor={args.scale_factor}_totEval={args.eval_size}_evalBatch={args.eval_batch_size}.json"
    
    generate_dataset(args, task_type="simplyrx", outdir=args.indir)
    dataset = os.path.join(args.indir, f"regex={args.regex}_trainMaxLen={args.max_length}_evalMaxLen={args.eval_max_length}.json")
    with open(dataset, "r") as f:
        data = json.load(f)

    agg_losses, num_samples, accs = [], [], []
    agg_train_ex, agg_train_labels = [], []
    msgdict = {}
    num_samples = log_scaling(args.tot_train_size, args.start_size, args.scale_factor)
    train_samples = []
    print(f"Training sizes per epoch: {num_samples}")
    for epoch in tqdm(range(len(num_samples))):
        l = len(agg_train_ex)
        r = num_samples[epoch]
        train_ex = data["train_ex"][l:r]
        train_labels = data["train_labels"][l:r]

        if args.use_ce:
            train_p = "\n".join([train_data_template.format(ex, label) for ex, label in zip(agg_train_ex, agg_train_labels)])
            eval_p = "\n".join([eval_data_template.format(ex) for ex in train_ex])

            prompt = prompt_template.format(train_p, eval_p)
            response = run_model(args.mkey, model, tokenizer, prompt, device=device, temp=args.temp)
            pred = extract_ans(response)
            ce_x, ce_y = [], []
            if pred is not None and len(pred) == len(train_labels):
                for string, p, l in zip(train_ex, pred, train_labels):
                    if p != l:
                        ce_x.append(string)
                        ce_y.append(l)
            
            agg_train_ex += ce_x
            agg_train_labels += ce_y
            train_samples.append(len(agg_train_ex))
        else:
            agg_train_ex += train_ex
            agg_train_labels += train_labels
            train_samples.append(len(agg_train_ex))

        eval_ex = data["eval_ex"]
        eval_labels = data["eval_labels"]
        train_p = "\n".join([train_data_template.format(ex, label) for ex, label in zip(agg_train_ex, agg_train_labels)])

        msgs = []
        acc = 0
        for i in tqdm(range(0, len(eval_ex), args.eval_batch_size)):
            j = min(i+args.eval_batch_size, len(eval_ex))
            eval_ex_batch = eval_ex[i:j]
            eval_labels_batch = eval_labels[i:j]
            eval_p = "\n".join([eval_data_template.format(ex) for ex in eval_ex_batch])

            prompt = prompt_template.format(
                regularization if args.use_reg else "",
                train_p, eval_p
            )

            acc_retried = []
            for retry in range(args.retries):
                response = run_model(args.mkey, model, tokenizer, prompt, device=device, temp=args.temp)

                pred = extract_ans(response)
                if pred is not None and len(pred) == len(eval_labels_batch):
                    acc_single = sum([int(p == l) for p, l in zip(pred, eval_labels_batch)])
                    acc_retried.append(acc_single)
                else:
                    acc_single = None

                msgs.append({
                    "BatchIndices": (i, j),
                    "Retry": retry,
                    "Prompt": prompt,
                    "Response": response,
                    "Acc": acc_single,
                    "Prediction": pred,
                    "GroundTruth": eval_labels_batch
                })
                msgdict[epoch] = {
                    "Logs": msgs
                }
                os.makedirs(os.path.dirname(config_name), exist_ok=True)
                with open(config_name, "w") as f:
                    json.dump(msgdict, f, indent=4)
                
            if len(acc_retried) > 0:
                acc += np.mean(acc_retried)

        acc /= len(eval_ex)
        accs.append(acc)
        print(f"Accuracy at epoch {epoch}: {acc}, total training samples: {len(agg_train_ex)}")
        
        msgdict[epoch] = {
            "Accuracy": acc,
            "NumTrainingSamples": len(agg_train_ex),
            "Logs": msgs
        }
        os.makedirs(os.path.dirname(config_name), exist_ok=True)
        with open(config_name, "w") as f:
            json.dump(msgdict, f, indent=4)

    plot_accuracy_curve(
        train_samples, accs, 
        os.path.dirname(config_name).replace("logs", "accuracy_curves"),
        os.path.basename(config_name)
    )
