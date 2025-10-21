import torch
import argparse
import random
import os
import json
import re
import tiktoken
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from time import sleep
from transformers import AutoTokenizer, AutoModelForCausalLM

from modeling.RNN import RNN
from tasks.rl import SimplyRegularLanguage
from learner import Learner
from teacher import Teacher
from curve import plot_loss_curve, plot_accuracy_curve
from keysecrets import api_key

device_map = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_map)
modelpaths = {
        "ds7":          "deepseek-ai/deepseek-llm-7b-chat",
        "ds-chat":      "deepseek-chat",
        "ds-reasoner":  "deepseek-reasoner",
        "qw-dsr1":      "DeepSeek-R1-Distill-Qwen-32B",
        "gm2.5":        "gemini-2.5-pro",
        "cl35":         "claude-3-5",
        "gpt3.5":       "gpt-3.5-turbo",
        "gpt4":         "gpt-4o",
        "gpt5":         "gpt-5",
}

prompt_template = """Task: Infer a single regular language (unknown but fixed) from labeled examples, then classify new strings against that same rule.
Please answer 0/1 to each line of the evaluating data.
You could think step by step, and finally output a list containing all the answers in order. (Please briefly explain your reasoning before the final answer)
Please wrap your final answer in <ans> and </ans> tags, for example: ... <ans>[1, 0, ...]</ans>
Training Data (Each line has one input-output pair separated by comma):
{0}

Evaluating Data (Each line has one input string):
{1}
"""

train_data_template = "{0}, {1}"
eval_data_template = "{0}"

def tokens_of_text(enc, text) -> int:
    return len(enc.encode(text, disallowed_special=()))

def useAPI(mkey):
    if mkey.startswith(("gpt", "ds", "gm", "cl")):
        return True
    return False

def run(mkey, model, tokenizer, msg, temp=0.3):
    msgdict = [{'role': 'user', 'content': msg}]
    if useAPI(mkey):
        inputs = msgdict
    else:
        inputs = tokenizer.apply_chat_template(
                msgdict,
                return_tensors="pt",
                add_generation_prompt=True)
        inputs = inputs.to(device)
    
    if useAPI(mkey):
        sleep(1)
        if mkey.startswith("gpt5"):
            outputs = model(inputs, max_completion_tokens=32768)
        else:
            outputs = model(inputs, max_tokens=8192, temperature=temp)
        res = outputs.choices[0].message.content
        print(f"usage: {outputs.usage}")
    else:
        outputs = model.generate(
            inputs, 
            max_new_tokens=32768,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=temp
        ) # other params: https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/text_generation
        
        res = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return res

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
    parser.add_argument("--use_ce", action="store_true", default=False)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    task = SimplyRegularLanguage(args.regex, args.max_length)
    mkey = args.mkey
    mpath = modelpaths[mkey]
    if useAPI(args.mkey):
        tokenizer = None
        if mkey.startswith("gpt"):
            oai_client = OpenAI(api_key=api_key)
            if mkey.startswith(("gpt3", "gpt4")):
                tokenizer = tiktoken.encoding_for_model(mpath)
        elif mkey.startswith("ds"):
            oai_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        elif mkey.startswith("gm"):
            oai_client = OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        elif mkey.startswith("cl"):
            oai_client = OpenAI(api_key=api_key, base_url="https://api.anthropic.com/v1")
        model = lambda msgdict, **k : oai_client.chat.completions.create(
                messages = msgdict,
                model = mpath,
                **k
        )
        devices = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(mpath)
        model = AutoModelForCausalLM.from_pretrained(
            mpath,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        model.eval()

    config_name = f"icl_model={args.mkey}_totTrain={args.tot_train_size}_startSize={args.start_size}_scaleFactor={args.scale_factor}_totEval={args.eval_size}_evalBatch={args.eval_batch_size}"\
                    + ("_ce" if args.use_ce else "")
    dataset = f".cache/dataset_regex={args.regex}_trainMaxLen={args.max_length}_evalMaxLen={args.eval_max_length}.json"
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
            response = run(mkey, model, tokenizer, prompt, args.temp)
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
        acc, max_token_len = 0, 0
        for i in tqdm(range(0, len(eval_ex), args.eval_batch_size)):
            j = min(i+args.eval_batch_size, len(eval_ex))
            eval_ex_batch = eval_ex[i:j]
            eval_labels_batch = eval_labels[i:j]
            eval_p = "\n".join([eval_data_template.format(ex) for ex in eval_ex_batch])

            prompt = prompt_template.format(train_p, eval_p)
            if mkey.startswith(("gpt3", "gpt4")):
                max_token_len = max(max_token_len, tokens_of_text(tokenizer, prompt))
            elif args.mkey.startswith("ds"):
                max_token_len = max(max_token_len, int(len(prompt) * 0.5))

            acc_retried = []
            for retry in range(args.retries):
                response = run(mkey, model, tokenizer, prompt, args.temp)

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
                os.makedirs(".cache", exist_ok=True)
                with open(f".cache/msgdict_{config_name}.json", "w") as f:
                    json.dump(msgdict, f, indent=4)
                
            if len(acc_retried) > 0:
                acc += np.mean(acc_retried)

        acc /= len(eval_ex)
        accs.append(acc)
        print(f"Accuracy at epoch {epoch}: {acc}, total training samples: {len(agg_train_ex)}, token length: {max_token_len}")
        
        msgdict[epoch] = {
            "Accuracy": acc,
            "NumTrainingSamples": len(agg_train_ex),
            "Logs": msgs
        }
        os.makedirs(".cache", exist_ok=True)
        with open(f".cache/msgdict_{config_name}.json", "w") as f:
            json.dump(msgdict, f, indent=4)

    plot_accuracy_curve(train_samples, accs, "accuracy_curves", config_name)