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
        "gm2.5":        "gemini-2.5-flash",
        "cl35":         "claude-3-5",
        "gpt3.5":       "gpt-3.5-turbo",
        "gpt4":         "gpt-4-turbo"
}

prompt_template = """Task: Infer a single regular language (unknown but fixed) from labeled examples, then classify new strings against that same rule.
Training Data:
{0}

Evaluating Data:
{1}

- Please answer True/False to each line of the evaluating data.
- You could think step by step, and finally output a list containing all the answers in order.
- Please wrap your final answer in <ans> and </ans> tags, for example: ... <ans>[True, False, ...]</ans>
"""

train_data_template = "String: {0}\nLabel: {1}"
eval_data_template = "String: {0}"

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
        outputs = model(inputs, max_tokens=1024, temperature=temp)
        res = outputs.choices[0].message.content
    else:
        outputs = model.generate(
            inputs, 
            max_new_tokens=1024,
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
        ans = [x.strip() for x in ans_str[1:-1].split(",")]
        return ans
    else:
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regex", type=str, default="(a(a)*b)*")           # (a(a)*b)* or (a b + b a) (a + b b + c)* (a c + b a)
    parser.add_argument("--max_length", type=int, default=8)
    parser.add_argument("--test_max_length", type=int, default=8)
    parser.add_argument("--mkey", type=str, default="gpt3.5")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_size", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=43)
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

    agg_losses, num_samples, accs = [], [], []
    agg_train_ex, agg_train_labels = [], []
    num_train_samples, num_train_pos_sam = 0, 0
    msgdict = {}
    for epoch in tqdm(range(args.rounds)):
        train_ex = task.generate_random_strings_balanced(
            m=args.batch_size, 
            n=args.max_length
        )
        train_labels = ["True" if task.accepts(x) else "False" for x in train_ex]
        agg_train_ex += train_ex
        agg_train_labels += train_labels

        eval_ex = task.generate_random_strings_balanced(
            m=args.eval_size, 
            n=args.max_length
        )
        eval_labels = ["True" if task.accepts(x) else "False" for x in eval_ex]

        train_p = "\n".join([train_data_template.format(ex, label) for ex, label in zip(agg_train_ex, agg_train_labels)])

        msgs = []
        acc, max_token_len = 0, 0
        for ex, label in zip(eval_ex, eval_labels):
            eval_p = eval_data_template.format(ex)
            prompt = prompt_template.format(train_p, eval_p)
            if args.mkey.startswith("gpt"):
                max_token_len = max(max_token_len, tokens_of_text(tokenizer, prompt))

            response = run(mkey, model, tokenizer, prompt)
            msgs.append({
                "Prompt": prompt,
                "Response": response
            })

            pred = extract_ans(response)
            if pred is not None and len(pred) == 1:
                acc += int(pred[0] == label)

        acc /= len(eval_ex)
        num_samples.append(len(agg_train_ex))
        accs.append(acc)
        print(f"Accuracy at epoch {epoch}: {acc}, total training samples: {len(agg_train_ex)}, token length: {max_token_len}")

        msgdict[epoch] = {
            "Accuracy": acc,
            "NumTrainingSamples": len(agg_train_ex),
            "Logs": msgs
        }

        os.makedirs(".cache", exist_ok=True)
        with open(f".cache/msgdict.json", "w") as f:
            json.dump(msgdict, f, indent=4)

    plot_accuracy_curve(num_samples, accs, "accuracy_curves", 
                        f"icl_model={args.mkey}_batch={args.batch_size}_eval={args.eval_size}")