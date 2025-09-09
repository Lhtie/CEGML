import torch
import argparse
import random
import os
import json
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
        "ds7":      "deepseek-ai/deepseek-llm-7b-chat",
        "gpt3":     "gpt-3.5-turbo",
        "gpt4":     "gpt-4-turbo"
}
oai_client = OpenAI(api_key=api_key)

def print_dfa_transitions(dfa):
    print("All states:")
    for state in dfa.states:
        print(f"  {state}")

    print("\nStart state:")
    print(f"  {dfa.start_state}")

    print("\nAccept states:")
    for state in dfa.final_states:
        print(f"  {state}")

    print("\nAll transitions:")
    for state in dfa.states:
        for symbol in dfa.symbols:
            try:
                target = dfa._transition_function(state, symbol)
                print(f"  {state} --{symbol}--> {target}")
            except KeyError:
                continue  # No transition defined

prompt_template = """Task: Infer a single regular language (unknown but fixed) from labeled examples, then classify new strings against that same rule.
Training Data:
{0}

Evaluating Data:
{1}

Please answer True/False to each evaluating data, and output a single list containing all the answers in order. Eg: [True, False, False, ...]
"""

train_data_template = "String: {0}\nLabel: {1}"
eval_data_template = "String: {0}"

def run(mkey, model, tokenizer, msgdict, msg, temp=0.3):
    msgdict.append({ 'role': 'user', 'content': msg })
    if mkey.startswith(("gpt3", "gpt4")):
        inputs = msgdict
    else:
        inputs = tokenizer.apply_chat_template(
                msgdict,
                return_tensors="pt",
                add_generation_prompt=True)
        inputs = inputs.to(device)
    
    if mkey.startswith(("gpt3", "gpt4")):
        sleep(1)
        outputs = model(inputs, max_tokens=1024, temperature=temp)
        ch = outputs.choices[0]
        print("finish_reason:", ch.finish_reason)        # 看是 stop/length/tool_calls/content_filter/...
        print("content_repr:", repr(ch.message.content)) # 看是不是空字符串/全空白
        print("usage:", getattr(outputs, "usage", None))    # 看 prompt/ completion tokens
        msgdict.append({
            'role': 'assistant',
            'content': outputs.choices[0].message.content
        })
    else:
        outputs = model.generate(
            inputs, 
            max_new_tokens=1024,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=temp
        ) # other params: https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/text_generation
        
        msgdict.append({
            'role': 'assistant',
            'content': tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        })
    return msgdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regex", type=str, default="(a(a)*b)*")           # (a(a)*b)* or (a b + b a) (a + b b + c)* (a c + b a)
    parser.add_argument("--max_length", type=int, default=8)
    parser.add_argument("--test_max_length", type=int, default=8)
    parser.add_argument("--mkey", type=str, default="gpt3")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    task = SimplyRegularLanguage(args.regex, args.max_length)
    mkey = args.mkey
    mpath = modelpaths[mkey]
    if args.mkey.startswith(("gpt3", "gpt4")):
        tokenizer = None
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
    num_train_samples, num_train_pos_sam = 0, 0
    msgdict = []
    for epoch in tqdm(range(args.rounds)):
        train_ex = task.generate_random_strings_balanced(
            n=args.batch_size, 
            m=args.max_length
        )
        train_labels = ["True" if task.accepts(x) else "False" for x in train_ex]

        eval_ex = task.generate_random_strings_balanced(
            n=args.batch_size, 
            m=args.max_length * 8
        )
        eval_labels = ["True" if task.accepts(x) else "False" for x in eval_ex]

        train_p = "\n".join([train_data_template.format(ex, label) for ex, label in zip(train_ex, train_labels)])
        eval_p = "\n".join(eval_data_template.format(ex) for ex in eval_ex)
        prompt = prompt_template.format(train_p, eval_p)

        # print(prompt)

        msgdict = run(mkey, model, tokenizer, msgdict, prompt)
        eval_pred = msgdict[-1]["content"][1:-1].split(",")
        eval = sum([int(x.strip() == y.strip()) for x, y in zip(eval_pred, eval_labels)]) / len(eval_labels)

        num_samples.append(len(train_ex))
        accs.append(eval)
        print(f"Accuracy at epoch {epoch}: {eval}, total training samples: {num_train_samples}")

        os.makedirs(".cache", exist_ok=True)
        with open(f".cache/msgdict.json", "w") as f:
            json.dump(msgdict, f, indent=4)

    print(f"Pos train / Tot train = {num_train_pos_sam} / {num_train_samples}")

    plot_accuracy_curve(num_samples, accs, "accuracy_curves", 
                        f"icl_model={args.mkey}")