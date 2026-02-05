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
from tasks.rl import PythonRegularLanguage, ExtRegularLanguage
from learner import Learner
from teacher import Teacher
from curve import plot_loss_curve, plot_accuracy_curve
from dataset import generate_dataset
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
        "gpt5":         "gpt-5"
}

sigma = "[A-Za-z0-9#]"
prompt_template = """TASK
You will be given labeled strings and must infer a single regular language that matches all positives (label 1) and rejects all negatives (label 0). Output a concise regex in our specified syntax (extended from pyformlang.regular_expression.PythonRegex).

INPUT FORMAT
- You receive a block titled “Training Data (Each line has one input-output pair separated by comma):”.
- Each line is "<string>, <label>" where label ∈ {{1, 0}}. The string may be empty; an empty string appears as nothing before the comma (", 1") and represents epsilon.
- The alphabet is fixed. Do not introduce other symbols.

EXT REGEX SYNTAX (Extended PythonRegex)

Alphabet
- The alphabet is fixed: Σ = {sigma}
- No other characters may appear anywhere in the regex.
- No escape sequences are supported. Do not use '\\' at all.
Atomic forms
1) Literal character: Any single symbol in Σ
2) Character class:
   - Syntax: [ ... ]
   - Contents may use only:
     * ranges like A-Z, 0-9
     * an individual literal symbol
Core operators (* are extended operators beyond PythonRegex)
- Concatenation: implicit by adjacency
  Example: ab means 'a' followed by 'b'
- Union (OR): |
  Example: a|b means 'a' or 'b'
- Grouping: ( ... )
  Parentheses define scope and precedence.
- *Conjunction / intersection: &
  Semantics: L(R1 & R2) = L(R1) ∩ L(R2)
- *Negation / complement: ~(R)
  Semantics: L(~(R)) = Σ* \ L(R)
  Negation must always be written with parentheses: ~( ... )
Quantifiers
Quantifiers apply to the immediately preceding atom or parenthesized group.
- * : zero or more
- + : one or more
- ? : zero or one
- {{n}} : exactly n repetitions (n is a nonnegative integer)
- {{n,m}}: between n and m repetitions inclusive (0 <= n <= m)
- {{n,}} : at least n repetitions, equivalent to “(E){{n}}(E)*”
Associativity
- Concatenation, &, and | are left-associative.
- Parenthesize whenever there is ambiguity.
Priority (from highest to lowest): Quantifiers, ~, Concatenation, &, |
Prohibited constructs (must not appear)
- Do not use '.' (dot). Use [A-Za-z0-9#] explicitly when you need Σ.
- Do not use negated character classes [^...].
- Do not use anchors ^ or $.
- Do not use word boundary \\b.
- Do not use lookarounds or backreferences.
{0}
INFERENCE STRATEGY

1) Start/end constraints:
   - Check if all positives start with a specific character or set.
     If so, encode a mandatory prefix (example: "^c.*" becomes "c.*" since anchors are implicit).
   - Check for a forced suffix or ending pattern.
     Place this outside repeating blocks when required.

2) Length/modular and block structure:
   - Look for fixed substrings repeated via "*".
   - If strings mix multiple allowed blocks internally, prefer:
       (block1|block2)*
   - If internal repetitions are freer than endings, use:
       (InternalBlockUnion)* FinalRestrictedBlock
   - If a singleton positive exists (example: "b"), include it using union only if it cannot be captured via star behavior.

3) Union design: star-of-union vs union-of-stars
   - If block types mix inside strings: (A|B|C)*
   - If each string repeats exactly one block type: A*|B*|C*

4) Compactness tactics:
   - Factor common prefixes/suffixes.
   - Use character classes when appropriate:
       [ab] instead of (a|b)
   - Factor repeated substrings inside unions:
       ab(c|d) instead of abc|abd

5) Handling epsilon:
   - Accept epsilon only if explicitly required.
   - Prefer x* instead of (|x)* or (x|).
   - Only use empty alternation (|) when unavoidable.

6) Avoid over-generalization:
   - Do NOT allow patterns contradicted by negatives.

7) Quality checks before finalizing:
   - Verify acceptance of all label-1 strings.
   - Verify rejection of all label-0 strings.
   - Check boundary cases (short strings, empty string).
   - Re-check syntax correctness and grouping.

OUTPUT FORMAT
- First, provide 1–3 concise sentences explaining the observed structure (mandatory prefix/set, block size/pattern, modular length, final-block restriction, epsilon/singleton handling).
- Then output ONLY the final regex wrapped in <ans> and </ans>, e.g.:
  <ans>(a*(c)|(b))*</ans>

Training Data (Each line has one input-output pair separated by comma):
{1}
"""
regularization = """
CONSTRAINTS
- Prefer simpler, more general regexes while staying consistent with all datapoints.
- Total regex length (ignoring spaces) must be ≤ 50 characters.
- Nesting depth of Kleene stars (*, +, ?) must be ≤ 3.
- Use only symbols that appear in the alphabet (except metacharacters such as (), |, *, +, ?, []).

"""

train_data_template = "{0}, {1}"

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
        if mkey.startswith(("gpt5", "gpt-5")):
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
    match = re.search(r"<ans>\s*(.*?)\s*</ans>", res, re.DOTALL)
    if match:
        ans_str = match.group(1)
        return ans_str
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

def savejson(ctx, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(ctx, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regex", type=str, default="[A-Za-z0-9#]*z[A-Za-z]*[A-Za-z0-9#]*")
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--eval_max_length", type=int, default=32)
    parser.add_argument("--mkey", type=str, default="gpt5")
    parser.add_argument("--tot_train_size", type=int, default=384)
    parser.add_argument("--eval_size", type=int, default=32)
    parser.add_argument("--start_size", type=int, default=3)
    parser.add_argument("--scale_factor", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--rerun", type=int, default=3)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--use_reg", default=False, action="store_true")
    parser.add_argument("--use_ce", default=False, action="store_true")
    parser.add_argument("--ce_epochs", type=int, default=8)
    parser.add_argument("--ce_start_size", type=int, default=8)
    parser.add_argument("--ce_batch_size", type=int, default=128)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    task = ExtRegularLanguage(args.regex, args.max_length, sigma=sigma)
    mkey = args.mkey
    if mkey in modelpaths:
        mpath = modelpaths[mkey]
    else: mpath = mkey
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
            device_map="auto",
        )
        model.eval()

    if not args.use_ce:
        config_name = f"logs/opt_prompt/icl_gen_extrx/model={args.mkey}/std/"
        config_name += "reg/" if args.use_reg else "noreg/"
        config_name += f"msgdict_regex={args.regex}_totTrain={args.tot_train_size}_startSize={args.start_size}_scaleFactor={args.scale_factor}.json"
    else:
        config_name = f"logs/opt_prompt/icl_gen_extrx/model={args.mkey}/ce/"
        config_name += "reg/" if args.use_reg else "noreg/"
        config_name += f"msgdict_regex={args.regex}_ceEpochs={args.ce_epochs}_ceBatch={args.ce_batch_size}.json"
        
    dataset = f"datasets/regex={args.regex}_trainMaxLen={args.max_length}_evalMaxLen={args.eval_max_length}.json"
    if not os.path.exists(dataset):
        generate_dataset(args, task_type="extrx")
        
    with open(dataset, "r") as f:
        data = json.load(f)
    dfa_gt, fst_gt, _ = task.regex_to_pynini_via_pyformlang(args.regex)
    eval_ex = data["eval_ex"]
    eval_labels = data["eval_labels"]
    teacher = Teacher(task)

    msgdict, finish_states = {}, {}
    msgdict["summary"] = None
    for runid in range(args.rerun):
        print(f"=== Rerun {runid} ===")
        agg_losses, accs = [], []
        agg_train_ex, agg_train_labels = [], []
        num_samples = log_scaling(args.tot_train_size, args.start_size, args.scale_factor)
        epochs = args.ce_epochs if args.use_ce else len(num_samples)
        current_guess = None
        msgdict[f"run-{runid}"] = {}
        
        for epoch in tqdm(range(epochs)):
            if args.use_ce:
                if current_guess is None:
                    ce_x = data["train_ex"][epoch*args.ce_start_size:(epoch+1)*args.ce_start_size]
                    ce_y = data["train_labels"][epoch*args.ce_start_size:(epoch+1)*args.ce_start_size]
                else:
                    ce_x, ce_y = teacher.generate_counterexamples(
                        bs=args.ce_batch_size,
                        regex_gt=args.regex,
                        regex_gen=current_guess
                    )
                agg_train_ex += ce_x
                agg_train_labels += ce_y

            else:
                l = len(agg_train_ex)
                r = num_samples[epoch]
                train_ex = data["train_ex"][l:r]
                train_labels = data["train_labels"][l:r]
                agg_train_ex += train_ex
                agg_train_labels += train_labels

            train_p = "\n".join([train_data_template.format(ex, label) for ex, label in zip(agg_train_ex, agg_train_labels)])

            msgs = []
            acc, max_token_len, best_eval = 0, 0, 0
            for i in range(args.retries):
                prompt = prompt_template.format(
                    regularization if args.use_reg else "",
                    train_p
                )
                if mkey.startswith(("gpt3", "gpt4")):
                    max_token_len = max(max_token_len, tokens_of_text(tokenizer, prompt))
                elif args.mkey.startswith("ds"):
                    max_token_len = max(max_token_len, int(len(prompt) * 0.5))

                response = run(mkey, model, tokenizer, prompt, args.temp)
                pred = extract_ans(response)
                msgs.append({
                    "Prompt": prompt,
                    "Response": response,
                    "Prediction": pred
                })

                try:
                    dfa_pred, fst_pred, sigma = task.regex_to_pynini_via_pyformlang(pred)
                    eq, witness = task.equivalent_and_witness(fst_gt, fst_pred, sigma)
                    acc = max(acc, int(eq))
                    msgs[-1]["Equivalent"] = eq
                    msgs[-1]["Witness"] = witness
                    msgs[-1]["scoreTrainSet"] = sum([int(int(dfa_pred.accepts(ex)) == label) for ex, label in zip(agg_train_ex, agg_train_labels)]) / len(agg_train_ex)
                    msgs[-1]["scoreEvalSet"] = sum([int(int(dfa_pred.accepts(ex)) == label) for ex, label in zip(eval_ex, eval_labels)]) / len(eval_ex)
                    if msgs[-1]["scoreEvalSet"] > best_eval:
                        best_eval = msgs[-1]["scoreEvalSet"]
                        current_guess = pred

                except Exception as e:
                    print(f"Error compiling regex: {e}")
                    continue

                msgdict[f"run-{runid}"][f"epoch-{epoch}"] = {
                    "Logs": msgs
                }
                savejson(msgdict, config_name)

            accs.append(acc)
            print(f"Accuracy at epoch {epoch}: {acc}, token length: {max_token_len}")

            msgdict[f"run-{runid}"][f"epoch-{epoch}"] = {
                "Accuracy": acc,
                "NumTrainingSamples": len(agg_train_ex),
                "Logs": msgs
            }
            savejson(msgdict, config_name)

            if acc == 1.0:
                print(f"Early stop at epoch {epoch}")
                break
            
        finish_states[f"run-{runid}"] = {
            "epochs": epoch + 1,
            "final_num_samples": len(agg_train_ex),
            "final_accuracy": accs[-1]
        }
        msgdict["summary"] = finish_states
        savejson(msgdict, config_name)

    # plot_accuracy_curve(list(range(len(num_samples))), accs, "accuracy_curves", config_name)