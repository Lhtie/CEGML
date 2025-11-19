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
        "gpt5":         "gpt-5"
}

# prompt_template = """Task: Infer a single regular language (unknown but fixed) from labeled examples, then directly output the infered regex string that is valid for pyformlang.regular_expression.Regex.
# Syntax rules:
# - Union is +; Concatenation is space-separated tokens (we do not need multi-char tokens); Kleene star is *;
# - Do not use |, ., ?, character classes [], {{m,n}}, lookaheads, or anchors.
# {0}
# You could think step by step, and finally output the regex. (Please briefly explain your reasoning before the final answer)
# Please wrap your final answer in <ans> and </ans> tags, for example: ... <ans>(a+b)*c</ans>
# Training Data (Each line has one input-output pair separated by comma):
# {1}
# """
# regularization = """Premises:
# - Prefer simpler regexes with fewer operators and literals while still consistent with the datapoints.
# - Concretely, the total lengths (ignore spaces) <= 50 characters
# - the depths of klene star nesting <= 3

# """

prompt_template = """You are given a learning task over regular languages. Your job for each task instance is:

- Infer a **single regular language** (unknown but fixed per instance) from **finite labeled examples**, then
- Output a **single regular expression string** (regex) that represents a language **consistent with all the labeled examples**.

The regex must be a valid input for `pyformlang.regular_expression.Regex` and must follow the **exact syntax and constraints** described below.

---

## Input Format (per task instance)

You will be given:

- A block of text titled **"Training Data (Each line has one input-output pair separated by comma):"**.
- Each subsequent line contains:
  - A string over the alphabet `{{a, b, c}}`, possibly empty, then
  - A comma `,` followed by a label:
    - `1` = positive example (string must be accepted by the target language)
    - `0` = negative example (string must be rejected by the target language)

Example of the format:

```text
Training Data (Each line has one input-output pair separated by comma):
ca, 0
acac, 1
, 1
bc, 1
```

Notes:

- The substring before the comma is the actual string. An *empty* substring (nothing before the comma) represents the **empty string** (ε).
- Strings consist only of characters `a`, `b`, `c`, with no spaces. You must interpret each character as a separate symbol.
- The assistant should not assume the presence of any symbol outside `{{a, b, c}}`.

---

## Output Requirements

You must output **one and only one regex** in the required syntax, wrapped inside `<ans>` and `</ans>` tags, e.g.:

```text
<ans>(a+b)* c</ans>
```

Additionally:

- You **must briefly explain your reasoning before the final `<ans>...</ans>` output**.
- The final answer line **must contain only** the `<ans>...</ans>` content (no extra commentary on that line).

---

## Regex Syntax Rules (pyformlang-compatible subset)

You must obey all of the following syntax rules:

1. **Alphabet symbols**:
   - Allowed terminal symbols: `a`, `b`, `c` (single characters).
   - You may also use the standard regex epsilon notation if supported by pyformlang: in many examples, epsilon must be represented *implicitly* by constructs like `X*` (which includes the empty string) rather than as a literal token. In earlier attempts, the literal `ε` symbol was rejected as “not in sigma”. Therefore:
     - **Do not use a literal symbol like `ε` in the regex**.
     - Encode the empty string via Kleene star of appropriate subexpressions (e.g., `(a b)*` includes ε).

2. **Operators**:
   - **Union**: `+`
   - **Concatenation**: juxtaposition of tokens separated by spaces, e.g. `(a+b) c a`
     - A space stands for concatenation between regexes/tokens.
   - **Kleene star**: `*` applied postfix: `R*`

3. **Forbidden constructs**:
   - Do **not** use:
     - `|` (alternative)
     - `.` (dot / any symbol)
     - `?` (optional)
     - Character classes: `[...]`
     - Quantifiers: `{{m}}`, `{{m,n}}`, `+` as “one or more” (note: `+` is union here, not repetition)
     - Lookaheads/lookbehinds
     - Anchors: `^`, `$`
     - Any explicit symbol representing epsilon like `ε` (use star for that).

4. **Grouping**:
   - Use parentheses `(...)` to group subexpressions, especially around unions and where precedence might be ambiguous.
   - Follow the convention that Kleene star `*` has highest precedence, then concatenation, then union `+`. When in doubt, use parentheses explicitly.

5. **Tokenization**:
   - Concatenation is between *tokens* separated by spaces. Each token must be:
     - a single symbol (`a`, `b`, `c`),
     - a grouped subexpression in parentheses, or
     - a grouped subexpression optionally followed by `*`.

   - **Do not write multi-character terminals** like `ac`, `ab`, `bc` as a single symbol. Instead, represent them as concatenation of single-character tokens, e.g.:
     - Incorrect: `(ac+ab+bc)*`  ← `ac` is not a single symbol
     - Correct: `(a c + a b + b c)*` or unions of appropriate starred concatenations as needed.

{0}
---

## Interpreting the Data and Inferring the Language

Your job is **to infer a regular pattern** that fits all data. Some common patterns from prior examples:

- **Fixed-length block repetition**:
  - Example: `c ( (a + c) (a + b + c) (a b c + a c c) )*`
    - Here, all nontrivial positives:
      - Start with `c`.
      - After the initial `c`, the remaining string is split into blocks of length 5.
      - Each block is constrained position-wise:
        - pos1 ∈ {{a, c}}
        - pos2 ∈ {{a, b, c}}
        - last 3 chars either `abc` or `acc` (represented as `(a b c + a c c)`).
    - The `*` indicates any number (including zero) of such blocks.
- **Union of different repetition types**:
  - Example: `(a b)* + (a c)* + (b c)*`
    - Language: strings made of **only** repeated `ab`, or **only** repeated `ac`, or **only** repeated `bc`, including the empty string.
    - Note how each bigram is represented as two concatenated symbols: `a b`, `a c`, `b c`.
    - This correctly captures that mixed patterns like `abac` are negative.
- **Block-structured stars**:
  - Example: `((a+b) (b+c) (a+c) a (b+c) (a+b+c))*`
    - This describes strings formed by repeating a **6-character block**, where at each position the symbol can be drawn from certain subsets:
      - pos1 ∈ {{a, b}} → `(a+b)`
      - pos2 ∈ {{b, c}} → `(b+c)`
      - pos3 ∈ {{a, c}} → `(a+c)`
      - pos4 = a alone → `a`
      - pos5 ∈ {{b, c}} → `(b+c)`
      - pos6 ∈ {{a, b, c}} → `(a+b+c)`
    - The entire string is zero or more repetitions of such blocks; empty string is allowed via the outer `*`.

**Strategy hints**:

1. **Examine positives and negatives carefully**:
   - Look for:
     - Shared prefixes or suffixes.
     - Length constraints (e.g., lengths of form `1 + 5k` or multiples of some block size).
     - Repeated subpatterns (`ab` repeated, certain substrings always appearing).
     - Allowed symbol choices at specific positions in fixed-size blocks.

2. **Check segmentability**:
   - For longer positives, try partitioning into equal-sized blocks (e.g., length-2, length-3, length-5, length-6).
   - See if each position in a block can be described as a union of symbols, as in `(a+b)` etc.

3. **Confirm against negatives**:
   - Ensure that:
     - Negative strings violate length constraints, or
     - Contain forbidden substrings, or
     - Break the block-wise position conditions, or
     - Mix different allowed block types when only pure repetition is allowed, etc.

4. **Representing the empty string**:
   - Use the fact that `R*` accepts the empty string.
   - Do **not** insert a literal epsilon symbol; instead, choose a starred pattern that includes ε when needed (e.g., `X*`, `((...) ...)*`).

5. **Avoid invalid tokens**:
   - Each letter is a separate token; pairs or triples must be represented via concatenation:
     - Write `a b` for the string “ab”; write `a b c` for “abc”.
     - For union of `abc` and `acc`: `(a b c + a c c)`.

6. **Respect all examples**:
   - Unlike some earlier incorrect guesses, *do not purposely ignore some positives or negatives* to simplify.
   - Validate at least mentally that all provided positives match and all negatives do not.

---

## Reasoning and Answer Format

Your response for each task instance must:

1. Provide a **brief but clear reasoning** section explaining:
   - The observed structure in the examples (e.g., block sizes, unions, prefix/suffix constraints).
   - Why the chosen regex matches all positives and rejects all negatives.
   - That you are honoring the syntax and complexity constraints.

2. End with a **single line** containing only the final regex surrounded by `<ans>` and `</ans>` tags. For example:

```text
<ans>c ( (a + c) (a + b + c) (a b c + a c c) )*</ans>
```

Do not add extra output after this line.

Follow these instructions exactly for every new dataset you receive.
Training Data (Each line has one input-output pair separated by comma):
{1}
"""
regularization = """---

## Structural and Complexity Constraints

Your inferred regex must satisfy the following constraints:

1. **Consistency**:
   - Every positive example string (labeled `1`) must be **accepted** by the regex language.
   - Every negative example string (labeled `0`) must be **rejected** by the regex language.

2. **Simplicity preference**:
   - Among all consistent regexes, **prefer simpler ones**:
     - Fewer operators and literals overall.
     - Simpler structural patterns.
   - However, **do not sacrifice correctness**: you must not knowingly violate any labeled example just to simplify.

3. **Formal restrictions**:
   - Let “length” mean the number of **non-space** characters in the regex string (including parentheses, operators, and symbols). This length must be:
     - `<= 50`
   - **Nesting depth of Kleene stars**:
     - The maximum depth of nested `*` operators must be `<= 3`.
     - Depth count example:
       - `a*` has depth 1.
       - `(a* b*)*` has depth 2.
       - `((a b)*)*` has depth 2.
       - Avoid patterns like `((a)*)* *` that would push depth above 3.

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
    parser.add_argument("--regex", type=str, default="(a(b+c)(a+b)c(a+c)b(a+b+c)(a+b+c))*")           # (a(a)*b)* or (a b + b a) (a + b b + c)* (a c + b a)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--eval_max_length", type=int, default=32)
    parser.add_argument("--mkey", type=str, default="gpt-5.1")
    parser.add_argument("--tot_train_size", type=int, default=384)
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

    task = SimplyRegularLanguage(args.regex, args.max_length)
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
        config_name = f"logs/opt_prompt/icl_gen/model={args.mkey}/std/"
        config_name += "reg/" if args.use_reg else "noreg/"
        config_name += f"msgdict_regex={args.regex}_totTrain={args.tot_train_size}_startSize={args.start_size}_scaleFactor={args.scale_factor}.json"
    else:
        config_name = f"logs/opt_prompt/icl_gen/model={args.mkey}/ce/"
        config_name += "reg/" if args.use_reg else "noreg/"
        config_name += f"msgdict_regex={args.regex}_ceEpochs={args.ce_epochs}_ceBatch={args.ce_batch_size}.json"
    dataset = f"datasets/regex={args.regex}_trainMaxLen={args.max_length}_evalMaxLen={args.eval_max_length}.json"
    with open(dataset, "r") as f:
        data = json.load(f)
    dfa_gt, fst_gt, sigma = task.regex_to_pynini_via_pyformlang(args.regex)
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
                    dfa_pred, fst_pred, _ = task.regex_to_pynini_via_pyformlang(pred, sigma)
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