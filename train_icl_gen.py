import torch
import argparse
import random
import os
import json
import re
import tiktoken
import pynini
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from time import sleep
from transformers import AutoTokenizer, AutoModelForCausalLM
from pyformlang.regular_expression import Regex
from pyformlang.finite_automaton import DeterministicFiniteAutomaton

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
        "gm2.5":        "gemini-2.5-pro",
        "cl35":         "claude-3-5",
        "gpt3.5":       "gpt-3.5-turbo",
        "gpt4":         "gpt-4o"
}

prompt_template = """Task: Infer a single regular language (unknown but fixed) from labeled examples, then directly output the infered regex string that is valid for pyformlang.regular_expression.Regex.
Syntax rules:
- Union is | or +; concatenation is a space or a dot .; Kleene star is *; epsilon is epsilon or $.
- Do not use ?, character classes [], {{m,n}}, lookaheads, or anchors.

You could think step by step (keep it concise so that the final answer is outputed), and finally output the regex.
Please wrap your final answer in <ans> and </ans> tags, for example: ... <ans>(a+b)*c</ans>
Training Data:
{0}
"""

train_data_template = "String: {0}\nLabel: {1}"

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
        outputs = model(inputs, max_tokens=8192, temperature=temp)
        res = outputs.choices[0].message.content
        print(f"usage: {outputs.usage}")
    else:
        outputs = model.generate(
            inputs, 
            max_new_tokens=8192,
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

def sigma_from_chars(chars):
    st = pynini.SymbolTable()
    for i, ch in enumerate(chars, start=1):
        st.add_symbol(ch, i)
    return st

def dfa_to_pynini_fst(dfa: DeterministicFiniteAutomaton, sigma: pynini.SymbolTable) -> pynini.Fst:
    f = pynini.Fst()
    idmap = {}
    for q in dfa.states:
        idmap[q] = f.add_state()
    f.set_start(idmap[dfa.start_state])
    for q in dfa.final_states:
        f.set_final(idmap[q])
    for q, a, t in dfa._transition_function.get_edges():
        il = sigma.find(a.value) if hasattr(a, "value") else sigma.find(a)
        if il == -1:
            raise ValueError(f"symbol {a} not in sigma")
        f.add_arc(idmap[q], pynini.Arc(il, il, pynini.Weight.one(f.weight_type()), idmap[t]))
    f.set_input_symbols(sigma); f.set_output_symbols(sigma)
    return pynini.minimize(pynini.determinize(f)).optimize()

def regex_to_pynini_via_pyformlang(rx: str, sigma=None):
    re = Regex(rx)
    nfa = re.to_epsilon_nfa()
    dfa = nfa.to_deterministic().minimize()
    if sigma is None:
        sigma = sigma_from_chars([s.value for s in dfa.symbols])
    fst = dfa_to_pynini_fst(dfa, sigma)
    return fst, sigma

def equivalent_and_witness(A: pynini.Fst, B: pynini.Fst, sigma: pynini.SymbolTable):
    """
    Returns (equivalent: bool, witness: str or None).
    We compute symmetric difference and test emptiness.
    If nonempty, we extract a shortest witness string using shortestpath.
    """
    # Symmetric difference = (A\B) ∪ (B\A)
    symdiff = (pynini.difference(A, B) | pynini.difference(B, A)).optimize()

    # Empty FSA has no valid start
    if symdiff.start() == pynini.NO_STATE_ID or symdiff.num_states() == 0:
        return True, None

    # Get a shortest string in the difference (witness)
    sp = pynini.shortestpath(symdiff).optimize()
    # Convert path to string using the symbol table
    sp.set_input_symbols(sigma)
    sp.set_output_symbols(sigma)
    s = pynini.shortestpath(sp).string(token_type=sigma)  # or rewrite.lattice_to_string(sp)
    return False, s
    
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
    parser.add_argument("--mkey", type=str, default="ds-chat")
    parser.add_argument("--tot_train_size", type=int, default=1280)
    parser.add_argument("--start_size", type=int, default=5)
    parser.add_argument("--scale_factor", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=3)
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

    config_name = f"icl_gen_model={args.mkey}_totTrain={args.tot_train_size}_startSize={args.start_size}_scaleFactor={args.scale_factor}"
    dataset = f".cache/dataset_regex={args.regex}_trainMaxLen={args.max_length}_evalMaxLen={args.eval_max_length}.json"
    with open(dataset, "r") as f:
        data = json.load(f)
    dfa_gt, sigma = regex_to_pynini_via_pyformlang(args.regex)

    agg_losses, num_samples, accs = [], [], []
    agg_train_ex, agg_train_labels = [], []
    msgdict = {}
    num_samples = log_scaling(args.tot_train_size, args.start_size, args.scale_factor)
    print(f"Training sizes per epoch: {num_samples}")
    for epoch in tqdm(range(len(num_samples))):
        l = len(agg_train_ex)
        r = num_samples[epoch]
        train_ex = data["train_ex"][l:r]
        train_labels = data["train_labels"][l:r]
        agg_train_ex += train_ex
        agg_train_labels += train_labels

        train_p = "\n".join([train_data_template.format(ex, label) for ex, label in zip(agg_train_ex, agg_train_labels)])

        msgs = []
        acc, max_token_len = 0, 0
        for i in range(args.retries):
            prompt = prompt_template.format(train_p)
            if args.mkey.startswith("gpt"):
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

            dfa_pred = None
            try:
                dfa_pred, _ = regex_to_pynini_via_pyformlang(pred, sigma)
                eq, witness = equivalent_and_witness(dfa_gt, dfa_pred, sigma)
                acc = max(acc, int(eq))
                msgs[-1]["Equivalent"] = eq
                msgs[-1]["Witness"] = witness

            except Exception as e:
                print(f"Error compiling regex: {e}")
                continue

        accs.append(acc)
        print(f"Accuracy at epoch {epoch}: {acc}, token length: {max_token_len}")

        msgdict[epoch] = {
            "Accuracy": acc,
            "NumTrainingSamples": len(agg_train_ex),
            "Logs": msgs
        }

        os.makedirs(".cache", exist_ok=True)
        with open(f".cache/msgdict_{config_name}.json", "w") as f:
            json.dump(msgdict, f, indent=4)

    plot_accuracy_curve(list(range(len(num_samples))), accs, "accuracy_curves", config_name)