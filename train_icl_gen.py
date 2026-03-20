import torch
import argparse
import random
import os
import json
import re
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace

from modeling.llm import is_vllm_model, load_model_and_tokenizer
from tasks.rl import SimplyRegularLanguage, ExtRegularLanguage
from learner import LearnerForICLGen
from teacher import Teacher
from curve import plot_loss_curve, plot_accuracy_curve
from dataset import generate_dataset
from keysecrets import api_key
from tasks.utils import dfa_accepts_ex

SIMPLYRX_PROMPT_TEMPLATE = """TASK
You will be given labeled strings and must infer a single regular language that matches all positives (label 1) and rejects all negatives (label 0). Output a concise regex in pyformlang.regular_expression.Regex syntax.

INPUT FORMAT
- You receive a block titled “Training Data (Each line has one input-output pair separated by comma):”.
- Each line is "<string>, <label>" where label ∈ {{1, 0}}. The string may be empty; an empty string appears as nothing before the comma (", 1") and represents epsilon.
- The alphabet is exactly the set of characters appearing in the data (typically a, b, c). Do not introduce other symbols.
{clustered_ce_instr}
PYFORMLANG REGEX SYNTAX
- Union: +
- Concatenation: space-separated symbols (each symbol is a single character from the alphabet or the literal epsilon).
- Kleene star: *
- Parentheses are allowed for grouping; use them whenever you union multi-symbol sequences or need precedence control.
- Spacing rules:
  - Concatenation uses spaces between every symbol: "a b", not "ab".
  - To union sequences, group them: "(a b c + a c c)".
- Epsilon handling: Use the literal epsilon when needed; prefer satisfying epsilon via an existing Kleene star rather than "epsilon + ...", unless epsilon is explicitly required at the top level.
- Do NOT use: | . ? [] {{}} anchors/lookarounds, multi-character tokens, or any symbol not present in the training data.

INFERENCE STRATEGY
1) Start/end constraints:
   - Check if all positives start with a specific letter or set (e.g., all non-empty positives start with c). If so, encode a mandatory prefix, e.g., "c ..." or "(b + c) ...".
   - Check for a forced suffix or final-block restriction (e.g., must end with b or a specific 2-letter tail). Place this outside any repeating block when needed.

2) Length/modular and block structure:
   - Look for fixed-length blocks repeated via "*".
   - More generally:
     - Use a star over a union of allowed blocks when strings can mix block types: "((block1) + (block2))*".
     - If internal blocks allow more endings than the final block, use: "(InternalBlockUnion)* FinalRestrictedBlock".
   - If a singleton positive (e.g., "b") exists alongside block-based strings, include it via a top-level union only if it cannot be captured by a prefix plus star (e.g., "c (...) *" already accepts "c" because the star can be epsilon).

3) Union design: star-of-union vs union-of-stars
   - If strings mix different block types within one string, prefer a star over a union of blocks: "((...)+(...))*".
   - If each positive is formed by repeating exactly one fixed block with no mixing, a compact union of stars can be better: "(a b)* + (a c)* + (b c)*".

4) Compactness tactics:
   - Factor repeated substrings (e.g., "(a+b+c) a b c (...)").
   - Use per-position unions like "(a+b)" or "(a+b+c)" instead of enumerating full strings.
   - Factor common prefixes/suffixes within unions: "(a b c a b + a b c c b)" instead of duplicating.

5) Handling epsilon:
   - Accept epsilon only if explicitly required by the data.
   - Prefer to obtain needed epsilon through an existing Kleene star (e.g., "c (block)*" accepts "c"; "(block)*" accepts epsilon). Use "epsilon +" only when unavoidable at top level (e.g., when the empty string is positive but cannot be included via a star elsewhere).

6) Avoid over-generalization:
   - Do not allow arbitrary middles like "(a+b+c)*" unless strictly supported by all positives and necessary to exclude negatives.
   - Do not invent constraints not universally implied by positives.

7) Quality checks before finalizing:
   - Verify your regex accepts every 1-labeled string and rejects every 0-labeled string.
   - Sanity-check near-misses from negatives (e.g., wrong start letter, wrong modular length, incomplete final block, mixing vs non-mixing).
   - Re-check syntax: unions around multi-symbol sequences, spaces everywhere in concatenation, and only allowed symbols.

{regularization_instr}
{agentic_reflection_instr}
OUTPUT FORMAT
- First, provide 1-3 concise sentences explaining the observed structure (mandatory prefix/set, block size/pattern, modular length, final-block restriction, epsilon/singleton handling), wrapped in <reasoning> and </reasoning>, e.g.:
  <reasoning>All positives start with c and then repeat 2-char blocks from a restricted set.</reasoning>
- Then output ONLY the final regex wrapped in <ans> and </ans>, e.g.:
  <ans>(a a* b)*</ans>

Training Data (Each line has one input-output pair separated by comma):
{0}
"""

EXTRX_SIGMA = "[A-Za-z0-9#]"
EXTRX_PROMPT_TEMPLATE = """TASK
You will be given labeled strings and must infer a single regular language that matches all positives (label 1) and rejects all negatives (label 0). Output a concise regex in our specified syntax (extended from pyformlang.regular_expression.PythonRegex).

INPUT FORMAT
- You receive a block titled “Training Data (Each line has one input-output pair separated by comma):”.
- Each line is "<string>, <label>" where label ∈ {{1, 0}}. The string may be empty; an empty string appears as nothing before the comma (", 1") and represents epsilon.
- The alphabet is fixed. Do not introduce other symbols.
{clustered_ce_instr}
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
  Semantics: L(~(R)) = Σ* \\ L(R)
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

{regularization_instr}
{agentic_reflection_instr}
OUTPUT FORMAT
- First, provide 1-3 concise sentences explaining the observed structure (mandatory prefix/set, block size/pattern, modular length, final-block restriction, epsilon/singleton handling), wrapped in <reasoning> and </reasoning>, e.g.:
  <reasoning>Accepted strings are length-5 alphanumerics that must not contain vowels.</reasoning>
- Then output ONLY the final regex wrapped in <ans> and </ans>, e.g.:
  <ans>(a*([A-Z])|(b))*</ans>

Training Data (Each line has one input-output pair separated by comma):
{0}
"""

SIMPLYRX_REGULARIZATION = """
CONSTRAINTS
- Prefer simpler, more general regexes while staying consistent with all datapoints.
- Total regex length (ignoring spaces) must be ≤ 50 characters.
- Nesting depth of Kleene stars must be ≤ 3.
- Use only symbols that appear in the training data (eg. a, b, c, epsilon).

"""

EXTRX_REGULARIZATION = """
CONSTRAINTS
- Prefer simpler, more general regexes while staying consistent with all datapoints.
- Total regex length (ignoring spaces) must be ≤ 50 characters.
- Nesting depth of Kleene stars (*, +, ?) must be ≤ 3.
- Use only symbols that appear in the alphabet (except metacharacters such as (), |, *, +, ?, []).
"""

SIMPLYRX_CLUSTRED_CE_INSTR = """
- The strings may contain grouped class of characters, e.g., [abc] for letter a or b or c etc.
- Each character class only represent one possible character in the string, e.g., "a[a-c]c" can represent "abc" but not "abcc".

"""

EXTRX_CLUSTRED_CE_INSTR = """
- The strings may contain grouped class of characters, e.g., [A-Z] for uppercase letters, [^0-9] for non-digits, etc.
- Each character class only represent one possible character in the string, e.g., "a[A-Z]c" can represent "aBc" but not "aBCc".

"""

AGENTIC_REFLECTION_INSTR = """
AGENTIC REFLECTION UPDATE
- You will receive the previous attempt's reasoning and regex, plus new counterexamples.
- First, briefly revise the previous reasoning to explain what failed and what should be changed.
- Then output an updated regex consistent with all training data and the counterexamples.
- Keep reasoning concise (1-3 sentences) and directly tied to the regex revision.

"""

AGENTIC_REPAIR_INSTR = """
AGENTIC REPAIR UPDATE
- You are repairing the previous attempt using the failure feedback and repair examples below.
- Repair goals:
  1) Produce a regex that compiles under the required regex syntax.
  2) Fix the specific mistakes exposed by the counterexamples, disagreement witness, and any reported errors.
  3) Preserve the parts of the previous solution that still fit the training data.
- What to do:
  - If the previous regex is invalid, first correct the syntax or unsupported constructs.
  - If a witness or repair example is rejected/accepted incorrectly, revise the regex so that string gets the correct label.
  - Keep the reasoning concise and focused on what changed from the previous attempt.
  - Return the repaired regex in the required output format.

"""

train_data_template = "{0}, {1}"

def extract_ans(res):
    if res is None: return None
    matches = re.search(r"(?:.*)<ans>\s*(.*?)\s*</ans>", res, re.DOTALL)
    if matches:
        return matches.group(1)
    return None

def extract_reasoning(res):
    if res is None: return None
    matches = re.search(r"(?:.*)<reasoning>\s*(.*?)\s*</reasoning>", res, re.DOTALL)
    if matches:
        return matches.group(1)
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


def build_reflection_prompt(previous_reasoning, previous_regex, train_ex, train_labels, feedback_note=None):
    ce_lines = "\n".join(
        [train_data_template.format(ex, label) for ex, label in zip(train_ex, train_labels)]
    )
    prompt = (
        AGENTIC_REFLECTION_INSTR
        + "Reasoning of previous epoch:\n"
        + (previous_reasoning or "(none)")
        + "\nRegex of previous epoch:\n"
        + (previous_regex or "(none)")
    )
    if feedback_note:
        prompt += "\nWhat failed in the previous epoch:\n" + feedback_note + "\n"
    prompt += (
        "\nNew counterexamples (string, label):\n"
        + ce_lines
        + "\n\n"
    )
    return prompt


def build_repair_prompt(previous_reasoning, previous_regex, train_ex, train_labels, feedback_note=None):
    ce_lines = "\n".join(
        [train_data_template.format(ex, label) for ex, label in zip(train_ex, train_labels)]
    )
    prompt = (
        AGENTIC_REPAIR_INSTR
        + "Reasoning of previous attempt:\n"
        + (previous_reasoning or "(none)")
        + "\nRegex of previous attempt:\n"
        + (previous_regex or "(none)")
    )
    if feedback_note:
        prompt += "\nRepair feedback:\n" + feedback_note + "\n"
    if ce_lines:
        prompt += (
            "\nRepair examples (string, label):\n"
            + ce_lines
            + "\n\n"
        )
    else:
        prompt += (
            "\nRepair examples (string, label):\n"
            + "(none available; the previous regex either failed to compile or no repair examples were found)\n\n"
        )
    return prompt


def build_retry_prompt(task, msg, train_ex, train_labels, max_examples=16):
    previous_regex = msg.get("Prediction")
    previous_reasoning = msg.get("Reasoning")

    feedback = []
    if previous_regex is None:
        feedback.append("Unable to extract a regex from previous response.")
    if msg.get("Error"):
        feedback.append(msg["Error"])
    if msg.get("scoreTrainSet") is not None:
        feedback.append(f"Training accuracy of previous regex: {msg['scoreTrainSet']:.3f}")
    if msg.get("scoreEvalSet") is not None:
        feedback.append(f"Eval accuracy of previous regex: {msg['scoreEvalSet']:.3f}")

    repair_ex, repair_labels = [], []
    pred = previous_regex
    dfa_pred = None
    if pred is not None and msg.get("Error") is None:
        try:
            dfa_pred, _, _ = task.regex_to_pynini_via_pyformlang(pred)
        except Exception:
            dfa_pred = None
        if dfa_pred is not None:
            for ex, label in zip(train_ex, train_labels):
                if int(dfa_accepts_ex(dfa_pred, ex)) != label:
                    repair_ex.append(ex)
                    repair_labels.append(label)

    if dfa_pred is not None and not repair_ex:
        return None, True

    if repair_ex:
        repair_ex = repair_ex[:max_examples]
        repair_labels = repair_labels[:max_examples]

    return build_repair_prompt(
        previous_reasoning=previous_reasoning,
        previous_regex=previous_regex,
        train_ex=repair_ex,
        train_labels=repair_labels,
        feedback_note="; ".join(feedback) if feedback else None,
    ), False

def run_episode(
    *,
    config,
    task,
    data,
    teacher,
    prompt_template,
    prompt_kwargs,
    generate_fn,
    on_retry=None,
    on_epoch_end=None,
):
    regex = task.regex_str
    _, fst_gt, _ = task.regex_to_pynini_via_pyformlang(regex)
    accs = []
    agg_train_ex, agg_train_labels = [], []
    num_samples = log_scaling(config.tot_train_size, config.start_size, config.scale_factor)
    epochs = config.ce_epochs if config.use_ce else len(num_samples)

    current_guess = None
    current_guess_reasoning = None
    epoch_results = []

    for epoch in range(epochs):
        reflection_prompt = ""
        if config.use_ce:
            if current_guess is None:
                train_ex = data["train_ex"][epoch * config.ce_start_size:(epoch + 1) * config.ce_start_size]
                train_labels = data["train_labels"][epoch * config.ce_start_size:(epoch + 1) * config.ce_start_size]
            else:
                train_ex, train_labels = teacher.generate_counterexamples(
                    bs=config.ce_batch_size,
                    regex_gt=regex,
                    regex_gen=current_guess,
                    clustered=config.ce_clustered,
                )
                if config.reasoning_mode == "agentic_reflection":
                    reflection_prompt = build_reflection_prompt(
                        current_guess_reasoning,
                        current_guess,
                        train_ex,
                        train_labels,
                    )
        else:
            l, r = len(agg_train_ex), num_samples[epoch]
            train_ex = data["train_ex"][l:r]
            train_labels = data["train_labels"][l:r]

        agg_train_ex += train_ex
        agg_train_labels += train_labels
        train_p = "\n".join(
            [train_data_template.format(ex, label) for ex, label in zip(agg_train_ex, agg_train_labels)]
        )

        msgs, acc = [], 0
        retry_prompt = ""
        retry_done = False
        for retry_idx in range(config.retries):
            iter_prompt_kwargs = dict(prompt_kwargs)
            if config.reasoning_mode == "agentic_reflection":
                iter_prompt_kwargs["agentic_reflection_instr"] = (
                    reflection_prompt + retry_prompt
                )
            msg = generate_fn(
                prompt_template=prompt_template,
                train_prompt=train_p,
                prompt_format_kwargs=iter_prompt_kwargs,
            )
            msg = teacher.judge_regex(
                msg=msg,
                fst_gt=fst_gt,
                train_ex=agg_train_ex,
                train_labels=agg_train_labels,
                eval_ex=data["eval_ex"],
                eval_labels=data["eval_labels"],
            )
            if msg.get("Equivalent"):
                acc = max(acc, 1)
            if (config.reasoning_mode == "agentic_reflection"):
                retry_prompt, retry_done = build_retry_prompt(
                    task=task,
                    msg=msg,
                    train_ex=agg_train_ex,
                    train_labels=agg_train_labels,
                )
            msgs.append(msg)
            if on_retry is not None:
                on_retry(epoch, msgs)
            if msg.get("Equivalent") or retry_done:
                break

        current_guess = msg.get("Prediction")
        current_guess_reasoning = msg.get("Reasoning")

        epoch_result = {
            "Accuracy": acc,
            "NumTrainingSamples": len(agg_train_ex),
            "CurrentGuess": current_guess,
            "CurrentGuessReasoning": current_guess_reasoning,
            "Logs": msgs,
        }
        epoch_results.append(epoch_result)
        accs.append(acc)

        if on_epoch_end is not None:
            on_epoch_end(epoch, epoch_result)

        if acc == 1.0:
            break

    return {
        "epoch_results": epoch_results,
        "epochs_completed": len(epoch_results),
        "final_num_samples": len(agg_train_ex),
        "final_accuracy": accs[-1] if accs else 0.0,
        "current_guess": current_guess,
        "current_guess_reasoning": current_guess_reasoning
    }

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, default="extrx", choices=["simplyrx", "extrx"])
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
    parser.add_argument("--ce_clustered", default=False, action="store_true")
    parser.add_argument(
        "--reasoning_mode",
        type=str,
        default="agentic_reflection",
        choices=["single_inference", "agentic_reflection"],
    )
    parser.add_argument("--indir", type=str, default="datasets")
    parser.add_argument("--outdir", type=str, default="logs/opt_prompt")
    args = parser.parse_args(argv)
    
    args.outdir = os.path.join(args.outdir, f"icl_gen_{args.task_type}")
    use_vllm = is_vllm_model(args.mkey)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not use_vllm and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if args.task_type == "extrx":
        task = ExtRegularLanguage(args.regex, args.max_length, alphabet=EXTRX_SIGMA)
        prompt_template = EXTRX_PROMPT_TEMPLATE
        prompt_kwargs = {
            "sigma": EXTRX_SIGMA,
            "regularization_instr": EXTRX_REGULARIZATION if args.use_reg else "",
            "agentic_reflection_instr": "",
            "clustered_ce_instr": EXTRX_CLUSTRED_CE_INSTR if args.ce_clustered else ""
        }
    else:
        task = SimplyRegularLanguage(args.regex, args.max_length)
        prompt_template = SIMPLYRX_PROMPT_TEMPLATE
        prompt_kwargs = {
            "regularization_instr": SIMPLYRX_REGULARIZATION if args.use_reg else "",
            "agentic_reflection_instr": "",
            "clustered_ce_instr": SIMPLYRX_CLUSTRED_CE_INSTR if args.ce_clustered else ""
        }

    model, tokenizer = load_model_and_tokenizer(args.mkey, api_key)
    teacher = Teacher(task)
    learner = LearnerForICLGen(args.mkey, model, tokenizer, task)

    if not args.use_ce:
        config_name = os.path.join(args.outdir, f"model={args.mkey}/std/")
        config_name += "reg/" if args.use_reg else "noreg/"
        config_name += f"msgdict_regex={args.regex}_totTrain={args.tot_train_size}_startSize={args.start_size}_scaleFactor={args.scale_factor}.json"
    else:
        config_name = os.path.join(args.outdir, f"model={args.mkey}/ce/")
        config_name += "reg/" if args.use_reg else "noreg/"
        config_name += f"{args.reasoning_mode}/"
        config_name += f"msgdict_regex={args.regex}_ceEpochs={args.ce_epochs}_ceBatch={args.ce_batch_size}{'_clustered' if args.ce_clustered else ''}.json"

    generate_dataset(args, task_type=args.task_type, outdir=args.indir)
    dataset = os.path.join(
        args.indir,
        f"regex={args.regex}_trainMaxLen={args.max_length}_evalMaxLen={args.eval_max_length}.json",
    )
    with open(dataset, "r") as f:
        data = json.load(f)

    msgdict, finish_states = {}, {}
    msgdict["summary"] = None
    for runid in range(args.rerun):
        print(f"=== Rerun {runid} ===")
        msgdict[f"run-{runid}"] = {}

        def generate_fn(
            prompt_template,
            train_prompt,
            prompt_format_kwargs,
        ):
            return learner.generate(
                prompt_template=prompt_template,
                train_prompt=train_prompt,
                prompt_format_kwargs=prompt_format_kwargs,
                temp=args.temp,
                answer_extractor=extract_ans,
                reasoning_extractor=extract_reasoning,
            )

        def on_retry(epoch, msgs):
            msgdict[f"run-{runid}"][f"epoch-{epoch}"] = {"Logs": msgs}
            savejson(msgdict, config_name)

        def on_epoch_end(epoch, epoch_result):
            msgdict[f"run-{runid}"][f"epoch-{epoch}"] = epoch_result
            savejson(msgdict, config_name)
            print(f"Accuracy at epoch {epoch}: {epoch_result['Accuracy']}")

        episode_result = run_episode(
            config=args,
            task=task,
            data=data,
            teacher=teacher,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            generate_fn=generate_fn,
            on_retry=on_retry,
            on_epoch_end=on_epoch_end,
        )

        finish_states[f"run-{runid}"] = {
            "epochs": episode_result["epochs_completed"],
            "final_num_samples": episode_result["final_num_samples"],
            "final_accuracy": episode_result["final_accuracy"],
        }
        msgdict["summary"] = finish_states
        savejson(msgdict, config_name)

if __name__ == "__main__":
    main()
