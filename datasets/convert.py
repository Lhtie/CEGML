import sys
import os
import json
import re

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from tasks.rl import NLRXRegularLanguage
from tasks.utils import split_regex_into_atoms

def _parse_unbounded_quant(q):
    m = re.fullmatch(r"\{\s*(\d+)\s*,\s*\}", q or "")
    return int(m.group(1)) if m else None

def normalize_unbounded_quant(node):
    """
    Transform {n,} into equivalent {n} + * concatenation in the tree.
    Returns a new node tree (does not mutate input).
    """
    def clone_base(n0):
        if n0["type"] == "atom":
            return {"type": "atom", "value": n0["value"], "quant": "", "negated": False}
        return {
            "type": "group",
            "children": [normalize_unbounded_quant(c) for c in n0.get("children", [])],
            "ops": list(n0.get("ops", [])),
            "quant": "",
            "negated": False,
        }

    num = _parse_unbounded_quant(node.get("quant", ""))
    base = clone_base(node)
    if num is None:
        base["quant"] = node.get("quant", "")
        base["negated"] = node.get("negated", False)
        return base
    if num == 0:
        base["quant"] = "*"
        base["negated"] = node.get("negated", False)
        return base

    fixed = dict(base, quant="{" + str(num) + "}")
    repeat = dict(base, quant="*")
    return {
        "type": "group",
        "children": [fixed, repeat],
        "ops": ["."],
        "quant": "",
        "negated": node.get("negated", False),
    }

def restrict_alphabet(node, alphabet):
    """
    Restrict atoms to given alphabet by replacing .* and [^...] with union of all alphabet symbols.
    Returns a new node tree (does not mutate input).
    """
    def make_union_group(alphabet, q, neg):
        return {
            "type": "atom",
            "value": f"[{''.join(alphabet)}]",
            "quant": q,
            "negated": neg,
        }

    if node["type"] == "atom":
        val = node["value"]
        q = node.get("quant", "")
        neg = node.get("negated", False)
        if val == ".":
            return make_union_group(alphabet, q, neg)
        if val.startswith("[^"):
            val = val[2:-1]
            return make_union_group([a for a in alphabet if a not in val], q, neg)
        if not val.startswith("["):
            if val == "\\b":
                val = ""
            elif not re.fullmatch(rf"[{''.join(alphabet)}]", val):
                val = "#"
        return {"type": "atom", "value": val, "quant": q, "negated": neg}

    return {
        "type": "group",
        "children": [restrict_alphabet(c, alphabet) for c in node.get("children", [])],
        "ops": list(node.get("ops", [])),
        "quant": node.get("quant", ""),
        "negated": node.get("negated", False),
    }

def tree_to_regex(node, par):
    """
    Serialize tree back to regex string. Uses "." as implicit concat.
    """
    neg = node.get("negated", False)
    if node["type"] == "atom":
        prefix = "~" if neg else ""
        return prefix + node["value"] + node.get("quant", "")

    children = node.get("children", [])
    ops = node.get("ops", [])

    parts = []
    for idx, child in enumerate(children):
        child_rx = tree_to_regex(child, node)
        parts.append(child_rx)
        if idx < len(ops) and ops[idx] != ".":
            parts.append(ops[idx])

    inner = "".join(parts) if par is None else "(" + "".join(parts) + ")"
    if neg:
        inner = "~" + inner
    return inner + node.get("quant", "")

def convert_nlrx_to_pyrx(nlrx_lines):
    alphabet = ["A-Z", "a-z", "0-9", "#"]
    
    pyrx_lines = []
    for line in nlrx_lines:
        tree = split_regex_into_atoms(line)
        tree = normalize_unbounded_quant(tree)
        tree = restrict_alphabet(tree, alphabet)
        line = tree_to_regex(tree, None)
        
        pyrx_lines.append(line)
    return pyrx_lines

if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "KB13.txt"), "r") as f:
        nlrx_lines = f.read().splitlines()
        nlrx_lines = [line.strip() for line in nlrx_lines]
    
    pyrx_lines = convert_nlrx_to_pyrx(nlrx_lines)

    with open(os.path.join(os.path.dirname(__file__), "KB13_pyrx.txt"), "w") as f:
        f.write("\n".join(pyrx_lines))
        