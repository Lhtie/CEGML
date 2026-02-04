import json
import re

def expand_char_class(s):
    # s the same form as "[A-Za-z0-9#]"
    assert s.startswith("[") and s.endswith("]")
    body = s[1:-1]
    out = []
    i = 0
    while i < len(body):
        if i + 2 < len(body) and body[i+1] == "-" and body[i].isalnum() and body[i+2].isalnum():
            start, end = body[i], body[i+2]
            out.extend(chr(c) for c in range(ord(start), ord(end) + 1))
            i += 3
        else:
            out.append(body[i])
            i += 1
    return out

def split_regex_into_atoms(regex):
    """
    Parse regex into a nesting tree by ().
    Returns a dict node:
      - group: {"type": "group", "children": [...], "ops": [...], "quant": "", "negated": False}
      - atom:  {"type": "atom", "value": "...", "quant": "", "negated": False}
    Operators "|", "&" are recorded between children in "ops".
    Unary "~" is recorded as node["negated"] = True on the following atom/group.
    Quantifiers (* + ? {m,n}) are attached to the preceding atom/group node.
    """
    def take_quant(s, i0):
        sn = len(s)
        if i0 < sn and s[i0] in {"*", "+", "?"}:
            return s[i0], i0 + 1
        if i0 < sn and s[i0] == "{":
            j = i0 + 1
            while j < sn and s[j] != "}":
                j += 1
            return s[i0:j + 1], min(j + 1, sn)
        return "", i0

    def parse_group(s):
        children, ops = [], []
        i1, sn = 0, len(s)
        pending_neg = False
        while i1 < sn:
            ch = s[i1]
            if ch in {"|", "&"}:
                ops.append(ch)
                i1 += 1
                continue
            if ch == "~":
                pending_neg = not pending_neg
                i1 += 1
                continue

            if ch == "(":
                depth, j = 1, i1 + 1
                while j < sn and depth > 0:
                    if s[j] == "\\" and j + 1 < sn: j += 2
                    else:
                        depth = depth + (s[j] == "(") - (s[j] == ")")
                        j += 1
                inner = s[i1 + 1:j - 1]
                sub_children, sub_ops = parse_group(inner)
                q, i1 = take_quant(s, j)
                if children and len(ops) < len(children):
                    ops.append(".")
                node = {"type": "group", "children": sub_children, "ops": sub_ops, "quant": q, "negated": False}
                if pending_neg:
                    node["negated"] = True
                    pending_neg = False
                children.append(node)
                continue

            if ch == "[":
                j = i1 + 1
                while j < sn:
                    if s[j] == "\\" and j + 1 < sn:
                        j += 2
                    else: 
                        j += 1
                        if s[j - 1] == "]": break
            elif ch == "\\" and i1 + 1 < sn:
                j = i1 + 2
            else:
                j = i1 + 1
            atom = s[i1:j]
            q, i1 = take_quant(s, j)
            if children and len(ops) < len(children):
                ops.append(".")
            node = {"type": "atom", "value": atom, "quant": q, "negated": False}
            if pending_neg:
                node["negated"] = True
                pending_neg = False
            children.append(node)

        return children, ops

    children, ops = parse_group(regex)
    return {"type": "group", "children": children, "ops": ops, "quant": "", "negated": False}
