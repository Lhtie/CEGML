import sys
import os
import json
import re

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from tasks.rl import PythonRegularLanguage

def split_regex_into_atoms(regex):
    """
    Simple atom splitter.
    - Treat (), [] blocks as atoms (supports nesting for () only).
    - Treat escapes like "\\b" as atoms.
    - Attach quantifiers (* + ? {m,n}) to the previous atom.
    - Operators "|", "&", "~" are standalone.
    """
    atoms, i, n = [], 0, len(regex)

    def take_quant(i0):
        if i0 < n and regex[i0] in {"*", "+", "?"}:
            return regex[i0], i0 + 1
        if i0 < n and regex[i0] == "{":
            j = i0 + 1
            while j < n and regex[j] != "}":
                j += 1
            return regex[i0:j + 1], min(j + 1, n)
        return "", i0

    while i < n:
        ch = regex[i]
        if ch in {"|", "&", "~"}:
            atoms.append(ch)
            i += 1
            continue

        if ch == "[":
            j = i + 1
            while j < n:
                if regex[j] == "\\" and j + 1 < n:
                    j += 2
                elif regex[j] == "]":
                    j += 1
                    break
                else:
                    j += 1
            atom = regex[i:j]
            q, i = take_quant(j)
            atoms.append(atom + q)
            continue

        if ch == "(":
            depth, j = 1, i + 1
            while j < n and depth > 0:
                if regex[j] == "\\" and j + 1 < n:
                    j += 2
                elif regex[j] == "(":
                    depth += 1
                    j += 1
                elif regex[j] == ")":
                    depth -= 1
                    j += 1
                else:
                    j += 1
            inner = regex[i + 1:j - 1]
            atoms.extend(split_regex_into_atoms(inner))
            q, i = take_quant(j)
            if q:
                atoms.append(q)
            continue

        if ch == "\\" and i + 1 < n:
            atom = regex[i:i + 2]
            q, i = take_quant(i + 2)
            atoms.append(atom + q)
            continue

        atom = ch
        q, i = take_quant(i + 1)
        atoms.append(atom + q)

    return atoms

def convert_nlrx_to_pyrx(nlrx_lines):
    alphabet = ["A-Z", "a-z", "0-9", "#"]
    
    pyrx_lines = []
    for line in nlrx_lines:
        if line.startswith("~"):
            match = re.search(r"~\((.*)\)", line)
            assert match is not None, f"Failed to eliminate overall ~: {line}"
            line = match.group(1)
        assert "~" not in line, f"Op ~ remains: {line}"
        
        if line.startswith("\.\*") and line.endswith("\.\*"):
            line = line[2:-2]
            
        line = line.replace("\\b", "#")
        
        while True:
            match = re.search(r"({[0-9]+,\s*})", line)
            if match is None: break
            group = match.group(1)
            

if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "KB13.txt"), "r") as f:
        nlrx_lines = f.read().splitlines()
        nlrx_lines = [line.strip() for line in nlrx_lines]
    
    pyrx_lines = convert_nlrx_to_pyrx(nlrx_lines)
