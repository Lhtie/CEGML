#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
JSON_FILE="$SCRIPT_DIR/regex_list.json"

# Emit first regex from each non-empty regex_list as NUL-delimited strings.
extract_first_regexes() {
  python3 - "$JSON_FILE" <<'PY'
import json, sys
path = sys.argv[1]
data = json.load(open(path, "r", encoding="utf-8"))
for state_block in data:
    for depth_block in state_block.get("regex_list", []):
        lst = depth_block.get("regex_list", [])
        if lst:
            sys.stdout.write(lst[0] + "\0")
PY
}

while IFS= read -r -d '' r; do
  echo "Running regex: $r"
  python3 "$REPO_ROOT/dataset.py" \
    --regex "$r" \
    --task_type "simplyrx" \
    --max_length 32 \
    --eval_max_length 32 \
    --tot_train_size 3000 \
    --eval_size 32 \
    --seed 42 \
    --outdir "$SCRIPT_DIR/regex_datasets"
done < <(extract_first_regexes)
