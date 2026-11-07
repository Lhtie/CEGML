#!/usr/bin/env bash
set -euo pipefail

# Usage: bash datasets/benchmarks/run_gpt_oss_120b.sh [icl|icl_gen]
# The default runner is direct input/output ICL. Each invocation runs 8 jobs:
# four benchmark IDs, each in standard and counterexample modes.

runner=${1:-icl}
if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [icl|icl_gen]" >&2
    exit 2
fi
case "$runner" in
    icl) runner_script=train_icl.py ;;
    icl_gen) runner_script=train_icl_gen.py ;;
    *) echo "Runner must be icl or icl_gen" >&2; exit 2 ;;
esac

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

python_bin=${PYTHON_BIN:-python3}
outdir=${OUTDIR:-logs/benchmarks}

run_pair() {
    local task_type=$1
    local benchmark_id=$2
    local common=(
        --task_type "$task_type"
        --benchmark_id "$benchmark_id"
        --mkey gpt-oss
        --use_reg
        --outdir "$outdir"
    )
    if [[ "$runner" == icl_gen ]]; then
        common+=(--rerun 1)
    fi

    echo "[$runner] $task_type / $benchmark_id / standard"
    "$python_bin" "$runner_script" "${common[@]}"

    echo "[$runner] $task_type / $benchmark_id / counterexamples"
    "$python_bin" "$runner_script" "${common[@]}" --use_ce --ce_batch_size 8
}

run_pair cryptography caesar:0
run_pair list_functions c102
run_pair instruction_induction antonyms
run_pair scan simple
