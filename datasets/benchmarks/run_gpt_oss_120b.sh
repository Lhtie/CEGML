#!/usr/bin/env bash
set -euo pipefail

# Existing smoke suite (8 jobs):
#   bash datasets/benchmarks/run_gpt_oss_120b.sh [icl|icl_gen]
# Every available ID/split (1164 jobs per runner):
#   python -m benchmarks.run_suite --runner icl_gen --suite all
# Preview, filter, or pass runner options:
#   python -m benchmarks.run_suite --runner icl_gen --suite all --dry-run
#   python -m benchmarks.run_suite --runner icl_gen --suite all --task list_functions -- --retries 3
# Smaller stratified suite (136 jobs per runner):
#   python -m benchmarks.run_suite --runner icl_gen --suite balanced
# Use runner "both" for both direct ICL and rule generation.
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
exec "${PYTHON_BIN:-python3}" -m benchmarks.run_suite "$@"
