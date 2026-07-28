#!/usr/bin/env python3
"""Count JSON files whose run-0 final accuracy is 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RUN_KEYS = ("run-0", "rerun-0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively scan a logs directory and count JSON files where "
            "summary.run-0.final_accuracy, or summary.rerun-0.final_accuracy, is 1."
        )
    )
    parser.add_argument("logs_dir", type=Path, help="Directory containing JSON logs.")
    parser.add_argument(
        "--list-success",
        action="store_true",
        help="Print files whose final_accuracy is 1.",
    )
    parser.add_argument(
        "--list-failed",
        action="store_true",
        help="Print valid files whose final_accuracy is not 1.",
    )
    parser.add_argument(
        "--list-invalid",
        action="store_true",
        help="Print JSON files missing a usable summary run or final_accuracy.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write the detailed classification as JSON.",
    )
    return parser.parse_args()


def get_run0_summary(data: dict[str, Any]) -> dict[str, Any] | None:
    summary = data.get("summary")
    if not isinstance(summary, dict):
        return None
    for key in RUN_KEYS:
        run = summary.get(key)
        if isinstance(run, dict):
            return run
    return None


def classify(path: Path) -> tuple[str, float | None, str | None]:
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        return "invalid", None, f"cannot read JSON: {exc}"

    if not isinstance(data, dict):
        return "invalid", None, "top-level JSON is not an object"

    run = get_run0_summary(data)
    if run is None:
        return "invalid", None, "missing summary.run-0 or summary.rerun-0"

    final_accuracy = run.get("final_accuracy")
    if not isinstance(final_accuracy, (int, float)) or isinstance(final_accuracy, bool):
        return "invalid", None, "missing numeric final_accuracy"

    return ("success" if final_accuracy == 1 else "failed"), float(final_accuracy), None


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    logs_dir = args.logs_dir
    if not logs_dir.is_dir():
        raise SystemExit(f"Not a directory: {logs_dir}")

    json_files = sorted(logs_dir.rglob("*.json"))
    success: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []

    for path in json_files:
        status, final_accuracy, reason = classify(path)
        entry = {
            "path": str(path),
            "relative_path": rel(path, logs_dir),
            "final_accuracy": final_accuracy,
        }
        if status == "success":
            success.append(entry)
        elif status == "failed":
            failed.append(entry)
        else:
            entry["reason"] = reason
            invalid.append(entry)

    valid_count = len(success) + len(failed)
    success_rate_valid = len(success) / valid_count if valid_count else 0.0
    success_rate_all = len(success) / len(json_files) if json_files else 0.0

    print(f"Directory: {logs_dir}")
    print(f"JSON files: {len(json_files)}")
    print(f"Valid run-0 summaries: {valid_count}")
    print(f"Invalid or missing summaries: {len(invalid)}")
    print(f"Final accuracy = 1: {len(success)}")
    print(
        f"Success among valid summaries: "
        f"{len(success)}/{valid_count} ({100 * success_rate_valid:.2f}%)"
    )
    print(
        f"Success among all JSON files: "
        f"{len(success)}/{len(json_files)} ({100 * success_rate_all:.2f}%)"
    )

    if args.list_success:
        print("\nSuccess files:")
        for item in success:
            print(item["relative_path"])

    if args.list_failed:
        print("\nFailed files:")
        for item in failed:
            print(f"{item['relative_path']}\tfinal_accuracy={item['final_accuracy']}")

    if args.list_invalid:
        print("\nInvalid files:")
        for item in invalid:
            print(f"{item['relative_path']}\t{item['reason']}")

    if args.output_json:
        output = {
            "directory": str(logs_dir),
            "num_json_files": len(json_files),
            "num_valid": valid_count,
            "num_invalid": len(invalid),
            "num_success": len(success),
            "success_rate_valid": success_rate_valid,
            "success_rate_all": success_rate_all,
            "success": success,
            "failed": failed,
            "invalid": invalid,
        }
        args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
