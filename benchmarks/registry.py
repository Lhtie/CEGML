"""Registry for the registered data-backed program-induction benchmarks."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    oracle: str

    def load_task_class(self):
        return importlib.import_module("tasks.benchmarks").BenchmarkTask

    def load_prompts(self):
        return importlib.import_module("prompting.benchmarks")

    def load_runner(self):
        return importlib.import_module("train_icl_gen").main


@dataclass(frozen=True)
class CounterexampleBatch:
    examples: list[Any]
    labels: list[Any]
    equivalent: bool | None = None


BENCHMARKS = {
    "ilp": BenchmarkSpec("ilp", "finite Horn forward chaining"),
    "cryptography": BenchmarkSpec("cryptography", "cvc5 bounded symbolic Python"),
    "list_functions": BenchmarkSpec("list_functions", "cvc5 sequence and bounded search"),
    "instruction_induction": BenchmarkSpec("instruction_induction", "LLM judge"),
    "scan": BenchmarkSpec("scan", "exhaustive enumeration"),
}


def get_benchmark(name: str) -> BenchmarkSpec:
    try:
        return BENCHMARKS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark {name!r}; choose from {', '.join(BENCHMARKS)}") from exc


def create_task(name: str, benchmark_id: str, **kwargs: Any):
    """Construct one of the registered data-backed benchmark tasks."""
    spec = get_benchmark(name)
    return spec.load_task_class()(spec.name, benchmark_id, **kwargs)


class BenchmarkHandle:
    def __init__(self, spec: BenchmarkSpec, task: Any):
        self.spec = spec
        self.task = task

    def counterexamples(self, hypothesis: str, limit: int = 8, **oracle_options: Any) -> CounterexampleBatch:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        verdict = self.task.verify(hypothesis, limit=limit, **oracle_options)
        if verdict.error:
            raise RuntimeError(verdict.error)
        return CounterexampleBatch(
            [x for x, _ in verdict.counterexamples], [y for _, y in verdict.counterexamples],
            verdict.equivalent,
        )


def open_benchmark(name: str, benchmark_id: str, **kwargs: Any) -> BenchmarkHandle:
    spec = get_benchmark(name)
    return BenchmarkHandle(spec, create_task(spec.name, benchmark_id, **kwargs))


def run(name: str, argv: list[str] | None = None):
    """Forward options to the benchmark branch of train_icl_gen.main."""
    spec = get_benchmark(name)
    arguments = list(argv or [])
    if "--task_type" in arguments or any(arg.startswith("--task_type=") for arg in arguments):
        raise ValueError("Pass task_type once, as the benchmark name")
    return spec.load_runner()(["--task_type", spec.name, *arguments])
