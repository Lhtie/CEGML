"""Entry point for the four added program-induction benchmarks."""

from .registry import (BENCHMARKS, BenchmarkHandle, BenchmarkSpec,
                       CounterexampleBatch, create_task, get_benchmark,
                       open_benchmark, run)

__all__ = ["BENCHMARKS", "BenchmarkHandle", "BenchmarkSpec", "CounterexampleBatch",
           "create_task", "get_benchmark", "open_benchmark", "run"]
