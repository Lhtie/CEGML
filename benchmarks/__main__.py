"""`python -m benchmarks <name> [original training options]`."""

import sys

from .registry import BENCHMARKS, get_benchmark, run


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help", "--list"}:
        print("Usage: python -m benchmarks <name> [original training options]")
        print("Names: " + ", ".join(BENCHMARKS))
        return 0 if arguments else 2
    get_benchmark(arguments[0])
    run(arguments[0], arguments[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
