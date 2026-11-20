"""Round budgets shared by regex and benchmark runners."""
from math import exp, log


def round_schedule(sizes, rounds=10):
    """Fit geometric accumulation from the initial size to the full budget.

    The final round reaches the available sample budget. Small pools may
    repeat integer sizes; this never creates additional training entries.
    Zero retains the original data-driven geometric schedule.
    """
    sizes = list(sizes)
    if rounds is None or rounds == 0:
        return sizes
    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    if not sizes:
        return []
    total = sizes[-1]
    if rounds == 1:
        return [total]
    start = min(sizes[0], total)
    if start <= 0:
        return [total] * rounds
    result = [
        min(
            total,
            max(
                start,
                round(exp(log(start) + (log(total) - log(start)) * i / (rounds - 1))),
            ),
        )
        for i in range(rounds)
    ]
    result[0] = start
    result[-1] = total
    return result
