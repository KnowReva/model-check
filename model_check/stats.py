"""How sure can you be? Two small pieces of statistics that keep a test on 20 or 50 examples honest.

wilson()   the range a model's real accuracy probably lies in, from k right out of n
           (the Wilson score interval, 95%). 18 out of 20 is "somewhere from 70% to 97%", not "90%".
mcnemar()  whether one model is really better than another on the same examples: only the examples
           where exactly one of them is right count (McNemar's exact test).
"""

import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """(low, high) of the 95% Wilson score interval for k successes out of n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar(only_a: int, only_b: int) -> float:
    """The chance (p-value, two-sided) of a split at least this uneven if both models were equally good:
    only_a = examples A got right and B wrong, only_b = the other way round."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def median(values: list[float]) -> float:
    s = sorted(values)
    if not s:
        return 0.0
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
