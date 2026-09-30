"""Test suite minimization: the smallest set of databases that together
verify every rule verified by a run.

`minimum_cover` finds an exact minimum. It starts from the greedy solution
and explores smaller selections, pruning any partial selection that cannot
improve on the best one found so far. The search stops after `time_limit`
seconds, in which case the best selection found is returned and marked as
not proven optimal.
"""
import time


def greedy_cover(universe, sets):
    uncovered, chosen = set(universe), []
    while uncovered:
        best = max(sets, key=lambda i: (len(sets[i] & uncovered), -i), default=None)
        if best is None or not sets[best] & uncovered:
            break
        chosen.append(best)
        uncovered -= sets[best]
    return sorted(chosen)


def minimum_cover(universe, sets, time_limit=120.0):
    """Return (greedy selection, minimum selection, proven optimal).

    `universe` is the set of rules to cover and `sets` maps a database index
    to the set of rules that database verifies."""
    greedy = greedy_cover(universe, sets)
    universe = frozenset(universe)
    if not universe:
        return greedy, [], True
    first_of = {}
    for i in sorted(sets):
        s = frozenset(sets[i] & universe)
        if s:
            first_of.setdefault(s, i)
    items = [(s, i) for s, i in first_of.items() if not any(s < t for t in first_of)]
    covering = {r: [(s, i) for s, i in items if r in s] for r in universe}
    best = [list(greedy)]
    deadline = time.time() + time_limit
    timed_out = [False]

    def search(uncovered, chosen):
        if timed_out[0] or time.time() > deadline:
            timed_out[0] = True
            return
        if not uncovered:
            if len(chosen) < len(best[0]):
                best[0] = list(chosen)
            return
        gain = max(len(s & uncovered) for s, _ in items)
        if len(chosen) + -(-len(uncovered) // gain) >= len(best[0]):
            return
        r = min(uncovered, key=lambda x: len(covering[x]))
        for s, i in sorted(covering[r], key=lambda si: -len(si[0] & uncovered)):
            chosen.append(i)
            search(uncovered - s, chosen)
            chosen.pop()

    search(universe, [])
    return greedy, sorted(best[0]), not timed_out[0]
