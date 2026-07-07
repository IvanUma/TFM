from __future__ import annotations

from typing import Iterable


def aggregate_simulation_seconds(
    simulation_times: Iterable[float], parallelize: bool = False
) -> float:

    values = list(simulation_times)
    if not values:
        return 0.0

    if parallelize and len(values) > 1:
        return max(values)

    return sum(values)
