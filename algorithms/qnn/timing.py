def aggregate_simulation_seconds(
    per_individual_seconds: list, parallelize: bool = False
) -> float:
    if not per_individual_seconds:
        return 0.0
    return max(per_individual_seconds) if parallelize else sum(per_individual_seconds)