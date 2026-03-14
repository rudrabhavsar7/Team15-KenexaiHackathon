from __future__ import annotations

import pandas as pd


def compute_reliability_scores(device_metrics_df: pd.DataFrame) -> pd.DataFrame:
    if device_metrics_df.empty:
        output = device_metrics_df.copy()
        output["reliability_score"] = pd.Series(dtype="float64")
        return output

    output = device_metrics_df.copy()

    max_alerts = max(float(output["total_alerts"].max()), 1.0)
    max_critical = max(float(output["critical_alerts"].max()), 1.0)
    max_incidents = max(float(output["incidents_count"].max()), 1.0)

    penalty = (
        0.45 * (output["total_alerts"] / max_alerts)
        + 0.35 * (output["critical_alerts"] / max_critical)
        + 0.20 * (output["incidents_count"] / max_incidents)
    )
    output["reliability_score"] = (100 * (1 - penalty)).clip(lower=0, upper=100).round(2)
    return output
