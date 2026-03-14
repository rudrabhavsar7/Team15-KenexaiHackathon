from __future__ import annotations

import pandas as pd


def detect_failure_patterns(
    alerts_df: pd.DataFrame,
    incidents_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    if alerts_df.empty:
        return pd.DataFrame(
            columns=[
                "organization",
                "device",
                "alert_type",
                "total_alerts",
                "incidents_impacted",
                "avg_severity",
            ]
        )

    aggregated = (
        alerts_df.groupby(["organization", "device", "alert_type"], as_index=False)
        .agg(
            total_alerts=("alert_id", "count"),
            incidents_impacted=("incident_id", "nunique"),
            avg_severity=("severity_weight", "mean"),
        )
        .sort_values(["incidents_impacted", "total_alerts", "avg_severity"], ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return aggregated
