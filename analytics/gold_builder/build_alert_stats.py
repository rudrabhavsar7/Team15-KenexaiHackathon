from __future__ import annotations

import pandas as pd


ALERT_STATS_COLUMNS = ["alert_type", "total_count", "devices_affected", "avg_severity"]


def build_alert_stats_gold(alerts_df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        alerts_df.groupby("alert_type", as_index=False)
        .agg(
            total_count=("alert_id", "count"),
            devices_affected=("device", "nunique"),
            avg_severity=("severity_weight", "mean"),
        )
        .sort_values("total_count", ascending=False)
        .reset_index(drop=True)
    )
    return stats[ALERT_STATS_COLUMNS]
