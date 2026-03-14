from __future__ import annotations

import numpy as np
import pandas as pd


SEVERITY_MAP = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
    "info": 1,
}


def engineer_alert_features(alerts_df: pd.DataFrame) -> pd.DataFrame:
    alerts = alerts_df.copy()
    alerts["timestamp"] = pd.to_datetime(alerts["timestamp"], errors="coerce", utc=True)
    alerts = alerts.dropna(subset=["timestamp", "device", "organization"]).sort_values("timestamp")
    alerts["severity"] = alerts["severity"].astype(str).str.lower().str.strip()
    alerts["severity_weight"] = alerts["severity"].map(SEVERITY_MAP).fillna(1).astype(float)

    alerts["alerts_per_device"] = alerts.groupby(["organization", "device"])["alert_id"].transform("count")

    first_last = alerts.groupby(["organization", "device"]).agg(
        first_ts=("timestamp", "min"),
        last_ts=("timestamp", "max"),
        total_alerts=("alert_id", "count"),
    )
    active_hours = (first_last["last_ts"] - first_last["first_ts"]).dt.total_seconds().div(3600).clip(lower=1 / 60)
    first_last["alert_frequency"] = first_last["total_alerts"] / active_hours

    alerts = alerts.merge(
        first_last[["alert_frequency"]].reset_index(),
        on=["organization", "device"],
        how="left",
    )

    alerts["time_since_prev_alert_min"] = (
        alerts.groupby(["organization", "device"])["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60)
        .fillna(0)
    )
    return alerts.reset_index(drop=True)
