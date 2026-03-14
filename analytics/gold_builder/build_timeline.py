from __future__ import annotations

import pandas as pd


TIMELINE_COLUMNS = ["timestamp", "alerts_count", "incidents_count", "critical_incidents"]


def build_incident_timeline_gold(
    alerts_df: pd.DataFrame,
    incidents_df: pd.DataFrame,
    freq: str = "5min",
) -> pd.DataFrame:
    alert_timeline = (
        alerts_df.assign(timestamp=lambda df: pd.to_datetime(df["timestamp"], utc=True).dt.floor(freq))
        .groupby("timestamp", as_index=False)
        .agg(alerts_count=("alert_id", "count"))
    )

    incident_timeline = (
        incidents_df.assign(timestamp=lambda df: pd.to_datetime(df["start_time"], utc=True).dt.floor(freq))
        .groupby("timestamp", as_index=False)
        .agg(
            incidents_count=("incident_id", "nunique"),
            critical_incidents=("critical_alert_count", lambda s: (s > 0).sum()),
        )
    )

    timeline = (
        alert_timeline.merge(incident_timeline, on="timestamp", how="outer")
        .fillna(0)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    timeline["alerts_count"] = timeline["alerts_count"].astype(int)
    timeline["incidents_count"] = timeline["incidents_count"].astype(int)
    timeline["critical_incidents"] = timeline["critical_incidents"].astype(int)
    return timeline[TIMELINE_COLUMNS]
