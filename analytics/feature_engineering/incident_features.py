from __future__ import annotations

import pandas as pd


def engineer_incident_features(incidents_df: pd.DataFrame, alerts_df: pd.DataFrame) -> pd.DataFrame:
    incidents = incidents_df.copy()
    if incidents.empty:
        incidents["incident_duration"] = pd.Series(dtype="float64")
        incidents["normalized_severity_score"] = pd.Series(dtype="float64")
        incidents["alerts_per_minute"] = pd.Series(dtype="float64")
        return incidents

    incidents["incident_duration"] = incidents["duration_minutes"].clip(lower=0)
    max_score = max(float(incidents["severity_score"].max()), 1.0)
    incidents["normalized_severity_score"] = incidents["severity_score"] / max_score
    incidents["alerts_per_minute"] = incidents["alert_count"] / incidents["incident_duration"].clip(lower=1)

    if "incident_id" in alerts_df.columns:
        status_counts = (
            alerts_df.assign(is_open=lambda df: (df["status"].astype(str).str.lower() != "resolved").astype(int))
            .groupby("incident_id", as_index=False)
            .agg(open_alert_ratio=("is_open", "mean"))
        )
        incidents = incidents.merge(status_counts, on="incident_id", how="left")
        incidents["open_alert_ratio"] = incidents["open_alert_ratio"].fillna(0)
    else:
        incidents["open_alert_ratio"] = 0.0

    return incidents
