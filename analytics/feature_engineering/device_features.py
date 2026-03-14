from __future__ import annotations

import pandas as pd


def build_device_feature_table(alerts_df: pd.DataFrame, incidents_df: pd.DataFrame) -> pd.DataFrame:
    alerts = alerts_df.copy()
    if "alert_frequency" not in alerts.columns:
        alerts["alert_frequency"] = 0.0

    alert_metrics = (
        alerts.assign(is_critical=lambda df: (df["severity"].astype(str).str.lower() == "critical").astype(int))
        .groupby(["organization", "device"], as_index=False)
        .agg(
            total_alerts=("alert_id", "count"),
            critical_alerts=("is_critical", "sum"),
            avg_severity=("severity_weight", "mean"),
            alert_frequency=("alert_frequency", "mean"),
        )
    )

    incident_metrics = (
        incidents_df.groupby(["organization", "device"], as_index=False)
        .agg(
            incidents_count=("incident_id", "nunique"),
            avg_incident_duration=("duration_minutes", "mean"),
            avg_incident_severity=("severity_score", "mean"),
        )
        if not incidents_df.empty
        else pd.DataFrame(
            columns=[
                "organization",
                "device",
                "incidents_count",
                "avg_incident_duration",
                "avg_incident_severity",
            ]
        )
    )

    device_features = alert_metrics.merge(
        incident_metrics,
        on=["organization", "device"],
        how="left",
    )
    return device_features.fillna(0)
