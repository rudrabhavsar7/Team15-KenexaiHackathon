from __future__ import annotations

import pandas as pd


def build_incident_statistics(alerts_df: pd.DataFrame, incidents_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    top_failing_devices = (
        alerts_df.assign(is_critical=lambda df: (df["severity"].astype(str).str.lower() == "critical").astype(int))
        .groupby(["organization", "device"], as_index=False)
        .agg(total_alerts=("alert_id", "count"), critical_alerts=("is_critical", "sum"))
        .merge(
            incidents_df.groupby(["organization", "device"], as_index=False)
            .agg(incidents_count=("incident_id", "nunique")),
            on=["organization", "device"],
            how="left",
        )
        .fillna({"incidents_count": 0})
        .sort_values(["incidents_count", "critical_alerts", "total_alerts"], ascending=False)
        .reset_index(drop=True)
    )

    most_frequent_alert_types = (
        alerts_df.groupby("alert_type", as_index=False)
        .agg(total_count=("alert_id", "count"), devices_affected=("device", "nunique"))
        .sort_values("total_count", ascending=False)
        .reset_index(drop=True)
    )

    severity_distribution = (
        alerts_df["severity"].astype(str).str.lower().value_counts(dropna=False).rename_axis("severity").reset_index(name="count")
    )

    incident_frequency_trends = (
        incidents_df.assign(date=pd.to_datetime(incidents_df["start_time"], utc=True).dt.date)
        .groupby("date", as_index=False)
        .agg(
            incidents_count=("incident_id", "nunique"),
            avg_severity=("severity_score", "mean"),
            avg_duration_minutes=("duration_minutes", "mean"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return {
        "top_failing_devices": top_failing_devices,
        "most_frequent_alert_types": most_frequent_alert_types,
        "severity_distribution": severity_distribution,
        "incident_frequency_trends": incident_frequency_trends,
    }
