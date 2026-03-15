from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def create_incident_summary(
    incidents_df: pd.DataFrame,
    device_metrics_df: pd.DataFrame,
    alert_stats_df: pd.DataFrame,
    incident_timeline_df: pd.DataFrame,
    alerts_df: pd.DataFrame | None = None,
    top_n: int = 5,
) -> dict:
    top_incident_causes = []
    if not incidents_df.empty and "incident_cause" in incidents_df.columns:
        top_incident_causes = (
            incidents_df["incident_cause"]
            .astype(str)
            .value_counts()
            .head(top_n)
            .rename_axis("incident_cause")
            .reset_index(name="count")
            .to_dict(orient="records")
        )

    top_incident_descriptions = []
    if not incidents_df.empty and "description" in incidents_df.columns:
        top_incident_descriptions = (
            incidents_df["description"]
            .astype(str)
            .value_counts()
            .head(top_n)
            .rename_axis("incident_description")
            .reset_index(name="count")
            .to_dict(orient="records")
        )

    top_alert_contexts = []
    recent_alerts_with_context = []
    if alerts_df is not None and not alerts_df.empty:
        context_columns = ["alert_type", "description", "cause"]
        if all(column in alerts_df.columns for column in context_columns):
            top_alert_contexts = (
                alerts_df.groupby(context_columns, dropna=False)
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .head(top_n)
                .to_dict(orient="records")
            )

        candidate_recent_columns = [
            "alert_id",
            "timestamp",
            "source",
            "organization",
            "device",
            "alert_type",
            "severity",
            "description",
            "cause",
            "incident_id",
        ]
        recent_columns = [column for column in candidate_recent_columns if column in alerts_df.columns]
        if recent_columns:
            recent_alerts_with_context = (
                alerts_df.sort_values("timestamp", ascending=False).head(top_n)[recent_columns].to_dict(orient="records")
            )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overview": {
            "total_incidents": int(incidents_df["incident_id"].nunique()) if not incidents_df.empty else 0,
            "total_devices": int(device_metrics_df["device"].nunique()) if not device_metrics_df.empty else 0,
            "total_alerts": int(device_metrics_df["total_alerts"].sum()) if not device_metrics_df.empty else 0,
            "avg_reliability_score": float(device_metrics_df["reliability_score"].mean()) if not device_metrics_df.empty else 0.0,
        },
        "top_failing_devices": device_metrics_df.sort_values(
            ["incidents_count", "critical_alerts", "total_alerts"],
            ascending=False,
        )
        .head(top_n)
        .to_dict(orient="records"),
        "most_frequent_alert_types": alert_stats_df.head(top_n).to_dict(orient="records"),
        "top_incident_causes": top_incident_causes,
        "top_incident_descriptions": top_incident_descriptions,
        "top_alert_contexts": top_alert_contexts,
        "recent_alerts_with_context": recent_alerts_with_context,
        "recent_incidents": incidents_df.sort_values("start_time", ascending=False).head(top_n).to_dict(orient="records"),
        "timeline_snapshot": incident_timeline_df.tail(top_n).to_dict(orient="records"),
    }
    return summary


def save_summary_json(summary: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, default=str)
