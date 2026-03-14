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
    top_n: int = 5,
) -> dict:
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
        "recent_incidents": incidents_df.sort_values("start_time", ascending=False).head(top_n).to_dict(orient="records"),
        "timeline_snapshot": incident_timeline_df.tail(top_n).to_dict(orient="records"),
    }
    return summary


def save_summary_json(summary: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, default=str)
