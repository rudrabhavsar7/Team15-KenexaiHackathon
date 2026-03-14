from __future__ import annotations

import pandas as pd


def calculate_pipeline_metrics(
    alerts_df: pd.DataFrame,
    incidents_df: pd.DataFrame,
    clustered_alerts_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    total_alerts = float(len(alerts_df))
    total_incidents = float(incidents_df["incident_id"].nunique()) if not incidents_df.empty else 0.0
    covered_alerts = float(alerts_df["incident_id"].notna().sum()) if "incident_id" in alerts_df.columns else 0.0

    metrics = {
        "total_alerts": total_alerts,
        "total_incidents": total_incidents,
        "alerts_per_incident": (total_alerts / total_incidents) if total_incidents > 0 else 0.0,
        "incident_coverage_ratio": (covered_alerts / total_alerts) if total_alerts > 0 else 0.0,
        "avg_incident_duration_minutes": float(incidents_df["duration_minutes"].mean()) if not incidents_df.empty else 0.0,
    }

    if clustered_alerts_df is not None and "alert_cluster" in clustered_alerts_df.columns:
        non_noise = (clustered_alerts_df["alert_cluster"] != -1).sum()
        metrics["clustered_alert_ratio"] = float(non_noise) / total_alerts if total_alerts > 0 else 0.0

    return metrics
