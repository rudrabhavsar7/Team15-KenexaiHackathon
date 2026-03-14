from __future__ import annotations

import pandas as pd

from analytics.incident_engine.incident_clustering import cluster_incidents_dbscan
from analytics.incident_engine.incident_rules import detect_incident_groups, normalize_alerts


def build_incidents_from_alerts(
    alerts_df: pd.DataFrame,
    window: str = "5min",
    eps: float = 0.9,
    min_samples: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized = normalize_alerts(alerts_df)
    incidents, alert_mapping = detect_incident_groups(normalized, window=window)
    incidents = cluster_incidents_dbscan(incidents, eps=eps, min_samples=min_samples)
    enriched_alerts = alert_mapping.merge(
        incidents[["incident_id", "cluster_id", "incident_type"]],
        on="incident_id",
        how="left",
    )
    return incidents, enriched_alerts
