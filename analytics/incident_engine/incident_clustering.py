from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def cluster_incidents_dbscan(
    incidents_df: pd.DataFrame,
    eps: float = 0.9,
    min_samples: int = 2,
) -> pd.DataFrame:
    if incidents_df.empty:
        enriched = incidents_df.copy()
        enriched["cluster_id"] = pd.Series(dtype="int64")
        return enriched

    clustered = incidents_df.copy()
    feature_frame = clustered[["alert_count", "severity_score", "duration_minutes"]].fillna(0.0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_frame)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(scaled)
    clustered["cluster_id"] = labels.astype(int)
    clustered["is_noise_cluster"] = (clustered["cluster_id"] == -1).astype(int)
    return clustered


def summarize_clusters(incidents_df: pd.DataFrame) -> pd.DataFrame:
    if incidents_df.empty or "cluster_id" not in incidents_df.columns:
        return pd.DataFrame(columns=["cluster_id", "incidents", "avg_severity_score", "avg_alert_count"])

    valid = incidents_df[incidents_df["cluster_id"] != -1]
    if valid.empty:
        return pd.DataFrame(columns=["cluster_id", "incidents", "avg_severity_score", "avg_alert_count"])

    summary = (
        valid.groupby("cluster_id", as_index=False)
        .agg(
            incidents=("incident_id", "nunique"),
            avg_severity_score=("severity_score", "mean"),
            avg_alert_count=("alert_count", "mean"),
        )
        .sort_values("incidents", ascending=False)
        .reset_index(drop=True)
    )
    return summary
