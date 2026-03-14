from __future__ import annotations

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def prepare_alert_clustering_features(alerts_df: pd.DataFrame) -> pd.DataFrame:
    prepared = alerts_df.copy()
    prepared["severity_code"] = prepared["severity"].astype("category").cat.codes
    prepared["source_code"] = prepared["source"].astype("category").cat.codes
    prepared["alert_type_code"] = prepared["alert_type"].astype("category").cat.codes
    prepared["entity_type_code"] = prepared["entity_type"].astype("category").cat.codes
    prepared["timestamp_epoch"] = pd.to_datetime(prepared["timestamp"], utc=True).astype("int64") // 10**9
    feature_columns = [
        "severity_weight",
        "severity_code",
        "source_code",
        "alert_type_code",
        "entity_type_code",
        "timestamp_epoch",
    ]
    return prepared[feature_columns].fillna(0)


def train_dbscan_alert_clusters(
    alerts_df: pd.DataFrame,
    eps: float = 0.8,
    min_samples: int = 3,
) -> pd.DataFrame:
    if alerts_df.empty:
        output = alerts_df.copy()
        output["alert_cluster"] = pd.Series(dtype="int64")
        return output

    output = alerts_df.copy()
    features = prepare_alert_clustering_features(output)
    scaled = StandardScaler().fit_transform(features)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    output["alert_cluster"] = model.fit_predict(scaled).astype(int)
    return output
