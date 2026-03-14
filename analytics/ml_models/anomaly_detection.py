from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_device_anomalies(
    device_metrics_df: pd.DataFrame,
    contamination: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    if device_metrics_df.empty:
        out = device_metrics_df.copy()
        out["anomaly_score"] = pd.Series(dtype="float64")
        out["is_anomalous"] = pd.Series(dtype="int64")
        return out

    model_input_columns = ["total_alerts", "critical_alerts", "incidents_count", "reliability_score"]
    out = device_metrics_df.copy()
    feature_frame = out[model_input_columns].fillna(0)

    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(feature_frame)
    out["anomaly_score"] = model.decision_function(feature_frame)
    out["is_anomalous"] = (model.predict(feature_frame) == -1).astype(int)
    return out
