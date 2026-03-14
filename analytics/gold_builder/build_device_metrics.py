from __future__ import annotations

import pandas as pd

from analytics.analytics_inner.reliability_scores import compute_reliability_scores
from analytics.feature_engineering.device_features import build_device_feature_table


DEVICE_COLUMNS = [
    "device",
    "organization",
    "total_alerts",
    "critical_alerts",
    "incidents_count",
    "reliability_score",
]


def build_device_metrics_gold(alerts_df: pd.DataFrame, incidents_df: pd.DataFrame) -> pd.DataFrame:
    base = build_device_feature_table(alerts_df, incidents_df)
    scored = compute_reliability_scores(base)
    existing_columns = [col for col in DEVICE_COLUMNS if col in scored.columns]
    ordered = scored[existing_columns].copy()
    return ordered.sort_values(["organization", "device"]).reset_index(drop=True)
