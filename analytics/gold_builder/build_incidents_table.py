from __future__ import annotations

import pandas as pd


INCIDENT_COLUMNS = [
    "incident_id",
    "device",
    "organization",
    "alert_count",
    "severity_score",
    "start_time",
    "end_time",
    "duration_minutes",
    "root_cause_candidate",
    "incident_cause",
    "description",
    "incident_type",
]


def build_incidents_gold(incidents_df: pd.DataFrame) -> pd.DataFrame:
    existing_columns = [col for col in INCIDENT_COLUMNS if col in incidents_df.columns]
    return incidents_df[existing_columns].copy().sort_values("start_time").reset_index(drop=True)
