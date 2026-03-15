from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = [
    "alert_id",
    "source",
    "timestamp",
    "organization",
    "device",
    "entity_type",
    "alert_type",
    "severity",
    "status",
    "description",
]

SEVERITY_MAP = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
    "info": 1,
}


def normalize_alerts(alerts_df: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS) - set(alerts_df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    normalized = alerts_df[REQUIRED_COLUMNS].copy()
    if "cause" in alerts_df.columns:
        normalized["cause"] = alerts_df["cause"]
    else:
        normalized["cause"] = alerts_df.get("description", "unknown")

    normalized["cause"] = normalized["cause"].fillna(normalized["description"]).astype(str)
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce", utc=True)
    normalized = normalized.dropna(subset=["timestamp", "device", "organization"])  # type: ignore[arg-type]
    normalized["severity"] = normalized["severity"].astype(str).str.lower().str.strip()
    normalized["severity_weight"] = normalized["severity"].map(SEVERITY_MAP).fillna(1).astype(float)
    normalized["is_critical"] = (normalized["severity"] == "critical").astype(int)
    return normalized.sort_values("timestamp").reset_index(drop=True)


def assign_time_window(alerts_df: pd.DataFrame, window: str = "5min") -> pd.DataFrame:
    framed = alerts_df.copy()
    framed["time_window"] = framed["timestamp"].dt.floor(window)
    return framed


def _mode_or_unknown(series: pd.Series, unknown: str = "unknown") -> str:
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return unknown
    return non_null.mode().iat[0]


def _classify_incident_type(alert_type: str, description: str) -> str:
    signal = f"{alert_type} {description}".lower()
    if any(token in signal for token in ["cpu", "memory", "disk", "resource"]):
        return "resource"
    if any(token in signal for token in ["offline", "down", "disconnect", "unreachable"]):
        return "availability"
    if any(token in signal for token in ["latency", "packet", "link", "interface", "network"]):
        return "network"
    if any(token in signal for token in ["auth", "login", "certificate", "security"]):
        return "security"
    return "other"


def detect_incident_groups(alerts_df: pd.DataFrame, window: str = "5min") -> tuple[pd.DataFrame, pd.DataFrame]:
    alerts = assign_time_window(alerts_df, window=window)
    group_cols = ["organization", "device", "time_window"]

    grouped = (
        alerts.groupby(group_cols, as_index=False)
        .agg(
            alert_count=("alert_id", "count"),
            severity_score=("severity_weight", "sum"),
            critical_alert_count=("is_critical", "sum"),
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
            root_cause_candidate=("alert_type", _mode_or_unknown),
            incident_cause=("cause", _mode_or_unknown),
            representative_description=("description", _mode_or_unknown),
        )
        .sort_values(["start_time", "organization", "device"])
        .reset_index(drop=True)
    )

    grouped["incident_id"] = [f"INC-{idx:06d}" for idx in range(1, len(grouped) + 1)]
    grouped["duration_minutes"] = (
        (grouped["end_time"] - grouped["start_time"]).dt.total_seconds().div(60).clip(lower=0)
    )
    grouped["incident_type"] = grouped.apply(
        lambda row: _classify_incident_type(
            str(row["root_cause_candidate"]), str(row["representative_description"])
        ),
        axis=1,
    )

    mapping = alerts.merge(
        grouped[group_cols + ["incident_id"]],
        on=group_cols,
        how="left",
    )

    grouped = grouped.rename(columns={"representative_description": "description"})
    incidents = grouped.drop(columns=["time_window"])
    return incidents, mapping
