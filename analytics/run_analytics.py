from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from analytics.analytics_inner.failure_patterns import detect_failure_patterns
from analytics.analytics_inner.incident_statistics import build_incident_statistics
from analytics.feature_engineering.alert_features import engineer_alert_features
from analytics.feature_engineering.incident_features import engineer_incident_features
from analytics.gold_builder.build_alert_stats import build_alert_stats_gold
from analytics.gold_builder.build_device_metrics import build_device_metrics_gold
from analytics.gold_builder.build_incidents_table import build_incidents_gold
from analytics.gold_builder.build_timeline import build_incident_timeline_gold
from analytics.incident_engine.incident_builder import build_incidents_from_alerts
from analytics.ml_models.anomaly_detection import detect_device_anomalies
from analytics.ml_models.clustering_model import train_dbscan_alert_clusters
from analytics.reports.evaluation_metrics import calculate_pipeline_metrics
from analytics.reports.incident_reports import create_incident_summary, save_summary_json


CANONICAL_ALERT_COLUMNS = [
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
    "cause",
]


def _resolve_default_silver_path() -> str:
    candidates = [
        Path("data/parquet_exports/alerts_clean.parquet"),
        Path("data/parquet_exports/alert_clen.parquet"),
        Path("silver/alerts_clean.parquet"),
        Path("data/alerts_clean.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "silver/alerts_clean.parquet"


def _normalize_silver_schema(alerts_df: pd.DataFrame) -> pd.DataFrame:
    normalized = alerts_df.copy()

    rename_map = {
        "source_system": "source",
        "occurred_at": "timestamp",
        "organization_name": "organization",
        "device_name": "device",
        "alert_message": "description",
    }
    normalized = normalized.rename(columns=rename_map)

    cause_candidates = [
        "cause",
        "root_cause",
        "root_cause_candidate",
        "probable_cause",
        "failure_cause",
        "reason",
    ]
    if "cause" not in normalized.columns:
        selected_cause_column = next((column for column in cause_candidates if column in normalized.columns), None)
        if selected_cause_column is not None:
            normalized["cause"] = normalized[selected_cause_column]
        elif "description" in normalized.columns:
            normalized["cause"] = normalized["description"]
        else:
            normalized["cause"] = "unknown"

    if "entity_type" not in normalized.columns:
        if "alert_category" in normalized.columns:
            normalized["entity_type"] = normalized["alert_category"]
        elif "network_name" in normalized.columns:
            normalized["entity_type"] = normalized["network_name"]
        else:
            normalized["entity_type"] = "unknown"

    if "status" not in normalized.columns:
        normalized["status"] = "open"

    if "source" not in normalized.columns:
        normalized["source"] = "unknown"

    normalized["cause"] = normalized["cause"].fillna(normalized.get("description", "unknown")).astype(str)

    missing = [column for column in CANONICAL_ALERT_COLUMNS if column not in normalized.columns]
    if missing:
        raise ValueError(f"Unable to normalize Silver alerts. Missing columns: {missing}")

    return normalized[CANONICAL_ALERT_COLUMNS].copy()


def load_silver_alerts(silver_path: str | Path) -> pd.DataFrame:
    source_path = Path(silver_path)

    if source_path.is_dir():
        parquet_files = sorted(source_path.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found in directory: {source_path}")

        normalized_batches: list[pd.DataFrame] = []
        skipped_files: list[str] = []

        for parquet_file in parquet_files:
            try:
                batch_df = pd.read_parquet(parquet_file)
                normalized_batch = _normalize_silver_schema(batch_df)
                normalized_batch["_source_file"] = parquet_file.name
                normalized_batches.append(normalized_batch)
            except Exception:
                skipped_files.append(parquet_file.name)

        if not normalized_batches:
            raise ValueError(
                "No alert-like parquet files could be normalized from directory "
                f"{source_path}. Ensure at least one file matches the Silver alert schema."
            )

        combined = pd.concat(normalized_batches, ignore_index=True)
        combined = combined.drop_duplicates(subset=["alert_id", "timestamp", "device"], keep="first")

        if skipped_files:
            print(
                "Skipped non-alert parquet files:",
                ", ".join(skipped_files),
            )

        return combined.drop(columns=["_source_file"], errors="ignore")

    suffix = source_path.suffix.lower()
    if suffix == ".parquet":
        alerts_df = pd.read_parquet(source_path)
    elif suffix == ".csv":
        alerts_df = pd.read_csv(source_path)
    else:
        raise ValueError(f"Unsupported Silver file format: {suffix}. Use .parquet, .csv, or a directory")

    return _normalize_silver_schema(alerts_df)


def save_gold_outputs(
    output_dir: str | Path,
    incidents_gold: pd.DataFrame,
    device_metrics_gold: pd.DataFrame,
    alert_stats_gold: pd.DataFrame,
    timeline_gold: pd.DataFrame,
    incident_summary: dict,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    incidents_gold.to_parquet(destination / "incidents.parquet", index=False)
    device_metrics_gold.to_parquet(destination / "device_metrics.parquet", index=False)
    alert_stats_gold.to_parquet(destination / "alert_stats.parquet", index=False)
    timeline_gold.to_parquet(destination / "incident_timeline.parquet", index=False)
    save_summary_json(incident_summary, destination / "incident_summary.json")


def run_pipeline(
    silver_path: str | Path | None = None,
    output_dir: str | Path = "analytics/outputs",
) -> dict[str, object]:
    if silver_path is None:
        silver_path = _resolve_default_silver_path()

    print(f"Using Silver source: {silver_path}")
    silver_alerts = load_silver_alerts(silver_path)

    alerts_features = engineer_alert_features(silver_alerts)
    incidents, alerts_with_incident = build_incidents_from_alerts(alerts_features, window="5min")
    alerts_clustered = train_dbscan_alert_clusters(alerts_with_incident)
    incidents_enriched = engineer_incident_features(incidents, alerts_clustered)

    incidents_gold = build_incidents_gold(incidents_enriched)
    device_metrics_gold = build_device_metrics_gold(alerts_clustered, incidents_enriched)
    device_metrics_gold = detect_device_anomalies(device_metrics_gold)
    alert_stats_gold = build_alert_stats_gold(alerts_clustered)
    timeline_gold = build_incident_timeline_gold(alerts_clustered, incidents_enriched, freq="5min")

    incident_statistics = build_incident_statistics(alerts_clustered, incidents_enriched)
    failure_patterns = detect_failure_patterns(alerts_clustered, incidents_enriched)
    pipeline_metrics = calculate_pipeline_metrics(alerts_clustered, incidents_enriched, alerts_clustered)

    incident_summary = create_incident_summary(
        incidents_df=incidents_gold,
        device_metrics_df=device_metrics_gold,
        alert_stats_df=alert_stats_gold,
        incident_timeline_df=timeline_gold,
        alerts_df=alerts_clustered,
    )

    save_gold_outputs(
        output_dir=output_dir,
        incidents_gold=incidents_gold,
        device_metrics_gold=device_metrics_gold,
        alert_stats_gold=alert_stats_gold,
        timeline_gold=timeline_gold,
        incident_summary=incident_summary,
    )

    return {
        "incidents": incidents_gold,
        "device_metrics": device_metrics_gold,
        "alert_stats": alert_stats_gold,
        "incident_timeline": timeline_gold,
        "incident_summary": incident_summary,
        "incident_statistics": incident_statistics,
        "failure_patterns": failure_patterns,
        "pipeline_metrics": pipeline_metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run analytics pipeline from Silver data to Gold outputs.")
    parser.add_argument(
        "--silver-path",
        default=None,
        help="Input Silver dataset path (.parquet or .csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="analytics/outputs",
        help="Output directory for Gold datasets",
    )
    args = parser.parse_args()
    run_pipeline(silver_path=args.silver_path, output_dir=args.output_dir)
