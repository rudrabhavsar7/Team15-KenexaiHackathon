"""
llm/rag/document_loader.py

RAG Pipeline — Stage 1: Document Loader
----------------------------------------
Loads structured incident datasets from various file formats produced by the
analytics pipeline and converts each record into a plain-text document suitable
for embedding and storage in a vector database.

Supported sources
-----------------
- analytics/outputs/incidents.parquet
- analytics/outputs/analytics_summary.json
- gold/alerts_table.parquet

Author: Team 15 — KenexAI Hackathon
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Field configuration
# ---------------------------------------------------------------------------

# Maps DataFrame column names → human-readable label used in the document.
# Order here defines the order of lines in the output document.
FIELD_LABELS: Dict[str, str] = {
    "incident_id":  "Incident ID",
    "device":       "Device",
    "alert_type":   "Alert Type",
    "severity":     "Severity",
    "timestamp":    "Timestamp",
    "service":      "Affected Service",
    "description":  "Description",
    "duration":     "Duration",
    "root_cause":   "Possible Root Cause",
}

# ---------------------------------------------------------------------------
# Column aliases
# ---------------------------------------------------------------------------

# Maps alternate/real-world column names → canonical FIELD_LABELS keys.
# Add new aliases here whenever a data source uses a different field name.
#
# Examples
# --------
#   Auvik / Meraki payloads  →  "alertName"    maps to "alert_type"
#   N-Central payloads       →  "alert_name"   maps to "alert_type"
#   Generic sources          →  "name"         maps to "device"
COLUMN_ALIASES: Dict[str, str] = {
    # Alert type variants
    "alert_name":           "alert_type",
    "alertName":            "alert_type",
    "alertname":            "alert_type",
    "check":                "alert_type",   # Meraki "check" field
    "alertType":            "alert_type",

    # Incident ID variants
    "alertId":              "incident_id",
    "alert_id":             "incident_id",
    "id":                   "incident_id",

    # Device / entity variants
    "entityName":           "device",
    "entity_name":          "device",
    "host":                 "device",
    "deviceName":           "device",
    "device_name":          "device",

    # Severity variants
    "alertSeverityString":  "severity",
    "alert_severity_string": "severity",
    "alertLevel":           "severity",
    "alert_level":          "severity",
    "status":               "severity",

    # Timestamp variants
    "date":                 "timestamp",
    "occurredAt":           "timestamp",
    "occurred_at":          "timestamp",
    "sentAt":               "timestamp",
    "sent_at":              "timestamp",

    # Description variants
    "alertDescription":     "description",
    "alert_description":    "description",
    "message":              "description",

    # Service variants
    "entityType":           "service",
    "entity_type":          "service",
    "companyName":          "service",
    "company_name":         "service",
}


# ---------------------------------------------------------------------------
# 1. load_parquet_file
# ---------------------------------------------------------------------------

def load_parquet_file(file_path: str) -> pd.DataFrame:
    """Load a Parquet dataset into a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the ``.parquet`` file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the records from the Parquet file.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist on disk.
    ValueError
        If the file cannot be parsed as a valid Parquet dataset.
    """
    if not os.path.exists("analytics/outputs/incidents.parquet"):
        raise FileNotFoundError(f"Parquet file not found: '{file_path}'")

    logger.info("Loading parquet file: %s", file_path)
    try:
        df = pd.read_parquet(file_path)
        logger.info("Loaded %d rows and %d columns from '%s'.", len(df), len(df.columns), file_path)
        return df
    except Exception as exc:
        raise ValueError(f"Failed to parse parquet file '{file_path}': {exc}") from exc


# ---------------------------------------------------------------------------
# 2. load_json_file
# ---------------------------------------------------------------------------

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load a JSON analytics summary file.

    The JSON file may contain either a top-level dict (analytics summary) or a
    list of records.  When a list is detected it is normalised into a dict with
    the key ``"records"`` so callers always receive a ``dict``.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the ``.json`` file.

    Returns
    -------
    dict
        Parsed JSON content.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist on disk.
    ValueError
        If the file contains invalid JSON.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found: '{file_path}'")

    logger.info("Loading JSON file: %s", file_path)
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        # Normalise a top-level list into a dict for a consistent return type.
        if isinstance(data, list):
            logger.debug("JSON root is a list — wrapping under key 'records'.")
            data = {"records": data}

        logger.info("JSON file loaded successfully (%d top-level keys).", len(data))
        return data
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in '{file_path}': {exc}") from exc


# ---------------------------------------------------------------------------
# 3. incident_row_to_document
# ---------------------------------------------------------------------------

def incident_row_to_document(
    row: pd.Series,
    field_labels: Optional[Dict[str, str]] = None,
) -> str:
    """Convert a single DataFrame row into a structured plain-text document.

    Only fields that are present **and** non-null in *row* are included.
    Missing or ``NaN`` fields are skipped gracefully so the function never
    raises a ``KeyError``.

    Field values that look like alert types or severity strings are title-cased
    automatically (e.g. ``"cpu_high"`` → ``"Cpu High"``).

    Parameters
    ----------
    row : pd.Series
        A single row from an incident DataFrame.
    field_labels : dict, optional
        Custom ``{column_name: label}`` mapping.  Defaults to the module-level
        ``FIELD_LABELS`` dict.

    Returns
    -------
    str
        A newline-separated text document ready for embedding.

    Example
    -------
    >>> s = pd.Series({"incident_id": 1, "device": "router-1", "severity": "critical"})
    >>> print(incident_row_to_document(s))
    Incident ID: 1
    Device: router-1
    Severity: Critical
    """
    labels = field_labels if field_labels is not None else FIELD_LABELS
    lines: List[str] = []

    for col, label in labels.items():
        value = row.get(col, None)

        # Skip missing or null values.
        if value is None or (isinstance(value, float) and pd.isna(value)):
            logger.debug("Column '%s' is missing or null — skipping.", col)
            continue

        # Stringify and lightly clean the value.
        value_str = str(value).strip()
        if not value_str or value_str.lower() == "nan":
            continue

        # Title-case values that use underscores as word separators
        # (e.g. alert_type / severity values from analytics pipelines).
        if "_" in value_str:
            value_str = value_str.replace("_", " ").title()

        lines.append(f"{label}: {value_str}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. build_incident_documents
# ---------------------------------------------------------------------------

def build_incident_documents(
    df: pd.DataFrame,
    field_labels: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Iterate over every row in *df* and produce a list of text documents.

    Rows that produce an empty document (all fields missing) are skipped with a
    warning so that the output list contains only meaningful documents.

    Parameters
    ----------
    df : pd.DataFrame
        Incident/alert records loaded from a parquet or JSON source.
    field_labels : dict, optional
        Custom field-to-label mapping forwarded to
        :func:`incident_row_to_document`.

    Returns
    -------
    List[str]
        One document string per valid row.
    """
    documents: List[str] = []
    skipped = 0

    for idx, row in df.iterrows():
        doc = incident_row_to_document(row, field_labels=field_labels)
        if not doc.strip():
            logger.warning("Row %s produced an empty document — skipping.", idx)
            skipped += 1
            continue
        documents.append(doc)

    logger.info(
        "Built %d document(s) from %d row(s) — %d skipped (empty).",
        len(documents), len(df), skipped,
    )
    return documents


def build_incident_summary_documents(summary: Dict[str, Any]) -> List[str]:
    """Build searchable documents from analytics gold-layer incident summary.

    The analytics output file (``incident_summary.json``) contains aggregate
    sections like overview, top failing devices, alert frequencies, recent
    incidents, and timeline snapshots. This function converts those sections
    into multiple compact text documents suitable for semantic retrieval.
    """
    documents: List[str] = []

    generated_at = summary.get("generated_at", "")
    overview = summary.get("overview", {}) or {}
    documents.append(
        "\n".join(
            [
                "Document Type: Incident Summary Overview",
                f"Generated At: {generated_at}",
                f"Total Incidents: {overview.get('total_incidents', 'N/A')}",
                f"Total Devices: {overview.get('total_devices', 'N/A')}",
                f"Total Alerts: {overview.get('total_alerts', 'N/A')}",
                f"Average Reliability Score: {overview.get('avg_reliability_score', 'N/A')}",
            ]
        )
    )

    for item in summary.get("top_failing_devices", []) or []:
        documents.append(
            "\n".join(
                [
                    "Document Type: Top Failing Device",
                    f"Generated At: {generated_at}",
                    f"Device: {item.get('device', 'N/A')}",
                    f"Organization: {item.get('organization', 'N/A')}",
                    f"Total Alerts: {item.get('total_alerts', 'N/A')}",
                    f"Critical Alerts: {item.get('critical_alerts', 'N/A')}",
                    f"Incidents Count: {item.get('incidents_count', 'N/A')}",
                    f"Reliability Score: {item.get('reliability_score', 'N/A')}",
                    f"Anomaly Score: {item.get('anomaly_score', 'N/A')}",
                    f"Is Anomalous: {item.get('is_anomalous', 'N/A')}",
                ]
            )
        )

    for item in summary.get("most_frequent_alert_types", []) or []:
        documents.append(
            "\n".join(
                [
                    "Document Type: Frequent Alert Type",
                    f"Generated At: {generated_at}",
                    f"Alert Type: {item.get('alert_type', 'N/A')}",
                    f"Total Count: {item.get('total_count', 'N/A')}",
                    f"Devices Affected: {item.get('devices_affected', 'N/A')}",
                    f"Average Severity: {item.get('avg_severity', 'N/A')}",
                ]
            )
        )

    for item in summary.get("recent_incidents", []) or []:
        documents.append(
            "\n".join(
                [
                    "Document Type: Recent Incident",
                    f"Generated At: {generated_at}",
                    f"Incident ID: {item.get('incident_id', 'N/A')}",
                    f"Device: {item.get('device', 'N/A')}",
                    f"Organization: {item.get('organization', 'N/A')}",
                    f"Alert Count: {item.get('alert_count', 'N/A')}",
                    f"Severity Score: {item.get('severity_score', 'N/A')}",
                    f"Start Time: {item.get('start_time', 'N/A')}",
                    f"End Time: {item.get('end_time', 'N/A')}",
                    f"Duration Minutes: {item.get('duration_minutes', 'N/A')}",
                    f"Root Cause Candidate: {item.get('root_cause_candidate', 'N/A')}",
                    f"Incident Type: {item.get('incident_type', 'N/A')}",
                ]
            )
        )

    for item in summary.get("timeline_snapshot", []) or []:
        documents.append(
            "\n".join(
                [
                    "Document Type: Incident Timeline Snapshot",
                    f"Generated At: {generated_at}",
                    f"Timestamp: {item.get('timestamp', 'N/A')}",
                    f"Alerts Count: {item.get('alerts_count', 'N/A')}",
                    f"Incidents Count: {item.get('incidents_count', 'N/A')}",
                    f"Critical Incidents: {item.get('critical_incidents', 'N/A')}",
                ]
            )
        )

    documents = [doc for doc in documents if doc.strip()]
    logger.info("Built %d document(s) from analytics incident summary JSON.", len(documents))
    return documents


def _looks_like_incident_summary(data: Dict[str, Any]) -> bool:
    """Return True when JSON matches analytics/outputs/incident_summary schema."""
    expected = {
        "overview",
        "top_failing_devices",
        "most_frequent_alert_types",
        "recent_incidents",
        "timeline_snapshot",
    }
    return isinstance(data, dict) and expected.issubset(set(data.keys()))


# ---------------------------------------------------------------------------
# 5b. normalize_columns  (internal helper)
# ---------------------------------------------------------------------------

def normalize_columns(
    df: pd.DataFrame,
    aliases: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Rename DataFrame columns using the ``COLUMN_ALIASES`` map.

    This allows the document builder to accept data from any source (Auvik,
    Meraki, N-Central, custom JSON) regardless of whether the source uses
    ``alert_name``, ``alertName``, or ``alert_type`` — they all get mapped
    to the canonical ``alert_type`` key used in ``FIELD_LABELS``.

    Only columns that are **not already** in ``FIELD_LABELS`` are renamed.
    This prevents stomping on a canonical column that happens to share a name
    with an alias target.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame loaded from parquet or JSON.
    aliases : dict, optional
        Custom alias map.  Defaults to the module-level ``COLUMN_ALIASES``.

    Returns
    -------
    pd.DataFrame
        DataFrame with aliased column names replaced by their canonical names.

    Example
    -------
    >>> df.columns.tolist()
    ['incident_id', 'device', 'alert_name', 'severity', 'description', 'timestamp']
    >>> normalize_columns(df).columns.tolist()
    ['incident_id', 'device', 'alert_type', 'severity', 'description', 'timestamp']
    """
    alias_map = aliases if aliases is not None else COLUMN_ALIASES
    canonical_cols = set(FIELD_LABELS.keys())

    rename: Dict[str, str] = {}
    for col in df.columns:
        # Only rename if: the column is an alias AND the target isn't already present.
        if col in alias_map and col not in canonical_cols:
            target = alias_map[col]
            if target not in df.columns:
                rename[col] = target
                logger.debug("Column alias: '%s' → '%s'.", col, target)
            else:
                logger.debug(
                    "Skipping alias '%s' → '%s': target column already exists.",
                    col, target,
                )

    if rename:
        df = df.rename(columns=rename)
        logger.info("Normalised %d column alias(es): %s.", len(rename), rename)
    else:
        logger.debug("No column aliases needed — all columns already canonical.")

    return df


# ---------------------------------------------------------------------------
# 5. load_incident_documents
# ---------------------------------------------------------------------------

def load_incident_documents(
    file_path: str,
    json_records_key: str = "records",
    field_labels: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Main entry point — load a dataset and return a list of text documents.

    Dispatches to either :func:`load_parquet_file` or :func:`load_json_file`
    based on the file extension, then calls :func:`build_incident_documents` to
    produce the final document list.

    Parameters
    ----------
    file_path : str
        Path to a ``.parquet`` or ``.json`` file.
    json_records_key : str, optional
        When the source is JSON, this key is used to find the list of records
        inside the parsed dict.  Defaults to ``"records"``.
    field_labels : dict, optional
        Custom field-to-label mapping.  Defaults to ``FIELD_LABELS``.

    Returns
    -------
    List[str]
        Text documents ready for embedding.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    ValueError
        If the file format is unsupported or the file is malformed.

    Examples
    --------
    >>> docs = load_incident_documents("analytics/outputs/incidents.parquet")
    >>> docs = load_incident_documents("analytics/outputs/analytics_summary.json")
    """
    ext = os.path.splitext(file_path)[-1].lower()

    # ---- Parquet ---------------------------------------------------------- #
    if ext == ".parquet":
        df = load_parquet_file(file_path)

    # ---- JSON ------------------------------------------------------------- #
    elif ext == ".json":
        raw = load_json_file(file_path)

        if _looks_like_incident_summary(raw):
            logger.info("Detected analytics incident summary schema in '%s'.", file_path)
            return build_incident_summary_documents(raw)

        # Try to extract a records list from the JSON.
        records: Any = raw.get(json_records_key, None)

        if records is None:
            # Treat the entire dict as a single flat record.
            logger.warning(
                "Key '%s' not found in JSON — treating the whole dict as one record.",
                json_records_key,
            )
            records = [raw]

        if not isinstance(records, list):
            raise ValueError(
                f"Expected a list under key '{json_records_key}' in the JSON, "
                f"got {type(records).__name__}."
            )

        df = pd.DataFrame(records)

    # ---- Unsupported ------------------------------------------------------ #
    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            "Only '.parquet' and '.json' files are supported."
        )

    if df.empty:
        logger.warning("Dataset at '%s' is empty — returning no documents.", file_path)
        return []

    # Normalise real-world column names (e.g. alert_name → alert_type).
    df = normalize_columns(df)

    return build_incident_documents(df, field_labels=field_labels)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python document_loader.py <path_to_dataset>")
        sys.exit(1)

    path = sys.argv[1]
    docs = load_incident_documents(path)

    print(f"\n{'='*60}")
    print(f"  Loaded {len(docs)} document(s) from: {path}")
    print(f"{'='*60}\n")

    for i, doc in enumerate(docs[:5], start=1):
        print(f"--- Document {i} ---")
        print(doc)
        print()

    if len(docs) > 5:
        print(f"... and {len(docs) - 5} more document(s).")
