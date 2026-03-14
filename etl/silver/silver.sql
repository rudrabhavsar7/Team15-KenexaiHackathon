-- Active: 1773470111546@@127.0.0.1@5432@kenexaihackathon
CREATE SCHEMA IF NOT EXISTS silver;
CREATE TABLE IF NOT EXISTS silver.alerts (
    alert_id TEXT PRIMARY KEY,

    source_system VARCHAR(50),

    device_name VARCHAR(150),
    device_identifier TEXT,

    organization_name TEXT,
    network_name TEXT,

    alert_type VARCHAR(150),
    severity VARCHAR(50),

    alert_message TEXT,

    occurred_at TIMESTAMP,

    ingestion_time TIMESTAMP
);

