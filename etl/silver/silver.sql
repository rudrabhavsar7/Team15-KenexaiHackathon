-- Active: 1773470111546@@127.0.0.1@5432@kenexaihackathon
CREATE SCHEMA IF NOT EXISTS silver;

CREATE TABLE silver.alerts (

    id BIGSERIAL PRIMARY KEY,

    source_system TEXT,           -- meraki / auvik / ncentral

    alert_id TEXT,
    correlation_id TEXT,

    organization_name TEXT,

    device_name TEXT,
    device_identifier TEXT,

    service_name TEXT,

    alert_type TEXT,

    severity TEXT,

    event_state TEXT,             -- triggered / resolved

    event_time TIMESTAMP,

    synthetic BOOLEAN,

    ingestion_time TIMESTAMP
);

CREATE TABLE silver.alerts_clean (

    id BIGSERIAL PRIMARY KEY,

    source_system TEXT,

    alert_id TEXT,

    correlation_id TEXT,

    organization_name TEXT,

    device_name TEXT,
    device_identifier TEXT,

    service_name TEXT,

    alert_type TEXT,

    severity TEXT,

    event_state TEXT,

    event_time TIMESTAMP,

    synthetic BOOLEAN,

    ingestion_time TIMESTAMP
);

