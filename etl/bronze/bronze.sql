-- Active: 1773470111546@@127.0.0.1@5432@kenexaihackathon
CREATE SCHEMA if NOT EXISTS bronze;

CREATE TABLE if NOT EXISTS bronze.auvik_alerts_raw (
    bronze_id BIGSERIAL PRIMARY KEY,

    ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    entity_id TEXT,
    subject TEXT,
    alert_status_string VARCHAR(50),
    alert_id TEXT,
    alert_name VARCHAR(150),

    entity_name VARCHAR(150),
    company_name VARCHAR(200),
    entity_type VARCHAR(50),

    alert_date TIMESTAMP,
    link TEXT,

    alert_status INT,
    correlation_id TEXT,

    alert_description TEXT,

    alert_severity_string VARCHAR(50),
    alert_severity INT,

    company_id TEXT
);

CREATE TABLE IF NOT EXISTS bronze.meraki_alerts_raw (
    bronze_id BIGSERIAL PRIMARY KEY,

    ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    app_key TEXT,
    alert_status VARCHAR(50),
    check_name VARCHAR(150),
    payload_version VARCHAR(20),
    shared_secret TEXT,

    sent_at TIMESTAMP,

    organization_id TEXT,
    organization_name TEXT,
    organization_url TEXT,

    network_id TEXT,
    network_name TEXT,
    network_url TEXT,

    device_serial TEXT,
    device_mac VARCHAR(50),
    device_name VARCHAR(150),
    device_url TEXT,
    device_model VARCHAR(50),

    alert_id TEXT,
    alert_type VARCHAR(150),
    alert_type_id VARCHAR(150),
    alert_level VARCHAR(50),

    occurred_at TIMESTAMP,
    device_host VARCHAR(150)
);

CREATE TABLE IF NOT EXISTS bronze.ncentral_alerts_raw (
    bronze_id BIGSERIAL PRIMARY KEY,

    ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    active_notification_trigger_id BIGINT,
    customer_name TEXT,

    device_uri TEXT,
    device_name VARCHAR(150),

    external_customer_id TEXT,

    affected_service VARCHAR(150),
    task_ident VARCHAR(50),

    ncentral_uri TEXT,

    qualitative_old_state VARCHAR(50),
    qualitative_new_state VARCHAR(50),

    time_of_state_change TIMESTAMP,

    probe_uri TEXT,

    quantitative_new_state TEXT,

    service_organization_name TEXT,

    remote_control_link TEXT
);