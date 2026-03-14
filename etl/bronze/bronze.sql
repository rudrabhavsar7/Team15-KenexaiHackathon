-- Active: 1773470111546@@127.0.0.1@5432@kenexaihackathon
CREATE SCHEMA if NOT EXISTS bronze;

CREATE TABLE bronze.auvik_alerts (

    id BIGSERIAL PRIMARY KEY,

    entity_id TEXT,
    subject TEXT,

    alert_status_string TEXT,
    alert_id TEXT,
    alert_name TEXT,

    entity_name TEXT,
    company_name TEXT,
    entity_type TEXT,

    alert_description TEXT,

    alert_severity_string TEXT,
    alert_severity INTEGER,

    correlation_id TEXT,      -- same as other sources

    alert_status INTEGER,

    event_time TIMESTAMP,

    link TEXT,

    event_state TEXT,         -- triggered / resolved
    synthetic BOOLEAN DEFAULT FALSE,

    ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE bronze.meraki_alerts (

    id BIGSERIAL PRIMARY KEY,

    app_key TEXT,
    status TEXT,
    check_name TEXT,
    version TEXT,
    shared_secret TEXT,

    sent_at TIMESTAMP,
    occurred_at TIMESTAMP,

    organization_id TEXT,
    organization_name TEXT,
    organization_url TEXT,

    network_id TEXT,
    network_name TEXT,
    network_url TEXT,

    device_serial TEXT,
    device_mac TEXT,
    device_name TEXT,
    device_url TEXT,
    device_model TEXT,

    host TEXT,

    alert_id TEXT,
    alert_type TEXT,
    alert_type_id TEXT,
    alert_level TEXT,

    event_state TEXT,       -- triggered / resolved
    synthetic BOOLEAN,
    correlation_id TEXT,

    ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE bronze.ncentral_alerts (

    id BIGSERIAL PRIMARY KEY,

    active_notification_trigger_id TEXT,

    customer_name TEXT,
    device_uri TEXT,
    device_name TEXT,

    affected_service TEXT,
    task_ident TEXT,

    ncentral_uri TEXT,

    qualitative_old_state TEXT,
    qualitative_new_state TEXT,

    time_of_state_change TIMESTAMP,

    probe_uri TEXT,

    quantitative_new_state TEXT,

    service_organization_name TEXT,

    remote_control_link TEXT,

    event_state TEXT,     -- triggered / resolved
    synthetic BOOLEAN,

    correlation_id TEXT,

    ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

