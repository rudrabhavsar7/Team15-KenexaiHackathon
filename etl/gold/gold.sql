-- Active: 1773470111546@@127.0.0.1@5432@kenexaihackathon
CREATE SCHEMA IF NOT EXISTS gold;

CREATE TABLE gold.dim_devices (
    device_id SERIAL PRIMARY KEY,
    device_name VARCHAR(150) UNIQUE,
    source_system VARCHAR(50)
);

CREATE TABLE gold.dim_severity (
    severity_id SERIAL PRIMARY KEY,
    severity VARCHAR(50),
    severity_score INT
);

CREATE TABLE gold.dim_alert_types (
    alert_type_id SERIAL PRIMARY KEY,
    alert_type VARCHAR(150),
    alert_category VARCHAR(100)
);

CREATE TABLE gold.dim_time (
    time_id SERIAL PRIMARY KEY,
    occurred_at TIMESTAMP,
    alert_hour INT,
    alert_day DATE
);

drop TABLE gold.fact_alerts;
CREATE TABLE gold.fact_alerts (

    alert_id TEXT PRIMARY KEY,

    device_id INT,
    severity_id INT,
    alert_type_id INT,
    time_id INT,

    source_system VARCHAR(50),

    FOREIGN KEY (device_id) REFERENCES gold.dim_devices(device_id),
    FOREIGN KEY (severity_id) REFERENCES gold.dim_severity(severity_id),
    FOREIGN KEY (alert_type_id) REFERENCES gold.dim_alert_types(alert_type_id),
    FOREIGN KEY (time_id) REFERENCES gold.dim_time(time_id)
);

CREATE TABLE gold.device_alert_summary AS

SELECT
device_name,
COUNT(*) AS total_alerts,
SUM(CASE WHEN severity='Critical' THEN 1 ELSE 0 END) AS critical_alerts,
SUM(CASE WHEN severity='Warning' THEN 1 ELSE 0 END) AS warning_alerts

FROM silver.alerts

GROUP BY device_name;

CREATE TABLE gold.incidents (
    incident_id SERIAL PRIMARY KEY,
    device_name VARCHAR(150),
    incident_type VARCHAR(150),
    severity VARCHAR(50),
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    alert_count INT
); 