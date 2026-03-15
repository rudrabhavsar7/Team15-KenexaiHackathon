-- Active: 1773470111546@@127.0.0.1@5432@kenexaihackathon
CREATE SCHEMA IF NOT EXISTS gold;

CREATE TABLE gold.dim_device (

    device_key SERIAL PRIMARY KEY,

    device_name TEXT,

    device_identifier TEXT,

    source_system TEXT
);

CREATE TABLE gold.dim_organization (

    organization_key SERIAL PRIMARY KEY,

    organization_name TEXT
);

CREATE TABLE gold.dim_alert_type (

    alert_type_key SERIAL PRIMARY KEY,

    alert_type TEXT,

    service_name TEXT
);

CREATE TABLE gold.dim_time (

    time_key SERIAL PRIMARY KEY,

    event_time TIMESTAMP,

    event_date DATE,

    hour INTEGER,

    day INTEGER,

    month INTEGER,

    year INTEGER
);

CREATE TABLE gold.fact_incidents (

    incident_key BIGSERIAL PRIMARY KEY,

    correlation_id TEXT,

    device_key INTEGER REFERENCES gold.dim_device(device_key),

    organization_key INTEGER REFERENCES gold.dim_organization(organization_key),

    alert_type_key INTEGER REFERENCES gold.dim_alert_type(alert_type_key),

    start_time_key INTEGER REFERENCES gold.dim_time(time_key),

    end_time_key INTEGER REFERENCES gold.dim_time(time_key),

    incident_duration INTERVAL,

    alert_count INTEGER,

    severity TEXT,

    synthetic BOOLEAN
);

CREATE TABLE gold.incident_tickets (

ticket_id SERIAL PRIMARY KEY,

incident_id INTEGER,

device_name TEXT,

issue TEXT,

status TEXT,

created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE gold.dim_device
ADD CONSTRAINT unique_device
UNIQUE(device_name, device_identifier);

ALTER TABLE gold.dim_organization
ADD CONSTRAINT unique_organization
UNIQUE(organization_name);

ALTER TABLE gold.dim_alert_type
ADD CONSTRAINT unique_alert_type
UNIQUE(alert_type, service_name);

ALTER TABLE gold.fact_incidents
ADD COLUMN incident_status TEXT DEFAULT 'OPEN';
