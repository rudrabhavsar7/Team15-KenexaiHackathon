import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px

st.set_page_config(
    page_title="Alert Intelligence",
    layout="wide"
)

@st.cache_data
def load_data(query):
    conn = psycopg2.connect(
        host="localhost",
        database="kenexaihackathon",
        user="postgres",
        password="09092002"
    )
    df = pd.read_sql(query, conn)
    conn.close()
    return df

alerts = load_data("SELECT * FROM gold.fact_alerts")
devices = load_data("SELECT * FROM gold.device_alert_summary")
incidents = load_data("SELECT * FROM gold.incidents")
dim_time = load_data("SELECT * FROM gold.dim_time")

if "occurred_at" in dim_time.columns:
    dim_time["datetime"] = pd.to_datetime(dim_time["occurred_at"])
else:
    dim_time["datetime"] = pd.NaT

st.title("🚨 Alert Incident Intelligence Platform")
st.markdown("Real-time infrastructure monitoring & incident analytics")
st.divider()

st.sidebar.header("Filters")

# device_name is used in device_alert_summary and incidents, device_id in fact_alerts
device_names = devices["device_name"].unique()
device_filter = st.sidebar.multiselect(
    "Select Devices",
    device_names
)

if device_filter:
    # Filter alerts by device_id matching device_name in dim_devices
    dim_devices = load_data("SELECT device_id, device_name FROM gold.dim_devices")
    filtered_device_ids = dim_devices[dim_devices["device_name"].isin(device_filter)]["device_id"].tolist()
    alerts = alerts[alerts["device_id"].isin(filtered_device_ids)]
    devices = devices[devices["device_name"].isin(device_filter)]
    incidents = incidents[incidents["device_name"].isin(device_filter)]

st.subheader("System Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Alerts", len(alerts))
col2.metric("Total Devices", len(devices))
col3.metric("Incidents", len(incidents))
col4.metric("Unique Alert Types", alerts["alert_type_id"].nunique() if "alert_type_id" in alerts.columns else 0)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Alert Analytics",
    "🖥 Device Health",
    "⚠ Incident Monitoring",
    "🔍 Root Cause Explorer"
])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Alert Severity Distribution")
        if "severity_id" in alerts.columns:
            severity_counts = alerts.groupby("severity_id").size().reset_index(name="count")
            fig = px.pie(
                severity_counts,
                names="severity_id",
                values="count",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Column 'severity_id' not found in alerts data.")
            st.subheader("Alert Type Distribution")
            if "alert_type_id" in alerts.columns:
                alert_type_counts = alerts.groupby("alert_type_id").size().reset_index(name="count")
                fig = px.bar(
                    alert_type_counts,
                    x="alert_type_id",
                    y="count",
                    labels={"alert_type_id": "Alert Type", "count": "Count"},
                    title="Alert Type Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Column 'alert_type_id' not found in alerts data.")
    with col2:
        st.subheader("Alerts Over Time")
        if "time_id" in alerts.columns and "time_id" in dim_time.columns and "datetime" in dim_time.columns:
            merged = alerts.merge(dim_time, on="time_id", how="left")
            timeline = merged.groupby("datetime").size().reset_index(name="alerts")
            fig = px.line(
                timeline,
                x="datetime",
                y="alerts",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Required columns ('time_id' in alerts and dim_time, 'datetime' in dim_time) not found for timeline plot.")

with tab2:
    st.subheader("Top Devices Generating Alerts")
    if "total_alerts" in devices.columns and "device_name" in devices.columns:
        fig = px.bar(
            devices.sort_values("total_alerts", ascending=False).head(10),
            x="total_alerts",
            y="device_name",
            orientation="h"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Columns 'total_alerts' or 'device_name' not found in devices data.")
    st.subheader("Device Alert Summary")
    st.dataframe(devices)

with tab3:
    st.subheader("Detected Incidents")
    if "alert_count" in incidents.columns:
        st.dataframe(
            incidents.sort_values("alert_count", ascending=False),
            use_container_width=True
        )
    else:
        st.dataframe(incidents, use_container_width=True)

with tab4:
    st.subheader("Root Cause Explorer")
    # Use device_name for selection
    if "device_name" in devices.columns:
        device = st.selectbox(
            "Select Device",
            devices["device_name"].unique()
        )
        # Find device_id for selected device_name
        dim_devices = load_data("SELECT device_id, device_name FROM gold.dim_devices")
        device_id = dim_devices[dim_devices["device_name"] == device]["device_id"].iloc[0]
        device_alerts = alerts[alerts["device_id"] == device_id]
        st.write("Recent Alert Messages")
        if "alert_message" in device_alerts.columns and "alert_id" in device_alerts.columns:
            st.dataframe(device_alerts[["alert_message","alert_id"]])
        else:
            st.dataframe(device_alerts)
    else:
        # Map severity_id to label for pie chart
        if "severity_id" in alerts.columns:
            # Load severity mapping table
            severity_map_df = load_data("SELECT severity_id, severity FROM gold.dim_severity")
            severity_map = dict(zip(severity_map_df["severity_id"], severity_map_df["severity"]))
            severity_counts = alerts.groupby("severity_id").size().reset_index(name="count")
            severity_counts["severity"] = severity_counts["severity_id"].map(severity_map)
            fig = px.pie(
                severity_counts,
                names="severity",
                values="count",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Column 'severity_id' not found in alerts data.")
