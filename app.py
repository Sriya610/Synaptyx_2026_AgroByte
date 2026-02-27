import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest


# ---------------------------
# Data + Model Functions
# ---------------------------
def generate_data(normal_only: bool = False) -> dict:
    """
    Generate one farm micro-climate sample.

    Ranges:
    - Temperature: 20-45 °C
    - Humidity: 20-90 %
    - Soil moisture: 10-80 %

    If normal_only=False, occasionally inject abnormal values to simulate anomalies.
    """
    temp = random.uniform(20, 45)
    humidity = random.uniform(20, 90)
    soil = random.uniform(10, 80)

    # Inject anomalies occasionally (~10% chance) for demo realism
    if not normal_only and random.random() < 0.10:
        anomaly_type = random.choice(["heat", "dry", "wet", "cold"])
        if anomaly_type == "heat":
            temp = random.uniform(46, 55)
            humidity = random.uniform(10, 35)
        elif anomaly_type == "dry":
            humidity = random.uniform(5, 25)
            soil = random.uniform(2, 15)
        elif anomaly_type == "wet":
            humidity = random.uniform(90, 100)
            soil = random.uniform(80, 95)
        elif anomaly_type == "cold":
            temp = random.uniform(10, 19)

    return {
        "timestamp": datetime.now(),
        "temperature": round(temp, 2),
        "humidity": round(humidity, 2),
        "soil_moisture": round(soil, 2),
    }


def calculate_csi(temp: float, humidity: float, soil: float) -> tuple[float, str]:
    """Calculate Crop Stress Index (CSI) and category."""
    temp_stress = (temp - 20) / (45 - 20)
    humidity_stress = (90 - humidity) / (90 - 20)
    moisture_stress = (80 - soil) / (80 - 10)

    csi = (
        0.4 * temp_stress
        + 0.3 * humidity_stress
        + 0.3 * moisture_stress
    ) * 100

    # Keep CSI within 0-100
    csi = round(float(np.clip(csi, 0, 100)), 2)

    if csi <= 30:
        status = "Healthy"
    elif csi <= 60:
        status = "Moderate"
    elif csi <= 80:
        status = "High Risk"
    else:
        status = "Critical"

    return csi, status


def train_model() -> IsolationForest:
    """Train Isolation Forest on 200 samples of normal data."""
    samples = [generate_data(normal_only=True) for _ in range(200)]
    df_train = pd.DataFrame(samples)
    features = df_train[["temperature", "humidity", "soil_moisture"]]

    model = IsolationForest(
        contamination=0.05,
        random_state=42,
    )
    model.fit(features)
    return model


def detect_anomaly(model: IsolationForest, temp: float, humidity: float, soil: float) -> int:
    """Return -1 for anomaly, 1 for normal sample."""
    x = pd.DataFrame([
        {
            "temperature": temp,
            "humidity": humidity,
            "soil_moisture": soil,
        }
    ])
    return int(model.predict(x)[0])


# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="Farm Micro-Climate Monitor", layout="wide")

st.title("🌾 Farm Micro-Climate Monitoring & Anomaly Detection")
st.caption("Real-time environmental monitoring with CSI scoring and Isolation Forest alerts.")

# Re-run script every 2 seconds for live simulation
st.markdown("<meta http-equiv='refresh' content='2'>", unsafe_allow_html=True)

if "model" not in st.session_state:
    st.session_state.model = train_model()

if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=[
            "timestamp",
            "temperature",
            "humidity",
            "soil_moisture",
            "csi",
            "status",
            "anomaly",
        ]
    )

# Generate new live point each run
point = generate_data()
csi, csi_status = calculate_csi(
    point["temperature"],
    point["humidity"],
    point["soil_moisture"],
)
anomaly_flag = detect_anomaly(
    st.session_state.model,
    point["temperature"],
    point["humidity"],
    point["soil_moisture"],
)

row = {
    **point,
    "csi": csi,
    "status": csi_status,
    "anomaly": anomaly_flag,
}
st.session_state.history = pd.concat(
    [st.session_state.history, pd.DataFrame([row])], ignore_index=True
).tail(120)

current = st.session_state.history.iloc[-1]

# Live metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Temperature (°C)", f"{current['temperature']:.2f}")
m2.metric("Humidity (%)", f"{current['humidity']:.2f}")
m3.metric("Soil Moisture (%)", f"{current['soil_moisture']:.2f}")
m4.metric("Crop Stress Index", f"{current['csi']:.2f}")

status_colors = {
    "Healthy": "#2ECC71",
    "Moderate": "#F1C40F",
    "High Risk": "#E67E22",
    "Critical": "#E74C3C",
}

st.markdown(
    f"""
    <div style='padding: 0.75rem 1rem; border-radius: 0.75rem; background: {status_colors[current['status']]}; color: white; width: fit-content; font-weight: 700;'>
        CSI Status: {current['status']}
    </div>
    """,
    unsafe_allow_html=True,
)

# Alerts
st.subheader("🚨 Alerts")
alert_messages = []

if current["csi"] > 75:
    alert_messages.append("Possible heat stress detected")
    alert_messages.append("Irrigation recommended")

if current["anomaly"] == -1:
    alert_messages.append("Isolation Forest detected an environmental anomaly")

if alert_messages:
    for msg in dict.fromkeys(alert_messages):
        st.error(msg)
else:
    st.success("No active alerts. Farm conditions are stable.")

# Real-time chart
st.subheader("📈 Live Trends")
plot_df = st.session_state.history.copy()
plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"])

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=plot_df["timestamp"],
        y=plot_df["temperature"],
        mode="lines",
        name="Temperature (°C)",
    )
)
fig.add_trace(
    go.Scatter(
        x=plot_df["timestamp"],
        y=plot_df["humidity"],
        mode="lines",
        name="Humidity (%)",
    )
)
fig.add_trace(
    go.Scatter(
        x=plot_df["timestamp"],
        y=plot_df["soil_moisture"],
        mode="lines",
        name="Soil Moisture (%)",
    )
)
fig.add_trace(
    go.Scatter(
        x=plot_df["timestamp"],
        y=plot_df["csi"],
        mode="lines",
        name="CSI",
        line=dict(width=3, dash="dot"),
    )
)

anomalies = plot_df[plot_df["anomaly"] == -1]
if not anomalies.empty:
    fig.add_trace(
        go.Scatter(
            x=anomalies["timestamp"],
            y=anomalies["csi"],
            mode="markers",
            name="Anomaly Points",
            marker=dict(size=10, color="red", symbol="x"),
        )
    )

fig.update_layout(
    template="plotly_white",
    height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=10, r=10, t=10, b=10),
)

st.plotly_chart(fig, use_container_width=True)

st.caption("Auto-refresh interval: 2 seconds. Retains latest 120 points for smooth demo.")
