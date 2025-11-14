
# st.caption("Simulated autoscaling decisions powered by ML predictions. Upload your own workload to explore adaptive scaling!")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import torch
from torch import nn
import time
import io
import joblib
import os

# Streamlit Config

st.set_page_config(page_title="☁️ ML Autoscaling Simulator", layout="wide")

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Data Loading and Normalization
@st.cache_data
def load_data(file_path="D:/cloud proj/data/aggregated_cpu.csv"):
    df = pd.read_csv(file_path)

    # Normalize timestamp column
    if "Timestamp [ms];" in df.columns:
        df["timestamp"] = pd.to_datetime(df["Timestamp [ms];"], unit="ms")
        df = df.drop(columns=["Timestamp [ms];"])
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        # auto-detect other time column names
        found = False
        for c in df.columns:
            if "time" in c.lower() or "date" in c.lower():
                df["timestamp"] = pd.to_datetime(df[c])
                found = True
                break
        if not found:
            st.warning("No timestamp column found — creating synthetic timestamps.")
            df["timestamp"] = pd.date_range(start="2020-01-01", periods=len(df), freq="H")

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

data = load_data()

# ML Models

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_random_forest(df):
    df = df.copy()
    df['target'] = df['cpu_pct'].shift(-1)
    df = df.dropna()

    X = df[['cpu_pct']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X)
    df['rf_pred'] = preds

    joblib.dump(model, "models/rf_model.pkl")
    return df, model

def train_lstm(df, seq_len=10, epochs=5):
    df = df.copy()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['cpu_pct']])

    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i+seq_len])
        y.append(scaled[i+seq_len])

    X = np.array(X)
    y = np.array(y)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMModel().to(device)
    X, y = X.to(device), y.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    progress = st.progress(0)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        progress.progress((epoch + 1) / epochs)

    model.eval()
    preds = model(X).detach().cpu().numpy()
    preds = scaler.inverse_transform(preds)

    df = df.iloc[seq_len:].copy()
    df['lstm_pred'] = preds
    torch.save(model.state_dict(), "models/lstm_model.pt")
    return df, model

# Autoscaling Simulator

def simulate_autoscaling(actual, predicted, threshold=70, cooldown=5, live=False):
    instances, cooldown_timer, cost, violations = 1, 0, 0, 0
    history = []
    placeholder = st.empty() if live else None

    for t in range(len(predicted)):
        per_instance_load = actual[t] / instances
        cost += instances * 0.01
        violations += int(per_instance_load > threshold)

        if cooldown_timer > 0:
            cooldown_timer -= 1
        else:
            if predicted[t] > threshold:
                instances += 1
                cooldown_timer = cooldown
            elif predicted[t] < 40 and instances > 1:
                instances -= 1
                cooldown_timer = cooldown

        history.append(instances)

        if live:
            with placeholder.container():
                st.metric("Active Instances", instances)
                st.progress(float(min(float(predicted[t]) / 100, 1.0)))
            time.sleep(0.05)

    return history, cost, violations

# Streamlit UI

st.title("☁️ ML-Driven Cloud Autoscaling Simulator")

# uploaded = st.file_uploader("📂 Upload CSV (must include 'cpu_pct' + timestamp column)", type=["csv"])
# if uploaded:
#     df = pd.read_csv(uploaded)
#     if "Timestamp [ms];" in df.columns:
#         df["timestamp"] = pd.to_datetime(df["Timestamp [ms];"], unit="ms")
#         df.drop(columns=["Timestamp [ms];"], inplace=True)
#     elif "timestamp" in df.columns:
#         df["timestamp"] = pd.to_datetime(df["timestamp"])
#     else:
#         st.error("No timestamp column found. Please include 'timestamp' or 'Timestamp [ms];'.")
#         st.stop()
#     st.success("✅ Custom workload uploaded successfully!")
# else:
#     df = data.copy()

uploaded = st.file_uploader("📂 Upload CSV (semicolon or comma separated)", type=["csv"])
if uploaded:
    try:
        # Read with semicolon separator and handle BOM
        df = pd.read_csv(uploaded, sep=';', engine='python', encoding='utf-8-sig')
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    # Clean column names
    df.columns = df.columns.str.replace('"', '').str.strip()

    # Fix common timestamp names
    if "Timestamp [ms]" in df.columns:
        df["timestamp"] = pd.to_datetime(df["Timestamp [ms]"], unit="ms", errors="coerce")
        df.drop(columns=["Timestamp [ms]"], inplace=True)
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        st.error("No timestamp column found (expected 'Timestamp [ms]'). Detected columns:")
        st.write(df.columns.tolist())
        st.stop()

    # Detect CPU usage column
    if "CPU usage [%]" in df.columns:
        df["cpu_pct"] = df["CPU usage [%]"].astype(float)
    elif "cpu_pct" in df.columns:
        df["cpu_pct"] = df["cpu_pct"].astype(float)
    else:
        st.error("Could not find CPU usage column (expected 'CPU usage [%]'). Detected columns:")
        st.write(df.columns.tolist())
        st.stop()

    df = df.sort_values("timestamp").reset_index(drop=True)
    st.success("✅ Custom workload uploaded successfully!")
else:
    df = data.copy()



colA, colB = st.columns(2)
retrain = colA.button("🔁 Retrain Models")
live_mode = colB.toggle("🟢 Live Autoscaling Simulation")

if retrain:
    with st.spinner("Training Random Forest..."):
        rf_df, rf_model = train_random_forest(df)
    with st.spinner("Training LSTM..."):
        lstm_df, lstm_model = train_lstm(df)
else:
    rf_df, _ = train_random_forest(df)
    lstm_df, _ = train_lstm(df)

# Merge predictions
merged = pd.merge(rf_df[['timestamp', 'rf_pred']],
                  lstm_df[['timestamp', 'lstm_pred']],
                  on='timestamp', how='outer')
merged = pd.merge(df, merged, on='timestamp', how='outer').sort_values('timestamp').ffill()

# Simulation Controls

threshold = st.slider("⚙️ Scaling Threshold (%)", 40, 90, 70)
cooldown = st.slider("🕒 Cooldown Period", 1, 10, 5)
model_choice = st.selectbox("🤖 Prediction Model", ["Random Forest", "LSTM"])

pred_col = "rf_pred" if model_choice == "Random Forest" else "lstm_pred"

instances, cost, violations = simulate_autoscaling(
    merged["cpu_pct"].values,
    merged[pred_col].values,
    threshold,
    cooldown,
    live=live_mode
)
merged["instances"] = instances


# Visualization

tab1, tab2 = st.tabs(["📈 CPU Forecast", "📊 Autoscaling Behavior"])

with tab1:
    fig = px.line(merged, x="timestamp", y=["cpu_pct", pred_col],
                  title=f"{model_choice} Predictions vs Actual CPU Utilization")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig2 = px.line(merged, x="timestamp", y="instances",
                   title="Active VM Instances Over Time")
    st.plotly_chart(fig2, use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("💰 Estimated Cost", f"${cost:.2f}")
col2.metric("⚠️ SLA Violations", f"{violations}")
col3.metric("🧠 Model", model_choice)

# Download Predictions

csv_buf = io.StringIO()
merged.to_csv(csv_buf, index=False)
st.download_button("⬇️ Download Predictions CSV", csv_buf.getvalue(),
                   "predictions.csv", "text/csv")

st.caption("💡 ML-powered autoscaling simulator — retrain models, upload workloads, and visualize real-time scaling.")


