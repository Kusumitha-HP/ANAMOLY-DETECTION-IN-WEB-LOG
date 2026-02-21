# scripts/block_suspicious_ips.py
import pandas as pd
import numpy as np
import plotly.express as px
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Step 1: Load Logs & Models
# -----------------------------
logs = pd.read_csv("data/simulated_logs.csv", parse_dates=["timestamp"])

autoencoder = load_model("models/autoencoder.h5")
lstm = load_model("models/lstm.h5")

# For event encoding
encoder = LabelEncoder()
logs["event_code"] = encoder.fit_transform(logs["event_type"])

# -----------------------------
# Step 2: Autoencoder Anomaly Detection
# -----------------------------
features = ["response_time"]
X = logs[features].fillna(0).values

reconstructed = autoencoder.predict(X, verbose=0)
mse = np.mean(np.power(X - reconstructed, 2), axis=1)
logs["autoencoder_score"] = mse
logs["autoencoder_anomaly"] = logs["autoencoder_score"] > np.percentile(mse, 95)  # top 5% anomalies

# -----------------------------
# Step 3: LSTM Anomaly Detection (sequence-based)
# -----------------------------
seq_len = 5
lstm_flags = []

for _, group in logs.groupby("user_id"):
    seq = group["event_code"].tolist()
    for i in range(len(seq) - seq_len):
        X_seq = np.array([seq[i:i+seq_len]])
        pred = np.argmax(lstm.predict(X_seq, verbose=0))
        if pred != seq[i+seq_len]:
            lstm_flags.append(True)
        else:
            lstm_flags.append(False)

# Pad flags to match length
logs["lstm_anomaly"] = False
logs.loc[logs.index[-len(lstm_flags):], "lstm_anomaly"] = lstm_flags

# -----------------------------
# Step 4: Classify Anomalies
# -----------------------------
def classify_anomaly(row):
    if row["autoencoder_anomaly"]:
        if row["response_time"] > 2.0:
            return "High Response Time", "Investigate Server / Optimize"
        elif "login_failed" in row["event_type"]:
            return "Brute Force Attack", "Block IP & Alert Security"
        elif row.get("total_requests", 0) > 100:
            return "DDoS Attack", "Rate-limit Requests / Block IP"
    if row["lstm_anomaly"]:
        return "Suspicious Behavior", "Monitor User Activity"
    return None, None

logs[["anomaly_name", "recommendation"]] = logs.apply(classify_anomaly, axis=1, result_type="expand")

# -----------------------------
# Step 5: Aggregate by IP
# -----------------------------
report = (
    logs.dropna(subset=["anomaly_name"])
    .groupby("ip_address")[["anomaly_name", "recommendation"]]
    .agg(lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])
    .reset_index()
)

report["count"] = logs.groupby("ip_address")["anomaly_name"].count().values
output_path = "data/anomaly_classification_report.csv"
report.to_csv(output_path, index=False)

print(f"\n✅ Anomaly classification report saved to: {os.path.abspath(output_path)}")
print("\nSample of anomaly classification report:")
print(report.head())

# -----------------------------
# Step 6: Visualization
# -----------------------------
if not report.empty:
    fig = px.bar(
        report,
        x="ip_address",
        y="count",
        color="anomaly_name",
        title="🔍 Anomaly Distribution by IP",
        labels={"count": "Number of Anomalies", "ip_address": "IP Address"}
    )
    fig.show()
