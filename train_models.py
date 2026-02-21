# models/train_models.py

import pandas as pd
import os
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------- Paths -----------------
DATA_CSV = "synthetic_logs_enhanced.csv"       # Use the new enhanced dataset
IFOREST_MODEL_PATH = os.path.join("models", "isolation_forest.pkl")
AUTOENCODER_MODEL_PATH = os.path.join("models", "autoencoder_model.h5")
LSTM_MODEL_PATH = os.path.join("models", "lstm_model.h5")

# ----------------- Load Data -----------------
def load_data():
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"{DATA_CSV} not found. Generate synthetic logs first.")

    df = pd.read_csv(DATA_CSV)
    df.columns = df.columns.str.strip()  # clean column names

    # Required columns for ML
    required_cols = ['response_time', 'user_id', 'product_id', 'ip_address', 'device_type', 'region']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in dataset: {missing}")

    # Convert categorical features to numeric codes
    for col in ['user_id', 'product_id', 'ip_address', 'device_type', 'region']:
        df[col] = df[col].astype('category').cat.codes

    # Features to use for models
    features = ['response_time', 'user_id', 'product_id', 'ip_address', 'device_type', 'region']
    return df[features], df.get('anomaly_type')  # anomaly_type optional for evaluation

# ----------------- Isolation Forest -----------------
def train_isolation_forest(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X_scaled)

    joblib.dump((model, scaler), IFOREST_MODEL_PATH)
    print(f"✅ Isolation Forest saved at {IFOREST_MODEL_PATH}")

# ----------------- Autoencoder -----------------
def train_autoencoder(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    input_dim = X_scaled.shape[1]

    model = keras.Sequential([
        layers.Dense(32, activation="relu", input_shape=(input_dim,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(input_dim, activation="linear")
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_scaled, X_scaled, epochs=15, batch_size=32, verbose=1)

    model.save(AUTOENCODER_MODEL_PATH)
    joblib.dump(scaler, "models/autoencoder_scaler.pkl")
    print(f"✅ Autoencoder saved at {AUTOENCODER_MODEL_PATH}")

# ----------------- LSTM Autoencoder -----------------
def train_lstm_autoencoder(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # reshape for LSTM (samples, timesteps, features)
    X_seq = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    input_dim = X_seq.shape[2]

    model = keras.Sequential([
        layers.LSTM(32, activation="relu", input_shape=(1, input_dim), return_sequences=True),
        layers.LSTM(16, activation="relu", return_sequences=False),
        layers.RepeatVector(1),
        layers.LSTM(16, activation="relu", return_sequences=True),
        layers.LSTM(32, activation="relu", return_sequences=True),
        layers.TimeDistributed(layers.Dense(input_dim))
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_seq, X_seq, epochs=15, batch_size=32, verbose=1)

    model.save(LSTM_MODEL_PATH)
    joblib.dump(scaler, "models/lstm_scaler.pkl")
    print(f"✅ LSTM Autoencoder saved at {LSTM_MODEL_PATH}")

# ----------------- Main -----------------
if __name__ == "__main__":
    print("🔹 Loading data...")
    X, y = load_data()
    print(f"Data loaded with {X.shape[0]} rows and {X.shape[1]} features.")

    print("\n=== Training Isolation Forest ===")
    train_isolation_forest(X)

    print("\n=== Training Autoencoder ===")
    train_autoencoder(X)

    print("\n=== Training LSTM Autoencoder ===")
    train_lstm_autoencoder(X)

    print("\n🎉 All models trained and saved successfully!")
