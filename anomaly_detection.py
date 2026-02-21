import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load preprocessed data
df = pd.read_csv('data/preprocessed_logs.csv', index_col=0, parse_dates=True)
data = df.values

# -----------------------------
# Step 1: Autoencoder for anomaly detection
# -----------------------------
input_dim = data.shape[1]

autoencoder = Sequential([
    Dense(16, activation='relu', input_shape=(input_dim,)),
    Dense(8, activation='relu'),
    Dense(16, activation='relu'),
    Dense(input_dim, activation='sigmoid')
])

autoencoder.compile(optimizer='adam', loss='mse')

# Train autoencoder on all data (assuming mostly normal)
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
autoencoder.fit(data, data, epochs=50, batch_size=32, callbacks=[early_stop], verbose=1)

# Reconstruction error
reconstructions = autoencoder.predict(data)
mse = np.mean(np.power(data - reconstructions, 2), axis=1)

# Set threshold for anomaly
threshold = np.percentile(mse, 95)
anomalies = mse > threshold
print(f"Autoencoder detected {np.sum(anomalies)} anomalies out of {len(data)} records.")

# -----------------------------
# Step 2: LSTM for sequence anomaly detection
# -----------------------------
timesteps = 5
X = []
for i in range(len(data) - timesteps):
    X.append(data[i:i+timesteps])
X = np.array(X)

lstm = Sequential([
    LSTM(32, activation='relu', input_shape=(timesteps, input_dim), return_sequences=False),
    Dense(timesteps * input_dim),
])
lstm.compile(optimizer='adam', loss='mse')

early_stop_lstm = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
lstm.fit(X, X.reshape(X.shape[0], -1), epochs=50, batch_size=16, callbacks=[early_stop_lstm], verbose=1)

# LSTM reconstruction error
lstm_recon = lstm.predict(X)
lstm_mse = np.mean(np.power(X.reshape(X.shape[0], -1) - lstm_recon, 2), axis=1)
lstm_threshold = np.percentile(lstm_mse, 95)
lstm_anomalies = lstm_mse > lstm_threshold
print(f"LSTM detected {np.sum(lstm_anomalies)} anomalies out of {len(lstm_mse)} sequences.")

# -----------------------------
# Step 3: Save anomaly results
# -----------------------------
df_anomaly = df.iloc[timesteps:].copy()
df_anomaly['autoencoder_anomaly'] = anomalies[timesteps:]
df_anomaly['lstm_anomaly'] = lstm_anomalies
df_anomaly.to_csv('data/anomaly_results.csv')
print("✅ Saved anomaly results to data/anomaly_results.csv")

# -----------------------------
# Optional: Plot anomaly scores
# -----------------------------
plt.figure(figsize=(10,4))
plt.plot(mse, label='Autoencoder MSE')
plt.plot(np.arange(timesteps, len(data)), lstm_mse, label='LSTM MSE')
plt.axhline(y=threshold, color='r', linestyle='--', label='AE threshold')
plt.axhline(y=lstm_threshold, color='g', linestyle='--', label='LSTM threshold')
plt.legend()
plt.show()
