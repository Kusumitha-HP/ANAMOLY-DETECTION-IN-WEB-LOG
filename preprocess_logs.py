import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the simulated logs
df = pd.read_csv('data/simulated_logs.csv', parse_dates=['timestamp'])

# Set timestamp as index
df.set_index('timestamp', inplace=True)

# Resample data per minute
agg_df = df.resample('1Min').agg({
    'user_id': pd.Series.nunique,        # Unique users
    'product_id': 'count',               # Total events
    'response_time': ['mean','max']      # Response time stats
})

# Flatten MultiIndex columns
agg_df.columns = ['unique_users', 'total_events', 'mean_response', 'max_response']

# Count event types
event_counts = df.groupby([pd.Grouper(freq='1Min'),'event_type']).size().unstack(fill_value=0)
agg_df = agg_df.join(event_counts)

# Fill NaN with 0
agg_df.fillna(0, inplace=True)

# Normalize for ML
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(agg_df)
scaled_df = pd.DataFrame(scaled_data, columns=agg_df.columns, index=agg_df.index)

# Save preprocessed data
scaled_df.to_csv('data/preprocessed_logs.csv')
print("✅ Preprocessing complete! Saved preprocessed_logs.csv in data folder.")
