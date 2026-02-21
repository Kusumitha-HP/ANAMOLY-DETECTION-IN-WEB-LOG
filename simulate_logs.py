"""import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# -----------------------------
# Setup
# -----------------------------
os.makedirs("data", exist_ok=True)

# Simulation parameters
num_users = 50
num_products = 50
num_events = 1000
start_time = datetime.now() - timedelta(hours=1)

# Event types and attributes
event_types = ['view', 'add_to_cart', 'checkout', 'login']
devices = ['mobile', 'desktop']
regions = ['North', 'South', 'East', 'West']

logs = []

# -----------------------------
# Step 1: Normal random events
# -----------------------------
for i in range(num_events):
    timestamp = start_time + timedelta(seconds=random.randint(0, 3600))
    user_id = random.randint(1, num_users)
    product_id = random.randint(1, num_products)
    ip_address = f"192.168.{random.randint(0,255)}.{random.randint(1,254)}"
    event_type = random.choices(event_types, weights=[50, 30, 15, 5])[0]
    login_status = random.choice(['success', 'failure']) if event_type == 'login' else None
    response_time = round(np.random.exponential(0.5), 2)
    device_type = random.choice(devices)
    region = random.choice(regions)

    logs.append([timestamp, user_id, ip_address, product_id, event_type, login_status, response_time, device_type, region])

# -----------------------------
# Step 2: Inject DDoS Attack
# -----------------------------
for malicious_ip in ['192.168.0.250', '192.168.0.251']:
    for i in range(40):  # >30 requests triggers DDoS
        logs.append([
            start_time + timedelta(seconds=i),
            random.randint(1, 50),
            malicious_ip,
            random.randint(1, 50),
            'view',
            None,
            round(np.random.exponential(0.5), 2),
            random.choice(devices),
            random.choice(regions)
        ])

# -----------------------------
# Step 3: Inject Brute Force Login
# -----------------------------
for i in range(6):  # >5 failed logins triggers Brute Force
    logs.append([
        start_time + timedelta(seconds=i),
        99,  # specific user
        '192.168.0.252',
        random.randint(1, 50),
        'login',
        'failure',
        round(np.random.exponential(0.5), 2),
        'desktop',
        'East'
    ])

# -----------------------------
# Step 4: Inject High Response Time
# -----------------------------
for i in range(5):
    logs.append([
        start_time + timedelta(seconds=i),
        random.randint(1, 50),
        f'192.168.{random.randint(1,255)}.{random.randint(1,254)}',
        random.randint(1, 50),
        'checkout',
        None,
        round(random.uniform(3.5, 5.0), 2),  # >3 sec
        random.choice(devices),
        random.choice(regions)
    ])

# -----------------------------
# Step 5: Create DataFrame
# -----------------------------
df = pd.DataFrame(logs, columns=[
    'timestamp', 'user_id', 'ip_address', 'product_id', 'event_type',
    'login_status', 'response_time', 'device_type', 'region'
])

# Sort by timestamp
df = df.sort_values('timestamp')

# -----------------------------
# Step 6: Save CSV
# -----------------------------
file_path = "data/simulated_logs.csv"
df.to_csv(file_path, index=False)
print(f"✅ Simulated enriched logs saved to: {os.path.abspath(file_path)}")
print(df.head(10))
"""
"""
import pandas as pd
import random
from datetime import datetime, timedelta

# Parameters
num_rows = 5000
regions = ['North', 'South', 'East', 'West']
device_types = ['Mobile', 'Desktop', 'Tablet']
event_types = ['view', 'click', 'purchase', 'login']
product_ids = [f'P{i:03d}' for i in range(1, 51)]  # 50 products
login_status_options = ['success', 'failed']

# Generate normal data
data = []
start_time = datetime.now() - timedelta(days=30)

for i in range(num_rows):
    timestamp = start_time + timedelta(minutes=random.randint(0, 30*24*60))
    region = random.choice(regions)
    device_type = random.choice(device_types)
    product_id = random.choice(product_ids)
    event_type = random.choice(event_types)
    login_status = random.choice(login_status_options) if event_type=='login' else 'success'
    response_time = round(random.uniform(0.1, 2.0), 2)  # seconds

    data.append([timestamp, region, device_type, product_id, event_type, login_status, response_time])

# Introduce anomalies (high response_time or repeated failed login)
anomalous_users = [f'U{i:03d}' for i in range(1, 11)]
for user in anomalous_users:
    num_anomalies = random.randint(2,4)
    for _ in range(num_anomalies):
        timestamp = start_time + timedelta(minutes=random.randint(0, 30*24*60))
        region = random.choice(regions)
        device_type = random.choice(device_types)
        product_id = random.choice(product_ids)
        event_type = 'login'
        login_status = 'failed'
        response_time = round(random.uniform(5.0, 10.0), 2)  # unusually high response time
        data.append([timestamp, region, device_type, product_id, event_type, login_status, response_time])

# Create DataFrame
df = pd.DataFrame(data, columns=['timestamp', 'region', 'device_type', 'product_id', 'event_type', 'login_status', 'response_time'])

# Save to Excel
df.to_excel('synthetic_logs_fixed.xlsx', index=False)

print("Synthetic Excel file 'synthetic_logs_fixed.xlsx' created with all required columns and anomalies.")
"""
import pandas as pd
import random
from datetime import datetime, timedelta
import os

# ----------------- Config -----------------
NUM_ROWS = 5000                 # normal log entries
NUM_ANOMALY_USERS = 10          # users with anomalies
ANOMALIES_PER_USER = (2, 5)

REGIONS = ['North', 'South', 'East', 'West']
DEVICE_TYPES = ['Mobile', 'Desktop', 'Tablet']
EVENT_TYPES = ['view', 'click', 'purchase', 'login']
PRODUCT_IDS = [f'P{i:03d}' for i in range(1, 51)]
USER_IDS = [f'U{i:03d}' for i in range(1, 201)]

OUTPUT_CSV = 'synthetic_logs_enhanced.csv'
OUTPUT_XLSX = 'synthetic_logs_enhanced.xlsx'

# ----------------- IP generator -----------------
def generate_ip():
    return ".".join(str(random.randint(1, 255)) for _ in range(4))

# ----------------- Normal logs -----------------
data = []
start_time = datetime.now() - timedelta(days=30)

for _ in range(NUM_ROWS):
    timestamp = start_time + timedelta(minutes=random.randint(0, 30*24*60))
    user_id = random.choice(USER_IDS)
    ip_address = generate_ip()
    region = random.choice(REGIONS)
    device_type = random.choice(DEVICE_TYPES)
    product_id = random.choice(PRODUCT_IDS)
    event_type = random.choice(EVENT_TYPES)
    login_status = random.choice(['success', 'failed']) if event_type=='login' else 'success'
    response_time = round(random.uniform(0.1, 2.0), 2)
    data.append([timestamp, user_id, ip_address, region, device_type, product_id, event_type, login_status, response_time, 'normal'])

# ----------------- Anomalies -----------------
for user in USER_IDS[:NUM_ANOMALY_USERS]:
    for _ in range(random.randint(*ANOMALIES_PER_USER)):
        timestamp = start_time + timedelta(minutes=random.randint(0, 30*24*60))
        ip_address = generate_ip()
        region = random.choice(REGIONS)
        device_type = random.choice(DEVICE_TYPES)
        product_id = random.choice(PRODUCT_IDS)
        anomaly_type = random.choice(['high_response', 'failed_login', 'mass_clicks', 'region_device', 'ip_abuse', 'ddos', 'sql_injection', 'intrusion'])

        # Initialize defaults
        event_type = random.choice(EVENT_TYPES)
        login_status = 'success'
        response_time = round(random.uniform(0.1, 2.0), 2)

        if anomaly_type == 'high_response':
            response_time = round(random.uniform(5.0, 10.0), 2)
        elif anomaly_type == 'failed_login':
            event_type = 'login'
            login_status = 'failed'
        elif anomaly_type == 'mass_clicks':
            event_type = 'click'
        elif anomaly_type == 'region_device':
            device_type = 'Tablet' if region in ['North','South'] else 'Mobile'
        elif anomaly_type == 'ip_abuse':
            ip_address = '192.168.0.100'  # repeated IP for multiple events
        elif anomaly_type == 'ddos':
            ip_address = '203.0.113.1'
            # Generate multiple events in short time
            for _ in range(random.randint(20,50)):
                ts = timestamp + timedelta(seconds=random.randint(0,60))
                data.append([ts, user, ip_address, region, device_type, product_id, random.choice(EVENT_TYPES), 'success', round(random.uniform(0.1, 2.0),2), 'ddos'])
            continue  # skip normal append to avoid duplicate
        elif anomaly_type == 'sql_injection':
            product_id = random.choice(["P001", "P002", "' OR '1'='1", "'; DROP TABLE logs;--"])
            event_type = 'purchase'
        elif anomaly_type == 'intrusion':
            event_type = 'login'
            login_status = 'failed'
            ip_address = generate_ip()

        data.append([timestamp, user, ip_address, region, device_type, product_id, event_type, login_status, response_time, anomaly_type])

# ----------------- Create DataFrame -----------------
df = pd.DataFrame(data, columns=[
    'timestamp','user_id','ip_address','region','device_type','product_id',
    'event_type','login_status','response_time','anomaly_type'
])

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ----------------- Save files -----------------
df.to_csv(OUTPUT_CSV, index=False)
df.to_excel(OUTPUT_XLSX, index=False)

print(f"✅ Enhanced synthetic logs generated with multiple anomaly types:\nCSV -> {OUTPUT_CSV}\nExcel -> {OUTPUT_XLSX}")
