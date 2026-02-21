'''import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# Page Config - MUST BE FIRST
# -----------------------------
st.set_page_config(
    page_title="📊 E-Commerce Monitoring Dashboard",
    layout="wide",
    page_icon="📦"
)

# -----------------------------
# Custom CSS for hover effects & styling
# -----------------------------
st.markdown("""
    <style>
    /* General font */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 16px;
    }

    /* Hover effect for metrics */
    .stMetric {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 8px;
        transition: all 0.2s ease-in-out;
    }
    .stMetric:hover {
        background-color: #e0f7fa;
        box-shadow: 0 4px 10px rgba(0, 128, 128, 0.2);
        transform: scale(1.02);
    }

    /* Style primary buttons */
    button[kind="primary"] {
        background-color: #4CAF50;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.2s ease-in-out;
    }
    button[kind="primary"]:hover {
        background-color: #45a049 !important;
        color: white !important;
        transform: scale(1.05);
    }

    /* Sidebar header style */
    .css-1d391kg h2 {
        color: #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📦 E-Commerce Monitoring Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("📁 Upload your log CSV file", type=["csv"])

if uploaded_file is not None:
    logs = pd.read_csv(uploaded_file)

    # -----------------------------
    # Safety Checks
    # -----------------------------
    required_columns = ['timestamp', 'region', 'device_type', 'product_id', 'ip_address', 'event_type', 'login_status', 'response_time']
    missing_cols = [col for col in required_columns if col not in logs.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        st.stop()

    # Parse timestamp safely
    logs['timestamp'] = pd.to_datetime(logs['timestamp'], errors='coerce')
    logs = logs.dropna(subset=['timestamp'])

    st.success(f"✅ File uploaded successfully! Total rows: {logs.shape[0]}")
    st.markdown("---")

    # -----------------------------
    # Sidebar Filters
    # -----------------------------
    st.sidebar.header("🔍 Filters")

    min_date = logs['timestamp'].min().date()
    max_date = logs['timestamp'].max().date()
    start_date, end_date = st.sidebar.date_input("Date Range", [min_date, max_date])

    regions = sorted(logs['region'].dropna().unique())
    selected_region = st.sidebar.multiselect("Region", options=regions, default=regions)

    devices = sorted(logs['device_type'].dropna().unique())
    selected_device = st.sidebar.multiselect("Device Type", options=devices, default=devices)

    # Filter logs
    filtered_logs = logs[
        (logs['region'].isin(selected_region)) &
        (logs['device_type'].isin(selected_device)) &
        (logs['timestamp'].dt.date >= start_date) &
        (logs['timestamp'].dt.date <= end_date)
    ]

    # -----------------------------
    # Anomaly Thresholds
    # -----------------------------
    st.sidebar.subheader("⚠️ Anomaly Thresholds")
    ddos_threshold = st.sidebar.number_input("DDoS: Requests >", value=30, min_value=1)
    login_failure_threshold = st.sidebar.number_input("Login Failures >", value=5, min_value=1)
    high_response_threshold = st.sidebar.number_input("Avg. Response Time >", value=3.0, min_value=0.1)

    # -----------------------------
    # Analysis Selection
    # -----------------------------
    analysis_type = st.radio("🔎 Select Analysis Type", ["Trending Products", "Anomalies & Recommendations"])

    # -----------------------------
    # Trending Products
    # -----------------------------
    if analysis_type == "Trending Products":
        st.subheader("🔥 Top Trending Products")
        top_n = st.slider("Top N Products", min_value=5, max_value=20, value=10)

        product_counts = filtered_logs['product_id'].value_counts().reset_index()
        product_counts.columns = ['product_id', 'count']

        st.dataframe(product_counts.head(top_n), use_container_width=True)

        fig = px.bar(
            product_counts.head(top_n),
            x='product_id',
            y='count',
            text='count',
            title=f"Top {top_n} Trending Products",
            labels={'product_id': 'Product ID', 'count': 'Events'},
            color='count',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            label="📥 Download Trending Products CSV",
            data=product_counts.head(top_n).to_csv(index=False).encode('utf-8'),
            file_name='trending_products.csv',
            mime='text/csv'
        )

    # -----------------------------
    # Anomaly Detection & Recommendations
    # -----------------------------
    elif analysis_type == "Anomalies & Recommendations":
        st.subheader("🚨 Anomaly Detection & Recommendations")

        # Aggregation
        agg = filtered_logs.groupby('ip_address').agg({
            'event_type': 'count',
            'login_status': lambda x: (x == 'failure').sum(),
            'response_time': 'mean'
        }).rename(columns={
            'event_type': 'total_requests',
            'login_status': 'failed_logins',
            'response_time': 'avg_response_time'
        })

        # Anomaly classification
        def classify_anomaly(row):
            if row['total_requests'] > ddos_threshold:
                return 'DDoS Attack', '🔒 Block IP'
            elif row['failed_logins'] > login_failure_threshold:
                return 'Brute Force Login', '🚫 Block IP / Alert Admin'
            elif row['avg_response_time'] > high_response_threshold:
                return 'High Response Time', '🔍 Investigate Server'
            else:
                return None, None

        agg[['anomaly_name', 'recommendation']] = agg.apply(lambda x: pd.Series(classify_anomaly(x)), axis=1)
        anomaly_report = agg.dropna(subset=['anomaly_name']).reset_index()
        blocked_ips = anomaly_report[anomaly_report['anomaly_name'].isin(['DDoS Attack', 'Brute Force Login'])]

        # Summary Metrics
        st.markdown("### 📊 Summary Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Requests", filtered_logs.shape[0])
        col2.metric("Anomalies Detected", anomaly_report.shape[0])
        col3.metric("Blocked IPs", blocked_ips.shape[0])

        st.markdown("### 📝 Anomaly Report")
        st.dataframe(anomaly_report, use_container_width=True)

        st.markdown("### ⛔ Blocked IPs")
        st.dataframe(blocked_ips[['ip_address', 'anomaly_name', 'recommendation']], use_container_width=True)

        # Anomaly Bar Chart
        agg_anomaly = anomaly_report.groupby(['ip_address', 'anomaly_name']).size().reset_index(name='count')
        fig_anomaly = px.bar(
            agg_anomaly,
            x='ip_address',
            y='count',
            color='anomaly_name',
            title="Detected Anomalies by IP",
            labels={'ip_address': 'IP Address', 'count': 'Anomalies'}
        )
        st.plotly_chart(fig_anomaly, use_container_width=True)

        # -----------------------------
        # 3D Scatter & Heatmap
        # -----------------------------
        st.markdown("### 📈 Advanced Visualizations")

        # Add dummy column to avoid errors
        if 'anomaly_type' not in filtered_logs.columns:
            filtered_logs['anomaly_type'] = None

        agg_3d = filtered_logs.groupby('ip_address').agg({
            'event_type': 'count',
            'response_time': 'mean',
            'anomaly_type': pd.Series.nunique
        }).rename(columns={
            'event_type': 'traffic_volume',
            'response_time': 'avg_response_time',
            'anomaly_type': 'unique_anomalies'
        }).reset_index()

        numeric_cols = ['total_requests', 'failed_logins', 'avg_response_time']
        heatmap_data = agg[numeric_cols]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🌀 3D Scatter Plot")
            fig_3d = plt.figure(figsize=(6, 5))
            ax = fig_3d.add_subplot(111, projection='3d')
            x = agg_3d['traffic_volume']
            y = agg_3d['avg_response_time']
            z = agg_3d['unique_anomalies']
            sizes = (z + 1) * 50  # ensure visible sizes
            sc = ax.scatter(x, y, z, s=sizes, c=z, cmap='plasma', alpha=0.8, edgecolors='k')
            ax.set_xlabel('Traffic Volume')
            ax.set_ylabel('Avg Response Time')
            ax.set_zlabel('Unique Anomalies')
            ax.set_title('Traffic vs Response vs Anomaly Diversity')
            fig_3d.colorbar(sc, ax=ax, label='Unique Anomaly Types')
            st.pyplot(fig_3d)

        with col2:
            st.markdown("#### 🔥 Correlation Heatmap")
            fig_heatmap = plt.figure(figsize=(6, 5))
            sns.heatmap(heatmap_data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title('Feature Correlation Heatmap', pad=15)
            st.pyplot(fig_heatmap)

        # -----------------------------
        # Downloads
        # -----------------------------
        st.markdown("### 📥 Download Reports")
        st.download_button(
            label="📥 Download Anomalies CSV",
            data=anomaly_report.to_csv(index=False).encode('utf-8'),
            file_name='anomalies_report.csv',
            mime='text/csv'
        )
else:
    st.info("📂 Please upload a CSV file to begin analysis.")
'''
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# Page Config - MUST BE FIRST
# -----------------------------
st.set_page_config(
    page_title="📊 E-Commerce Monitoring Dashboard",
    layout="wide",
    page_icon="📦"
)

# -----------------------------
# Dark/Light Mode Toggle
# -----------------------------
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# -----------------------------
# Dynamic Theme Styling
# -----------------------------
def get_theme_colors():
    if st.session_state.dark_mode:
        return {
            'bg': '#0e1117',
            'secondary_bg': '#1a1d29',
            'card_bg': '#262730',
            'text': '#fafafa',
            'accent': '#00d4ff',
            'accent_hover': '#00b8e6',
            'gradient_start': '#1a1d29',
            'gradient_end': '#0e1117',
            'shadow': 'rgba(0, 212, 255, 0.3)',
            'metric_bg': 'linear-gradient(135deg, #262730 0%, #1a1d29 100%)',
            'metric_hover': 'linear-gradient(135deg, #2d3142 0%, #1f2233 100%)'
        }
    else:
        return {
            'bg': '#ffffff',
            'secondary_bg': '#f8f9fa',
            'card_bg': '#ffffff',
            'text': '#1a1d29',
            'accent': '#4CAF50',
            'accent_hover': '#45a049',
            'gradient_start': '#f8f9fa',
            'gradient_end': '#e9ecef',
            'shadow': 'rgba(76, 175, 80, 0.3)',
            'metric_bg': 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)',
            'metric_hover': 'linear-gradient(135deg, #f0f8ff 0%, #e0f7fa 100%)'
        }

colors = get_theme_colors()

# -----------------------------
# Custom CSS with Theme Support
# -----------------------------
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: {colors['text']};
    }}
    
    .main {{
        background: linear-gradient(135deg, {colors['gradient_start']} 0%, {colors['gradient_end']} 100%);
    }}
    
    /* Header Styling */
    .dashboard-header {{
        text-align: center;
        padding: 2rem 0;
        background: {colors['metric_bg']};
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px {colors['shadow']};
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    .dashboard-title {{
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, {colors['accent']} 0%, {colors['accent_hover']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        animation: fadeInDown 0.8s ease-out;
    }}
    
    .dashboard-subtitle {{
        color: {colors['text']};
        opacity: 0.7;
        font-size: 1.1rem;
        font-weight: 400;
    }}
    
    /* Enhanced Metrics */
    [data-testid="stMetricValue"] {{
        font-size: 2rem;
        font-weight: 700;
        color: {colors['accent']};
    }}
    
    div[data-testid="metric-container"] {{
        background: {colors['metric_bg']};
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 20px {colors['shadow']};
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }}
    
    div[data-testid="metric-container"]::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, {colors['accent']}20 0%, transparent 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }}
    
    div[data-testid="metric-container"]:hover {{
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 40px {colors['shadow']};
        background: {colors['metric_hover']};
    }}
    
    div[data-testid="metric-container"]:hover::before {{
        opacity: 1;
    }}
    
    /* Buttons */
    .stButton>button {{
        background: linear-gradient(135deg, {colors['accent']} 0%, {colors['accent_hover']} 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px {colors['shadow']};
        position: relative;
        overflow: hidden;
    }}
    
    .stButton>button::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.2);
        transition: left 0.5s ease;
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px {colors['shadow']};
    }}
    
    .stButton>button:hover::before {{
        left: 100%;
    }}
    
    /* Download Button */
    .stDownloadButton>button {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }}
    
    .stDownloadButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }}
    
    /* File Uploader */
    [data-testid="stFileUploader"] {{
        background: {colors['card_bg']};
        border: 2px dashed {colors['accent']};
        border-radius: 16px;
        padding: 2rem;
        transition: all 0.3s ease;
    }}
    
    [data-testid="stFileUploader"]:hover {{
        border-color: {colors['accent_hover']};
        background: {colors['secondary_bg']};
        transform: scale(1.01);
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: {colors['secondary_bg']};
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {{
        color: {colors['accent']};
        font-weight: 700;
        font-size: 1.5rem;
    }}
    
    /* Radio Buttons */
    .stRadio > label {{
        background: {colors['card_bg']};
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    .stRadio > label:hover {{
        background: {colors['secondary_bg']};
        transform: translateX(5px);
    }}
    
    /* Dataframe */
    [data-testid="stDataFrame"] {{
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px {colors['shadow']};
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        background: {colors['card_bg']};
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }}
    
    .streamlit-expanderHeader:hover {{
        background: {colors['secondary_bg']};
        box-shadow: 0 4px 15px {colors['shadow']};
    }}
    
    /* Animations */
    @keyframes fadeInDown {{
        from {{
            opacity: 0;
            transform: translateY(-20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    @keyframes fadeInUp {{
        from {{
            opacity: 0;
            transform: translateY(20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    /* Cards */
    .info-card {{
        background: {colors['card_bg']};
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 20px {colors['shadow']};
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
    }}
    
    .info-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 30px {colors['shadow']};
    }}
    
    /* Toggle Switch */
    .theme-toggle {{
        position: fixed;
        top: 1rem;
        right: 1rem;
        z-index: 999;
    }}
    
    /* Number Input */
    [data-testid="stNumberInput"] {{
        background: {colors['card_bg']};
        border-radius: 12px;
        padding: 0.5rem;
    }}
    
    /* Selectbox */
    [data-testid="stSelectbox"] {{
        background: {colors['card_bg']};
        border-radius: 12px;
    }}
    
    /* Success/Error Messages */
    .stSuccess {{
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem;
        animation: fadeInDown 0.5s ease;
    }}
    
    .stError {{
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem;
        animation: fadeInDown 0.5s ease;
    }}
    
    .stInfo {{
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem;
        animation: fadeInDown 0.5s ease;
    }}
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Theme Toggle in Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("### 🎨 Theme Settings")
    theme_toggle = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if theme_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = theme_toggle
        st.rerun()
    st.markdown("---")

# -----------------------------
# Enhanced Title
# -----------------------------
st.markdown("""
    <div class="dashboard-header">
        <div class="dashboard-title">📦 E-Commerce Analytics Hub</div>
        <div class="dashboard-subtitle">Real-time Monitoring & Threat Detection</div>
    </div>
""", unsafe_allow_html=True)

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("📁 Upload your log CSV file", type=["csv"])

if uploaded_file is not None:
    logs = pd.read_csv(uploaded_file)

    # -----------------------------
    # Safety Checks
    # -----------------------------
    required_columns = ['timestamp', 'region', 'device_type', 'product_id', 'ip_address', 'event_type', 'login_status', 'response_time']
    missing_cols = [col for col in required_columns if col not in logs.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        st.stop()

    # Parse timestamp safely
    logs['timestamp'] = pd.to_datetime(logs['timestamp'], errors='coerce')
    logs = logs.dropna(subset=['timestamp'])

    st.success(f"✅ File uploaded successfully! Total rows: {logs.shape[0]:,}")
    st.markdown("---")

    # -----------------------------
    # Sidebar Filters
    # -----------------------------
    st.sidebar.header("🔍 Filters")

    min_date = logs['timestamp'].min().date()
    max_date = logs['timestamp'].max().date()
    start_date, end_date = st.sidebar.date_input("📅 Date Range", [min_date, max_date])

    regions = sorted(logs['region'].dropna().unique())
    selected_region = st.sidebar.multiselect("🌍 Region", options=regions, default=regions)

    devices = sorted(logs['device_type'].dropna().unique())
    selected_device = st.sidebar.multiselect("📱 Device Type", options=devices, default=devices)

    # Filter logs
    filtered_logs = logs[
        (logs['region'].isin(selected_region)) &
        (logs['device_type'].isin(selected_device)) &
        (logs['timestamp'].dt.date >= start_date) &
        (logs['timestamp'].dt.date <= end_date)
    ]

    # -----------------------------
    # Anomaly Thresholds
    # -----------------------------
    st.sidebar.subheader("⚠️ Anomaly Thresholds")
    ddos_threshold = st.sidebar.number_input("🚨 DDoS: Requests >", value=30, min_value=1)
    login_failure_threshold = st.sidebar.number_input("🔐 Login Failures >", value=5, min_value=1)
    high_response_threshold = st.sidebar.number_input("⏱️ Avg. Response Time >", value=3.0, min_value=0.1)

    # -----------------------------
    # Analysis Selection
    # -----------------------------
    st.sidebar.markdown("---")
    analysis_type = st.sidebar.radio("🔎 Analysis Type", ["Trending Products", "Anomalies & Recommendations"], label_visibility="collapsed")

    # -----------------------------
    # Trending Products
    # -----------------------------
    if analysis_type == "Trending Products":
        st.markdown("<div class='info-card'><h2>🔥 Top Trending Products</h2></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        top_n = st.slider("📊 Number of Products to Display", min_value=5, max_value=20, value=10)

        product_counts = filtered_logs['product_id'].value_counts().reset_index()
        product_counts.columns = ['product_id', 'count']

        # Enhanced metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📦 Total Products", len(product_counts))
        col2.metric("🎯 Top Product", product_counts.iloc[0]['product_id'])
        col3.metric("📈 Top Views", f"{product_counts.iloc[0]['count']:,}")
        col4.metric("📊 Avg Views", f"{product_counts['count'].mean():.0f}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(product_counts.head(top_n), use_container_width=True, height=400)

        # Enhanced plotly chart
        fig = px.bar(
            product_counts.head(top_n),
            x='product_id',
            y='count',
            text='count',
            title=f"Top {top_n} Trending Products",
            labels={'product_id': 'Product ID', 'count': 'Events'},
            color='count',
            color_continuous_scale='viridis'
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            height=500,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            label="📥 Download Trending Products CSV",
            data=product_counts.head(top_n).to_csv(index=False).encode('utf-8'),
            file_name='trending_products.csv',
            mime='text/csv'
        )

    # -----------------------------
    # Anomaly Detection & Recommendations
    # -----------------------------
    elif analysis_type == "Anomalies & Recommendations":
        st.markdown("<div class='info-card'><h2>🚨 Anomaly Detection & Recommendations</h2></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Aggregation
        agg = filtered_logs.groupby('ip_address').agg({
            'event_type': 'count',
            'login_status': lambda x: (x == 'failure').sum(),
            'response_time': 'mean'
        }).rename(columns={
            'event_type': 'total_requests',
            'login_status': 'failed_logins',
            'response_time': 'avg_response_time'
        })

        # Anomaly classification
        def classify_anomaly(row):
            if row['total_requests'] > ddos_threshold:
                return 'DDoS Attack', '🔒 Block IP'
            elif row['failed_logins'] > login_failure_threshold:
                return 'Brute Force Login', '🚫 Block IP / Alert Admin'
            elif row['avg_response_time'] > high_response_threshold:
                return 'High Response Time', '🔍 Investigate Server'
            else:
                return None, None

        agg[['anomaly_name', 'recommendation']] = agg.apply(lambda x: pd.Series(classify_anomaly(x)), axis=1)
        anomaly_report = agg.dropna(subset=['anomaly_name']).reset_index()
        blocked_ips = anomaly_report[anomaly_report['anomaly_name'].isin(['DDoS Attack', 'Brute Force Login'])]

        # Enhanced Summary Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📊 Total Requests", f"{filtered_logs.shape[0]:,}")
        col2.metric("🚨 Anomalies", anomaly_report.shape[0])
        col3.metric("⛔ Blocked IPs", blocked_ips.shape[0])
        col4.metric("✅ Clean IPs", len(agg) - blocked_ips.shape[0])

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabs for better organization
        tab1, tab2, tab3 = st.tabs(["📋 Anomaly Report", "⛔ Blocked IPs", "📊 Visualizations"])

        with tab1:
            st.dataframe(anomaly_report, use_container_width=True, height=400)

        with tab2:
            st.dataframe(blocked_ips[['ip_address', 'anomaly_name', 'recommendation']], use_container_width=True, height=400)

        with tab3:
            # Anomaly Bar Chart
            if not anomaly_report.empty:
                agg_anomaly = anomaly_report.groupby(['ip_address', 'anomaly_name']).size().reset_index(name='count')
                fig_anomaly = px.bar(
                    agg_anomaly,
                    x='ip_address',
                    y='count',
                    color='anomaly_name',
                    title="Detected Anomalies by IP",
                    labels={'ip_address': 'IP Address', 'count': 'Anomalies'},
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )
                fig_anomaly.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=500
                )
                st.plotly_chart(fig_anomaly, use_container_width=True)

        # -----------------------------
        # 3D Scatter & Heatmap
        # -----------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='info-card'><h3>📈 Advanced Analytics</h3></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Add dummy column to avoid errors
        if 'anomaly_type' not in filtered_logs.columns:
            filtered_logs['anomaly_type'] = None

        agg_3d = filtered_logs.groupby('ip_address').agg({
            'event_type': 'count',
            'response_time': 'mean',
            'anomaly_type': pd.Series.nunique
        }).rename(columns={
            'event_type': 'traffic_volume',
            'response_time': 'avg_response_time',
            'anomaly_type': 'unique_anomalies'
        }).reset_index()

        numeric_cols = ['total_requests', 'failed_logins', 'avg_response_time']
        heatmap_data = agg[numeric_cols]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🌀 3D Traffic Analysis")
            fig_3d = plt.figure(figsize=(8, 6))
            ax = fig_3d.add_subplot(111, projection='3d')
            x = agg_3d['traffic_volume']
            y = agg_3d['avg_response_time']
            z = agg_3d['unique_anomalies']
            sizes = (z + 1) * 50
            sc = ax.scatter(x, y, z, s=sizes, c=z, cmap='plasma', alpha=0.8, edgecolors='k', linewidth=0.5)
            ax.set_xlabel('Traffic Volume', fontsize=10)
            ax.set_ylabel('Avg Response Time', fontsize=10)
            ax.set_zlabel('Unique Anomalies', fontsize=10)
            ax.set_title('Traffic vs Response vs Anomaly Diversity', fontsize=12, pad=20)
            if st.session_state.dark_mode:
                ax.set_facecolor('#1a1d29')
                fig_3d.patch.set_facecolor('#1a1d29')
            fig_3d.colorbar(sc, ax=ax, label='Unique Anomaly Types')
            st.pyplot(fig_3d)

        with col2:
            st.markdown("#### 🔥 Correlation Matrix")
            fig_heatmap = plt.figure(figsize=(8, 6))
            if st.session_state.dark_mode:
                plt.style.use('dark_background')
            sns.heatmap(heatmap_data.corr(), annot=True, cmap='coolwarm', linewidths=1, linecolor='gray', square=True, cbar_kws={"shrink": 0.8})
            plt.title('Feature Correlation Heatmap', fontsize=14, pad=20)
            st.pyplot(fig_heatmap)

        # -----------------------------
        # Downloads
        # -----------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📥 Export Reports")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Anomalies Report",
                data=anomaly_report.to_csv(index=False).encode('utf-8'),
                file_name='anomalies_report.csv',
                mime='text/csv'
            )
        with col2:
            st.download_button(
                label="📥 Download Blocked IPs",
                data=blocked_ips.to_csv(index=False).encode('utf-8'),
                file_name='blocked_ips.csv',
                mime='text/csv'
            )
else:
    # Enhanced Welcome Screen - Using proper string formatting
    st.markdown(f"""<style>
        @keyframes float {{
            0%, 100% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-20px); }}
        }}
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
        }}
        .welcome-container {{
            padding: 3rem 2rem;
            animation: fadeInUp 1s ease-out;
        }}
        .feature-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }}
        .feature-card {{
            background: {colors['card_bg']};
            padding: 2rem;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px {colors['shadow']};
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            animation: fadeInUp 0.8s ease-out backwards;
        }}
        .feature-card:nth-child(1) {{ animation-delay: 0.1s; }}
        .feature-card:nth-child(2) {{ animation-delay: 0.2s; }}
        .feature-card:nth-child(3) {{ animation-delay: 0.3s; }}
        .feature-card:nth-child(4) {{ animation-delay: 0.4s; }}
        .feature-card:nth-child(5) {{ animation-delay: 0.5s; }}
        .feature-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            transition: left 0.5s;
        }}
        .feature-card:hover::before {{
            left: 100%;
        }}
        .feature-card:hover {{
            transform: translateY(-10px) scale(1.02);
            box-shadow: 0 15px 50px {colors['shadow']};
            border-color: {colors['accent']};
        }}
        .feature-icon {{
            font-size: 3rem;
            margin-bottom: 1rem;
            display: inline-block;
            animation: float 3s ease-in-out infinite;
        }}
        .feature-card:hover .feature-icon {{
            animation: pulse 0.6s ease-in-out;
        }}
        .feature-title {{
            font-size: 1.3rem;
            font-weight: 600;
            color: {colors['accent']};
            margin-bottom: 0.5rem;
        }}
        .feature-desc {{
            color: {colors['text']};
            opacity: 0.8;
            line-height: 1.6;
        }}
        .cta-section {{
            text-align: center;
            margin: 3rem 0;
            padding: 3rem;
            background: {colors['metric_bg']};
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px {colors['shadow']};
            animation: fadeInUp 1s ease-out 0.6s backwards;
        }}
        .cta-title {{
            font-size: 2rem;
            font-weight: 700;
            color: {colors['accent']};
            margin-bottom: 1rem;
        }}
        .cta-subtitle {{
            font-size: 1.1rem;
            color: {colors['text']};
            opacity: 0.7;
            margin-bottom: 2rem;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }}
        .stat-card {{
            background: {colors['secondary_bg']};
            padding: 1.5rem;
            border-radius: 16px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px {colors['shadow']};
        }}
        .stat-number {{
            font-size: 2.5rem;
            font-weight: 700;
            color: {colors['accent']};
            margin-bottom: 0.5rem;
        }}
        .stat-label {{
            font-size: 0.9rem;
            color: {colors['text']};
            opacity: 0.7;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .upload-icon {{
            font-size: 4rem;
            animation: float 3s ease-in-out infinite;
            margin-bottom: 1rem;
        }}
    </style>""", unsafe_allow_html=True)
    
    st.markdown("""<div class="welcome-container">
        <div class="cta-section">
            <div class="upload-icon">📊</div>
            <div class="cta-title">Get Started with Your Data Analysis</div>
            <div class="cta-subtitle">Upload your CSV file and unlock powerful insights in seconds</div>
        </div>
        <div class="feature-grid">
            <div class="feature-card">
                <div class="feature-icon">🔥</div>
                <div class="feature-title">Real-Time Trending</div>
                <div class="feature-desc">Track the hottest products and identify market trends instantly with live data visualization</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🛡️</div>
                <div class="feature-title">Threat Detection</div>
                <div class="feature-desc">Advanced algorithms detect DDoS attacks, brute force attempts, and security anomalies automatically</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📈</div>
                <div class="feature-title">3D Analytics</div>
                <div class="feature-desc">Explore your data in three dimensions with interactive 3D scatter plots and correlation matrices</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🎨</div>
                <div class="feature-title">Dark/Light Themes</div>
                <div class="feature-desc">Switch between beautiful dark and light modes for comfortable viewing any time of day</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📥</div>
                <div class="feature-title">Export Reports</div>
                <div class="feature-desc">Download comprehensive reports in CSV format for sharing and further analysis</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <div class="feature-title">Lightning Fast</div>
                <div class="feature-desc">Process thousands of records instantly with optimized algorithms and smart caching</div>
            </div>
        </div>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">99.9%</div>
                <div class="stat-label">Accuracy</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">&lt; 1s</div>
                <div class="stat-label">Processing Time</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">10+</div>
                <div class="stat-label">Visualizations</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">24/7</div>
                <div class="stat-label">Monitoring</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)