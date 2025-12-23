import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter

# --- Configuration & Styling ---
st.set_page_config(
    page_title="Automotive MFDD Calculator (ECE R13)",
    page_icon="🚗",
    layout="wide"
)

# --- HEADER WITH LOGO ---
col_logo, col_title = st.columns([1, 15])
with col_logo:
    # Using the favicon as the ST logo source
    st.image("https://serinthomas.co.in/favicon.ico", width=50)
with col_title:
    st.title("Passenger Vehicle MFDD Analyzer")

st.markdown("""
This tool, developed by Serin Thomas, calculates **Mean Fully Developed Deceleration (MFDD)** for passenger vehicles according to **ECE R13 / SAE J299** standards.
It analyzes the deceleration phase between **80% and 10%** of the initial braking speed. write to services@serinthomas.in for custom solutions.
""")
st.divider()

# --- Sidebar: Test Parameters ---
st.sidebar.header("1. Input Units")
time_unit = st.sidebar.selectbox("Time Unit in File", ["Seconds (s)", "Milliseconds (ms)"], index=0)
vel_unit = st.sidebar.selectbox("Velocity Unit in File", ["km/h", "m/s", "mph"], index=0)

st.sidebar.header("2. Analysis Settings")
start_threshold_pct = st.sidebar.slider("MFDD Start Threshold (Standard: 80%)", 50, 95, 80)
end_threshold_pct = st.sidebar.slider("MFDD End Threshold (Standard: 10%)", 5, 40, 10)
smooth_data = st.sidebar.checkbox("Apply Smoothing Filter", value=True)
window_length = st.sidebar.slider("Smoothing Window", 5, 51, 9, step=2, disabled=not smooth_data)

# --- Helper Functions ---
def load_data(file):
    if file.name.endswith('.csv'):
        # Try reading with default settings
        df = pd.read_csv(file)
        # If it read everything into 1 column, try semicolon separator
        if len(df.columns) < 2:
            file.seek(0)
            df = pd.read_csv(file, sep=';')
        # If still 1 column, try tab separator
        if len(df.columns) < 2:
            file.seek(0)
            df = pd.read_csv(file, sep='\t')
        return df
    else:
        return pd.read_excel(file)

def normalize_units(df, t_col, v_col, t_unit, v_unit):
    df_norm = df.copy()
    
    # Normalize Time to Seconds
    if t_unit == "Milliseconds (ms)":
        df_norm[t_col] = df_norm[t_col] / 1000.0
    
    # Create normalized Velocity columns
    if v_unit == "km/h":
        df_norm['v_ms'] = df_norm[v_col] / 3.6
        df_norm['v_kmh'] = df_norm[v_col]
    elif v_unit == "m/s":
        df_norm['v_ms'] = df_norm[v_col]
        df_norm['v_kmh'] = df_norm[v_col] * 3.6
    elif v_unit == "mph":
        df_norm['v_ms'] = df_norm[v_col] * 0.44704
        df_norm['v_kmh'] = df_norm[v_col] * 1.60934
        
    return df_norm

def find_nearest_idx(array, value):
    return (np.abs(array - value)).argmin()

# --- Main Application ---
uploaded_file = st.file_uploader("Upload Test Data (CSV/Excel)", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    # 1. Load Data
    try:
        raw_df = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # 2. Column Selection (FIXED FOR CRASH)
    cols = raw_df.columns.tolist()
    
    # Safety logic: Only default to index 1 if a second column actually exists
    default_time_idx = 0
    default_vel_idx = 1 if len(cols) > 1 else 0

    col1, col2 = st.columns(2)
    with col1:
        t_col = st.selectbox("Select Time Column", cols, index=default_time_idx)
    with col2:
        v_col = st.selectbox("Select Velocity Column", cols, index=default_vel_idx)

    # Validation: Ensure different columns are selected if possible
    if t_col == v_col and len(cols) > 1:
        st.warning("⚠️ You have selected the same column for Time and Velocity.")

    # 3. Process Data
    df = normalize_units(raw_df, t_col, v_col, time_unit, vel_unit)

    # Smoothing
    if smooth_data:
        try:
            df['v_smooth_kmh'] = savgol_filter(df['v_kmh'], window_length, 3)
            df['v_smooth_ms'] = df['v_smooth_kmh'] / 3.6
        except:
            st.warning("Data too short for smoothing. Using raw data.")
            df['v_smooth_kmh'] = df['v_kmh']
            df['v_smooth_ms'] = df['v_ms']
    else:
        df['v_smooth_kmh'] = df['v_kmh']
        df['v_smooth_ms'] = df['v_ms']

    # --- NEW FEATURE: Full Time Domain Plot ---
    st.subheader("🏁 Full Time Domain Velocity Data")
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(
        x=df[t_col],
        y=df['v_smooth_kmh'],
        mode='lines',
        name='Velocity (km/h)',
        line=dict(color='gray', width=2)
    ))
    fig_raw.update_layout(
        xaxis_title=f"Time (s)",
        yaxis_title="Velocity (km/h)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )
    st.plotly_chart(fig_raw, use_container_width=True)
    st.divider()
    # ------------------------------------------

    # 4. Detect Braking Event
    idx_peak = df['v_smooth_kmh'].idxmax()
    v_0 = df.loc[idx_peak, 'v_smooth_kmh'] # v_initial (km/h)
    
    brake_df = df.loc[idx_peak:].copy().reset_index(drop=True)
    
    brake_df['dt'] = brake_df[t_col].diff().fillna(0)
    brake_df['distance'] = (brake_df['v_smooth_ms'] * brake_df['dt']).cumsum()

    # 5. Determine MFDD Points (v_b and v_e)
    target_b_pct = start_threshold_pct / 100.0
    target_e_pct = end_threshold_pct / 100.0
    
    v_b_target = target_b_pct * v_0
    v_e_target = target_e_pct * v_0
    
    idx_b = find_nearest_idx(brake_df['v_smooth_kmh'].values, v_b_target)
    idx_e = find_nearest_idx(brake_df['v_smooth_kmh'].values, v_e_target)
    
    v_b = brake_df.loc[idx_b, 'v_smooth_kmh']
    v_e = brake_df.loc[idx_e, 'v_smooth_kmh']
    s_b = brake_df.loc[idx_b, 'distance']
    s_e = brake_df.loc[idx_e, 'distance']
    t_b = brake_df.loc[idx_b, t_col]
    t_e = brake_df.loc[idx_e, t_col]

    # 6. Calculate MFDD
    dist_interval = s_e - s_b
    
    if dist_interval > 0:
        mfdd = (v_b**2 - v_e**2) / (25.92 * dist_interval)
    else:
        mfdd = 0
        st.error("Distance interval is zero. Check data resolution.")

    # Calculate Total Stopping Distance
    try:
        idx_stop = brake_df[brake_df['v_smooth_kmh'] < 1.0].index[0]
        total_stop_dist = brake_df.loc[idx_stop, 'distance']
        total_stop_time = brake_df.loc[idx_stop, t_col] - brake_df.loc[0, t_col]
    except:
        total_stop_dist = brake_df['distance'].iloc[-1]
        total_stop_time = brake_df[t_col].iloc[-1] - brake_df[t_col].iloc[0]

    # --- Results Dashboard ---
    st.subheader("📊 Braking Analysis Results")
    
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
    col_metric1.metric("MFDD (m/s²)", f"{mfdd:.3f}")
    col_metric2.metric("Initial Speed (v₀)", f"{v_0:.1f} km/h")
    col_metric3.metric("Stopping Dist", f"{total_stop_dist:.2f} m")
    col_metric4.metric("Stopping Time", f"{total_stop_time:.2f} s")
    
    st.markdown(f"**Parameters:** $v_b$ @ {int(start_threshold_pct)}% | $v_e$ @ {int(end_threshold_pct)}%")

    # --- Plots ---
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=brake_df[t_col], y=brake_df['v_smooth_kmh'], 
        mode='lines', name='Velocity (km/h)', line=dict(color='#1f77b4', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=[t_b, t_e], y=[v_b, v_e], 
        mode='markers', name='MFDD Points', marker=dict(size=12, color='red', symbol='x')
    ))

    fig.add_shape(type="rect", x0=t_b, y0=0, x1=t_e, y1=v_0,
        fillcolor="red", opacity=0.1, layer="below", line_width=0
    )

    fig.update_layout(
        title=f"Braking Phase Zoom (v_max: {v_0:.1f} km/h)",
        xaxis_title="Time (s)",
        yaxis_title="Velocity (km/h)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👋 Upload a CSV/Excel file to start.")