import glob
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from style import apply_custom_css, draw_metric_card, render_header

# Config Halaman
st.set_page_config(page_title="Visualisasi Data - BMKG", page_icon="📈", layout="wide")
apply_custom_css()

# Header Halaman
render_header(
    title="Visualisasi & Analisis Parameter Iklim BMKG YIA",
    subtitle="Eksplorasi grafik tren temporal, korelasi antar variabel, dan distribusi parameter iklim BMKG."
)

@st.cache_data
def load_data():
    all_dfs = []
    
    # Read Excel (.xlsx)
    excel_files = glob.glob("*.xlsx") + glob.glob("**/*.xlsx", recursive=True)
    for f in excel_files:
        try:
            temp_df = pd.read_excel(f)
            all_dfs.append(temp_df)
        except Exception:
            pass

    # Read CSV (.csv)
    csv_files = glob.glob("*.csv") + glob.glob("**/*.csv", recursive=True)
    for f in csv_files:
        if "filtered" not in f and "full" not in f and "hasil_prediksi" not in f:
            try:
                temp_df = pd.read_csv(f)
                all_dfs.append(temp_df)
            except Exception:
                pass

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True).drop_duplicates()
        date_col = next((c for c in combined_df.columns if "tanggal" in c.lower() or "date" in c.lower()), None)
        if date_col:
            combined_df.rename(columns={date_col: "TANGGAL"}, inplace=True)
            combined_df["TANGGAL"] = pd.to_datetime(combined_df["TANGGAL"], errors="coerce")
            combined_df = combined_df.dropna(subset=["TANGGAL"]).sort_values(by="TANGGAL")
            return combined_df

    # Fallback Data Dummy jika file belum terbaca
    dates = pd.date_range(start="2024-07-14", end="2026-07-22", freq="D")
    return pd.DataFrame({
        "TANGGAL": dates,
        "RH_AVG": [80.0 + (i % 5) for i in range(len(dates))],
        "RR": [0.0 + (i % 10) * 2 for i in range(len(dates))],
        "SS": [6.0 + (i % 4) * 0.5 for i in range(len(dates))],
        "FF_AVG": [2.5 + (i % 3) * 0.2 for i in range(len(dates))],
        "TAVG": [26.0 + (i % 5) * 0.4 for i in range(len(dates))]
    })

df = load_data()

# ------------------------------------------------------
# 1. FILTER RENTANG TANGGAL
# ------------------------------------------------------
st.markdown('<div class="section-header">1. Filter Rentang Waktu Visualisasi</div>', unsafe_allow_html=True)

min_date_val = df["TANGGAL"].min().date()
max_date_val = df["TANGGAL"].max().date()

col_f1, col_f2 = st.columns(2)
with col_f1:
    start_date = st.date_input("Mulai Tanggal:", min_date_val, min_value=min_date_val, max_value=max_date_val)
with col_f2:
    end_date = st.date_input("Sampai Tanggal:", max_date_val, min_value=min_date_val, max_value=max_date_val)

# Filter Dataset berdasarkan pilihan
filtered_df = df[(df["TANGGAL"].dt.date >= start_date) & (df["TANGGAL"].dt.date <= end_date)]

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------
# 2. TREN TEMPORAL PARAMETER IKLIM
# ------------------------------------------------------
st.markdown('<div class="section-header">2. Grafik Tren Parameter Iklim</div>', unsafe_allow_html=True)

# Pilihan variabel untuk diplot
numeric_cols = [c for c in filtered_df.columns if c != "TANGGAL" and pd.api.types.is_numeric_dtype(filtered_df[c])]

selected_vars = st.multiselect(
    "Pilih Parameter Yang Ingin Ditampilkan Pada Grafik Tren:",
    options=numeric_cols,
    default=[c for c in ["TAVG", "RH_AVG", "RR"] if c in numeric_cols]
)

if selected_vars:
    fig_line = px.line(
        filtered_df,
        x="TANGGAL",
        y=selected_vars,
        title="Tren Harian Parameter Iklim BMKG",
        labels={"value": "Nilai Parameter", "TANGGAL": "Tanggal", "variable": "Parameter"},
        template="plotly_white"
    )
    fig_line.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_line, use_container_width=True)
else:
    st.warning("⚠️ Silakan pilih setidaknya satu parameter untuk ditampilkan.")

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------
# 3. MATRIKS KORELASI & DISTRIBUSI
# ------------------------------------------------------
st.markdown('<div class="section-header">3. Analisis Korelasi & Distribusi Data</div>', unsafe_allow_html=True)

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Matriks Korelasi (Heatmap)")
    if len(numeric_cols) > 1:
        corr_matrix = filtered_df[numeric_cols].corr().round(2)
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            title="Korelasi Antar Parameter Iklim"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Parameter numerik tidak cukup untuk menghitung korelasi.")

with col_chart2:
    st.subheader("Distribusi Parameter")
    dist_var = st.selectbox("Pilih Parameter Untuk Histogram:", numeric_cols)
    if dist_var:
        fig_hist = px.histogram(
            filtered_df,
            x=dist_var,
            nbins=30,
            marginal="box",
            color_discrete_sequence=["#2563EB"],
            title=f"Distribusi Frekuensi {dist_var}"
        )
        st.plotly_chart(fig_hist, use_container_width=True)