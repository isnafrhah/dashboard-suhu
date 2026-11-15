import glob
import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from style import apply_custom_css, draw_metric_card, render_header

# Config Halaman
st.set_page_config(
    page_title="Visualisasi Data – CuacaTani Cabai",
    page_icon="📈",
    layout="wide"
)
apply_custom_css()

# Header Halaman
render_header(
    title="Visualisasi & Analisis Parameter Iklim Kawasan Temon",
    subtitle="Eksplorasi grafik tren temporal harian, matriks korelasi antar-variabel, dan distribusi parameter iklim di Kawasan Temon, Kulon Progo.",
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data_bmkg_fix.csv"


@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        df.columns = df.columns.str.strip()
        df["TANGGAL"] = pd.to_datetime(df["TANGGAL"], errors="coerce")
        df = df.dropna(subset=["TANGGAL"]).sort_values(by="TANGGAL").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Gagal membaca file data_bmkg_fix.csv: {e}")
        dates = pd.date_range(start="2024-07-14", end="2026-07-22", freq="D")
        return pd.DataFrame({
            "TANGGAL": dates,
            "RH_AVG": [80.0] * len(dates),
            "RR": [0.0] * len(dates),
            "SS": [6.0] * len(dates),
            "FF_AVG": [2.5] * len(dates),
            "TAVG": [26.0] * len(dates)
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

# Filter Dataset berdasarkan pilihan tanggal
filtered_df = df[(df["TANGGAL"].dt.date >= start_date) & (df["TANGGAL"].dt.date <= end_date)]

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------
# 2. TREN TEMPORAL PARAMETER IKLIM
# ------------------------------------------------------
st.markdown(
    '<div class="section-header">2. Grafik Tren Temporal Parameter Iklim</div>',
    unsafe_allow_html=True,
)

ignored_cols = [
    "TANGGAL",
    "YEAR",
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "DAYOFWEEK",
    "QUARTER",
    "WEEKOFYEAR",
]

numeric_cols = [
    c
    for c in filtered_df.columns
    if c not in ignored_cols and pd.api.types.is_numeric_dtype(filtered_df[c])
]

selected_vars = st.multiselect(
    "Pilih Parameter Cuaca untuk Ditampilkan pada Grafik Tren:",
    options=numeric_cols,
    default=[c for c in ["TAVG", "RH_AVG", "RR"] if c in numeric_cols]
)

if selected_vars:
    fig_line = px.line(
        filtered_df,
        x="TANGGAL",
        y=selected_vars,
        labels={"value": "Nilai Parameter", "TANGGAL": "Tanggal Pengamatan", "variable": "Parameter Cuaca"},
        color_discrete_sequence=["#174C3A", "#D6AD55", "#2D725A", "#8D6B28", "#5B8266", "#3D5A4C"],
        template="plotly_white",
    )
    fig_line.update_layout(
        title=dict(
            text="Tren Harian Parameter Iklim Kawasan Temon, Kulon Progo",
            font=dict(family="Manrope", size=14, color="#174C3A")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        margin=dict(l=10, r=10, t=45, b=55),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#29352F")
        ),
        xaxis=dict(showgrid=True, gridcolor="#EAF3E8"),
        yaxis=dict(showgrid=True, gridcolor="#EAF3E8"),
    )
    st.plotly_chart(fig_line, use_container_width=True)
else:
    st.warning("⚠️ Silakan pilih setidaknya satu parameter cuaca untuk ditampilkan pada grafik.")

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------
# 3. MATRIKS KORELASI & DISTRIBUSI
# ------------------------------------------------------
st.markdown(
    '<div class="section-header">3. Analisis Korelasi & Distribusi Data</div>',
    unsafe_allow_html=True
)

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown("<h4 style='color:#174C3A; font-family:Manrope,sans-serif; margin-top:0;'>Matriks Korelasi (Heatmap)</h4>", unsafe_allow_html=True)

    corr_cols = [
        "TAVG",
        "RH_AVG",
        "RR",
        "SS",
        "FF_AVG",
        "PRESSURE"
    ]

    available_corr_cols = [
        col for col in corr_cols if col in filtered_df.columns
    ]

    if len(available_corr_cols) > 1:
        corr_matrix = filtered_df[available_corr_cols].corr().round(2)

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale=["#FFFFFF", "#D5E4D8", "#A8C3AD", "#2D725A", "#174C3A"],
            title="Korelasi Pearson Antar-Parameter Iklim"
        )
        fig_corr.update_layout(
            title=dict(
                text="Korelasi Pearson Antar-Parameter Iklim",
                font=dict(family="Manrope", size=13, color="#174C3A")
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=360,
            margin=dict(l=10, r=10, t=40, b=10),
        )

        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Parameter numerik tidak cukup untuk menghitung korelasi.")

with col_chart2:
    st.markdown("<h4 style='color:#174C3A; font-family:Manrope,sans-serif; margin-top:0;'>Distribusi Frekuensi Parameter</h4>", unsafe_allow_html=True)

    dist_var = st.selectbox(
        "Pilih Parameter untuk Histogram:",
        numeric_cols
    )

    if dist_var:
        fig_hist = px.histogram(
            filtered_df,
            x=dist_var,
            nbins=30,
            marginal="box",
            color_discrete_sequence=["#174C3A"],
        )

        fig_hist.update_layout(
            title=dict(
                text=f"Distribusi Frekuensi Parameter {dist_var}",
                font=dict(family="Manrope", size=13, color="#174C3A")
            ),
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title=dist_var,
            yaxis_title="Frekuensi",
            height=360,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(showgrid=True, gridcolor="#EAF3E8"),
            yaxis=dict(showgrid=True, gridcolor="#EAF3E8"),
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True
        )
    else:
        st.info("Tidak ada parameter numerik yang tersedia.")
