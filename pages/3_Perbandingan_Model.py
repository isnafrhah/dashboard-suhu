import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # <-- Tambahkan baris ini!
import streamlit as st

from style import apply_custom_css, draw_metric_card, render_header


# ======================================================
# CONFIG HALAMAN
# ======================================================

st.set_page_config(
    page_title="Perbandingan Model - BMKG",
    page_icon="📊",
    layout="wide"
)

apply_custom_css()


# ======================================================
# HEADER
# ======================================================

render_header(
    title="Performa Model Machine Learning",
    subtitle=(
        "Evaluasi dan komparasi hasil prediksi temperatur harian "
        "berdasarkan data aktual BMKG YIA."
    ),
)


# ======================================================
# PATH PROJECT
# ======================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

HASIL_MODEL_PATH = PROJECT_DIR / "hasil_model_forecast.csv"


# ======================================================
# LOAD HASIL MODEL
# ======================================================

@st.cache_data
def load_model_results():

    if not HASIL_MODEL_PATH.exists():
        return None

    try:

        df = pd.read_csv(HASIL_MODEL_PATH)

        # Bersihkan nama kolom
        df.columns = df.columns.str.strip()

        # Penyesuaian nama kolom jika diperlukan
        rename_map = {
            "Model": "Model Algorithm",
            "MAE": "MAE (°C)",
            "RMSE": "RMSE (°C)",
            "R²": "R² Score",
            "R2": "R² Score",
        }

        df.rename(
            columns=rename_map,
            inplace=True
        )

        # Kolom yang wajib tersedia
        required_columns = [
            "Model Algorithm",
            "MAE (°C)",
            "MSE",
            "RMSE (°C)",
            "R² Score",
        ]

        # Cek kolom
        missing_columns = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:

            st.error(
                "Kolom pada `hasil_model_forecast.csv` tidak sesuai. "
                f"Kolom yang hilang: {missing_columns}"
            )

            return None

        # Konversi kolom numerik
        numeric_columns = [
            "MAE (°C)",
            "MSE",
            "RMSE (°C)",
            "R² Score",
        ]

        for col in numeric_columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        # Hapus baris kosong
        df = (
            df
            .dropna(subset=required_columns)
            .reset_index(drop=True)
        )

        return df

    except Exception as e:

        st.error(
            f"Terjadi kesalahan saat membaca "
            f"`hasil_model_forecast.csv`: {e}"
        )

        return None


# ======================================================
# LOAD DATA
# ======================================================

df_metrics = load_model_results()


# ======================================================
# VALIDASI HASIL MODEL
# ======================================================

if df_metrics is None or df_metrics.empty:

    st.error(
        "File `hasil_model_forecast.csv` tidak ditemukan atau kosong. "
        "Pastikan file berada di folder utama project."
    )

    st.stop()


# ======================================================
# 1. RINGKASAN PERFORMA MODEL
# ======================================================

st.markdown(
    '<div class="section-header">'
    '1. Ringkasan Performa Model (Metrik Evaluasi)'
    '</div>',
    unsafe_allow_html=True
)


# ------------------------------------------------------
# MODEL TERBAIK
# ------------------------------------------------------

best_model = df_metrics.loc[
    df_metrics["R² Score"].idxmax()
]

best_rmse = df_metrics.loc[
    df_metrics["RMSE (°C)"].idxmin()
]

best_mae = df_metrics.loc[
    df_metrics["MAE (°C)"].idxmin()
]


# ======================================================
# METRIC CARDS
# ======================================================

c1, c2, c3, c4 = st.columns(4)


with c1:

    draw_metric_card(
        "Model Terbaik",
        best_model["Model Algorithm"],
        "Performa Tertinggi"
    )


with c2:

    draw_metric_card(
        "R² Score Terbaik",
        f"{best_model['R² Score']:.4f}",
        "Semakin mendekati 1 semakin baik"
    )


with c3:

    draw_metric_card(
        "RMSE Terendah",
        f"{best_rmse['RMSE (°C)']:.4f} °C",
        f"{best_rmse['Model Algorithm']}"
    )


with c4:

    draw_metric_card(
        "MAE Terendah",
        f"{best_mae['MAE (°C)']:.4f} °C",
        f"{best_mae['Model Algorithm']}"
    )


st.markdown(
    "<br>",
    unsafe_allow_html=True
)


# ======================================================
# 2. TABEL METRIK
# ======================================================

st.subheader(
    "Tabel Metrik Evaluasi"
)


display_metrics = df_metrics.copy()


display_metrics["MAE (°C)"] = (
    display_metrics["MAE (°C)"]
    .round(6)
)

display_metrics["MSE"] = (
    display_metrics["MSE"]
    .round(6)
)

display_metrics["RMSE (°C)"] = (
    display_metrics["RMSE (°C)"]
    .round(6)
)

display_metrics["R² Score"] = (
    display_metrics["R² Score"]
    .round(6)
)


st.dataframe(
    display_metrics,
    use_container_width=True,
    hide_index=True
)


st.markdown(
    "<br>",
    unsafe_allow_html=True
)


# ======================================================
# 3. GRAFIK PERBANDINGAN METRIK
# ======================================================

st.markdown(
    '<div class="section-header">'
    '2. Grafik Perbandingan Metrik Evaluasi'
    '</div>',
    unsafe_allow_html=True
)


col_m1, col_m2 = st.columns(2)


# ------------------------------------------------------
# GRAFIK R²
# ------------------------------------------------------

with col_m1:

    fig_r2 = px.bar(
        df_metrics,
        x="Model Algorithm",
        y="R² Score",
        color="Model Algorithm",
        text="R² Score",
        title=(
            "Perbandingan R² Score "
            "(Mendekati 1.0 Lebih Baik)"
        ),
        template="plotly_dark"
    )

    fig_r2.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside"
    )

    fig_r2.update_layout(
        showlegend=False,
        yaxis_title="R² Score",
        xaxis_title="Model Algorithm"
    )

    st.plotly_chart(
        fig_r2,
        use_container_width=True
    )


# ------------------------------------------------------
# GRAFIK RMSE
# ------------------------------------------------------

with col_m2:

    fig_rmse = px.bar(
        df_metrics,
        x="Model Algorithm",
        y="RMSE (°C)",
        color="Model Algorithm",
        text="RMSE (°C)",
        title=(
            "Perbandingan RMSE "
            "(Mendekati 0.0 Lebih Baik)"
        ),
        template="plotly_dark"
    )

    fig_rmse.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside"
    )

    fig_rmse.update_layout(
        showlegend=False,
        yaxis_title="RMSE (°C)",
        xaxis_title="Model Algorithm"
    )

    st.plotly_chart(
        fig_rmse,
        use_container_width=True
    )


st.markdown(
    "<br>",
    unsafe_allow_html=True
)


# ======================================================
# 4. KESIMPULAN
# ======================================================

st.markdown(
    '<div class="section-header">'
    '3. Kesimpulan Perbandingan Model'
    '</div>',
    unsafe_allow_html=True
)


st.info(
    f"Model **{best_model['Model Algorithm']}** memiliki performa "
    f"terbaik berdasarkan nilai R² tertinggi sebesar "
    f"**{best_model['R² Score']:.4f}**, dengan MAE sebesar "
    f"**{best_model['MAE (°C)']:.4f} °C** dan RMSE sebesar "
    f"**{best_model['RMSE (°C)']:.4f} °C**. "
    "Berdasarkan hasil evaluasi pada dataset pengujian, "
    "model tersebut digunakan sebagai model utama untuk "
    "prediksi temperatur pada dashboard."
)


# ======================================================
# 5. GRAFIK AKTUAL VS PREDIKSI
# ======================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    '<div class="section-header">4. Grafik Aktual vs Prediksi</div>',
    unsafe_allow_html=True
)

HASIL_PREDIKSI_PATH = PROJECT_DIR / "hasil_prediksi_forecast.csv"
@st.cache_data
def load_hasil_prediksi():
    df = pd.read_csv(HASIL_PREDIKSI_PATH)
    df["TANGGAL"] = pd.to_datetime(df["TANGGAL"], errors="coerce")
    return df.dropna(subset=["TANGGAL"]).sort_values(by="TANGGAL")

df_pred = load_hasil_prediksi()

model_cols = ["Linear Regression", "Decision Tree", "Random Forest"]
pilihan_model = st.selectbox("Pilih Model:", model_cols, index=2)

fig_compare = go.Figure()

# Ganti color ke #FFFFFF (Putih) atau #F1F5F9 (Abu-abu terang) agar kontras di latar gelap
fig_compare.add_trace(go.Scatter(
    x=df_pred["TANGGAL"], 
    y=df_pred["Aktual (TAVG)"],
    mode="lines", 
    name="Aktual", 
    line=dict(color="#FFFFFF", width=2) # <-- Warna diganti jadi putih terang
))

fig_compare.add_trace(go.Scatter(
    x=df_pred["TANGGAL"], 
    y=df_pred[pilihan_model],
    mode="lines", 
    name=f"Prediksi ({pilihan_model})",
    line=dict(color="#3B82F6", dash="dash", width=2) # <-- Biru cerah putus-putus
))

fig_compare.update_layout(
    title=f"Aktual vs Prediksi - {pilihan_model}",
    xaxis_title="Tanggal", 
    yaxis_title="Temperatur (°C)",
    template="plotly_dark" # <-- Sesuaikan template ke dark agar serasi dengan tema aplikasi
)

st.plotly_chart(fig_compare, use_container_width=True)