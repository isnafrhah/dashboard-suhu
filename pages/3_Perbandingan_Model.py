import glob
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from style import apply_custom_css, draw_metric_card, render_header

# Config Halaman
st.set_page_config(page_title="Perbandingan Model - BMKG", page_icon="⚖️", layout="wide")
apply_custom_css()

# Header Halaman
render_header(
    title="Perbandingan Performa Model Machine Learning",
    subtitle="Evaluasi dan komparasi hasil prediksi temperatur harian berdasarkan berbagai algoritma regresi."
)

@st.cache_data
def load_prediction_data():
    # 1. Cari file hasil prediksi jika ada
    pred_files = glob.glob("**/hasil_prediksi.csv", recursive=True) + glob.glob("hasil_prediksi.csv")
    if pred_files:
        try:
            df_pred = pd.read_csv(pred_files[0])
            date_col = next((c for c in df_pred.columns if "tanggal" in c.lower() or "date" in c.lower()), None)
            if date_col:
                df_pred.rename(columns={date_col: "TANGGAL"}, inplace=True)
                df_pred["TANGGAL"] = pd.to_datetime(df_pred["TANGGAL"], errors="coerce")
            return df_pred
        except Exception:
            pass

    # 2. Dummy Data Evaluasi jika file belum ada
    dates = pd.date_range(start="2024-07-14", end="2026-07-22", freq="D")
    actual = np.sin(np.linspace(0, 20, len(dates))) * 2 + 27 + np.random.normal(0, 0.4, len(dates))
    
    # Simulasi Prediksi Beberapa Algoritma
    rf_pred = actual + np.random.normal(0, 0.25, len(dates))
    dt_pred = actual + np.random.normal(0, 0.45, len(dates))
    lr_pred = actual + np.random.normal(0, 0.55, len(dates))

    return pd.DataFrame({
        "TANGGAL": dates,
        "Aktual (TAVG)": actual,
        "Random Forest": rf_pred,
        "Decision Tree": dt_pred,
        "Linear Regression": lr_pred
    })

df_pred = load_prediction_data()

# ------------------------------------------------------
# 1. RINGKASAN METRIK EVALUASI MODEL
# ------------------------------------------------------
st.markdown('<div class="section-header">1. Ringkasan Performa Model (Metrik Evaluasi)</div>', unsafe_allow_html=True)

# Hitung Metrik Evaluasi
model_columns = [c for c in df_pred.columns if c not in ["TANGGAL", "Aktual (TAVG)", "Tanggal"]]

metrics_data = []
actual = df_pred["Aktual (TAVG)"] if "Aktual (TAVG)" in df_pred.columns else df_pred.iloc[:, 1]

for col in model_columns:
    pred = df_pred[col]
    mae = np.mean(np.abs(actual - pred))
    mse = np.mean((actual - pred) ** 2)
    rmse = np.sqrt(mse)
    
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    metrics_data.append({
        "Model Algorithm": col,
        "MAE (°C)": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE (°C)": round(rmse, 4),
        "R² Score": round(r2, 4)
    })

df_metrics = pd.DataFrame(metrics_data)

# Tampilkan Metric Card Model Terbaik (R2 Terbesar)
best_model = df_metrics.loc[df_metrics["R² Score"].idxmax()]

c1, c2, c3, c4 = st.columns(4)
with c1:
    draw_metric_card("Model Terbaik", f"{best_model['Model Algorithm']}", "Performa Tertinggi")
with c2:
    draw_metric_card("R² Score Terbaik", f"{best_model['R² Score']:.4f}", "Akurasi Prediksi")
with c3:
    draw_metric_card("RMSE Terendah", f"{best_model['RMSE (°C)']} °C", "Rata-rata Error Root")
with c4:
    draw_metric_card("MAE Terendah", f"{best_model['MAE (°C)']} °C", "Rata-rata Error Absolute")

st.markdown("<br>", unsafe_allow_html=True)

# Tabel Metrik Komparasi
st.subheader("Tabel Metrik Evaluasi")
st.dataframe(df_metrics, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------
# 2. GRAFIK KOMPARASI METRIK
# ------------------------------------------------------
st.markdown('<div class="section-header">2. Grafik Perbandingan Metrik Evaluasi</div>', unsafe_allow_html=True)

col_m1, col_m2 = st.columns(2)

with col_m1:
    fig_r2 = px.bar(
        df_metrics,
        x="Model Algorithm",
        y="R² Score",
        color="Model Algorithm",
        text="R² Score",
        title="Perbandingan R² Score (Mendekati 1.0 Lebih Baik)",
        template="plotly_white"
    )
    fig_r2.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig_r2.update_layout(showlegend=False, yaxis_range=[0, 1.1])
    st.plotly_chart(fig_r2, use_container_width=True)

with col_m2:
    fig_rmse = px.bar(
        df_metrics,
        x="Model Algorithm",
        y="RMSE (°C)",
        color="Model Algorithm",
        text="RMSE (°C)",
        title="Perbandingan RMSE (Mendekati 0.0 Lebih Baik)",
        template="plotly_white"
    )
    fig_rmse.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig_rmse.update_layout(showlegend=False)
    st.plotly_chart(fig_rmse, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------
# 3. GRAFIK AKTUAL VS PREDIKSI TEMPORAL
# ------------------------------------------------------
st.markdown('<div class="section-header">3. Plot Grafik Nilai Aktual vs Hasil Prediksi</div>', unsafe_allow_html=True)

selected_models = st.multiselect(
    "Pilih Model Untuk Di-plot Bersama Nilai Aktual:",
    options=model_columns,
    default=model_columns[:2] if len(model_columns) >= 2 else model_columns
)

if selected_models:
    plot_cols = ["TANGGAL", "Aktual (TAVG)"] + selected_models
    df_plot = df_pred[plot_cols].melt(id_vars=["TANGGAL"], var_name="Parameter/Model", value_name="Suhu (°C)")
    
    fig_compare = px.line(
        df_plot,
        x="TANGGAL",
        y="Suhu (°C)",
        color="Parameter/Model",
        title="Tren Temperatur Harian: Nilai Aktual vs Prediksi Model",
        template="plotly_white"
    )
    fig_compare.update_layout(hovermode="x unified")
    st.plotly_chart(fig_compare, use_container_width=True)
else:
    st.warning("⚠️ Pilih minimal satu model untuk melihat grafik perbandingan.")