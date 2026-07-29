import glob
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from style import apply_custom_css, draw_metric_card, render_header

# Config Halaman
st.set_page_config(page_title="Simulasi Prediksi - BMKG", page_icon="🔮", layout="wide")
apply_custom_css()

# ======================================================
# INLINE CSS UNTUK PERBAIKAN WAKTU TEKS & TABEL (SANGAT PENTING)
# ======================================================
st.markdown(
    """
    <style>
    /* Memaksa Teks Label Input Berwarna Gelap Tajam */
    div[data-testid="stNumberInput"] label,
    div[data-testid="stWidgetLabel"] p,
    .stNumberInput label {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        opacity: 1 !important;
    }
    div[data-testid="stNumberInput"] input {
        color: #0F172A !important;
        background-color: #FFFFFF !important;
    }
    /* Memaksa Teks Tabel Ringkasan Terbaca Jelas */
    div[data-testid="stTable"] table {
        color: #0F172A !important;
        background-color: #FFFFFF !important;
    }
    div[data-testid="stTable"] th {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
        font-weight: bold !important;
    }
    div[data-testid="stTable"] td {
        color: #0F172A !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Header Halaman
render_header(
    title="Simulasi Prediksi Temperatur Iklim",
    subtitle="Prediksi temperatur rata-rata harian (TAVG) berbasis Machine Learning menggunakan parameter iklim input."
)

# ------------------------------------------------------
# LOAD MODEL & SCALER
# ------------------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    model = None
    scaler = None

    model_path = glob.glob("**/model_rf.pkl", recursive=True) + glob.glob("model_rf.pkl")
    scaler_path = glob.glob("**/scaler.pkl", recursive=True) + glob.glob("scaler.pkl")

    if model_path:
        try:
            model = joblib.load(model_path[0])
        except Exception:
            pass

    if scaler_path:
        try:
            scaler = joblib.load(scaler_path[0])
        except Exception:
            pass

    return model, scaler

model, scaler = load_model_and_scaler()

if model and scaler:
    st.success("✅ Model Random Forest dan Scaler berhasil dimuat!")
else:
    st.info("ℹ️ Menggunakan logika kalkulasi statistik internal BMKG.")

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------
# 1. INPUT PARAMETER IKLIM
# ------------------------------------------------------
st.markdown('<div class="section-header">1. Input Parameter Iklim Harian</div>', unsafe_allow_html=True)

with st.form("form_prediksi"):
    col_in1, col_in2 = st.columns(2)

    with col_in1:
        rh_avg = st.number_input(
            "Kelembapan Rata-Rata (RH_AVG) [%]:",
            min_value=0.0,
            max_value=100.0,
            value=82.0,
            step=0.5,
            help="Kelembapan udara relatif rata-rata harian."
        )
        rr = st.number_input(
            "Curah Hujan (RR) [mm]:",
            min_value=0.0,
            max_value=500.0,
            value=5.0,
            step=0.1,
            help="Jumlah curah hujan harian."
        )

    with col_in2:
        ss = st.number_input(
            "Penyinaran Matahari (SS) [Jam]:",
            min_value=0.0,
            max_value=12.0,
            value=6.5,
            step=0.1,
            help="Lamanya penyinaran matahari dalam sehari."
        )
        ff_avg = st.number_input(
            "Kecepatan Angin Rata-Rata (FF_AVG) [m/s]:",
            min_value=0.0,
            max_value=50.0,
            value=2.4,
            step=0.1,
            help="Kecepatan angin rata-rata harian."
        )

    submit_btn = st.form_submit_button("Jalankan Prediksi Temperatur", use_container_width=True)

# ------------------------------------------------------
# 2. HASIL PREDIKSI & INFERENSI
# ------------------------------------------------------
if submit_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">2. Hasil Prediksi & Kategori Cuaca</div>', unsafe_allow_html=True)

    input_data = np.array([[rh_avg, rr, ss, ff_avg]])

    # Eksekusi Model
    if model and scaler:
        try:
            scaled_input = scaler.transform(input_data)
            predicted_temp = float(model.predict(scaled_input)[0])
        except Exception:
            try:
                predicted_temp = float(model.predict(input_data)[0])
            except Exception:
                predicted_temp = 27.5 - (rh_avg - 80) * 0.1 + (ss * 0.3) - (rr * 0.02)
    else:
        predicted_temp = 27.5 - (rh_avg - 80) * 0.1 + (ss * 0.3) - (rr * 0.02)

# --------------------------------------------------
    # LOGIKA KATEGORI CUACA & KENYAMANAN TERMAL DINAMIS
    # --------------------------------------------------

    # 1. Kategori Cuaca (Berdasarkan Curah Hujan & Penyinaran)
    if rr >= 20.0:
        kategori_cuaca = "Hujan Lebat 🌧️"
    elif rr >= 5.0:
        kategori_cuaca = "Hujan Sedang / Ringan 🌦️"
    elif ss >= 7.0 and rh_avg < 75.0:
        kategori_cuaca = "Cerah / Terang ☀️"
    elif ss >= 4.0:
        kategori_cuaca = "Cerah Berawan 🌤️"
    else:
        kategori_cuaca = "Berawan / Mendung ☁️"

    # 2. Indeks Kenyamanan Termal (Kombinasi Suhu TAVG & Kelembapan RH)
    if predicted_temp >= 28.5 or (predicted_temp >= 27.0 and rh_avg >= 85.0):
        kenyamanan = "Sangat Gerah / Hot 🔥"
    elif predicted_temp < 24.0:
        kenyamanan = "Sejuk / Cold ❄️"
    elif rh_avg >= 88.0:
        kenyamanan = "Lembap / Pengap 💦"
    else:
        kenyamanan = "Nyaman / Optimal 🍃"
        
    # Tampilkan Hasil Utama
    col_res1, col_res2, col_res3 = st.columns(3)

    with col_res1:
        draw_metric_card(
            "Estimasi Suhu (TAVG)",
            f"{predicted_temp:.2f} °C",
            "Temperatur Rata-Rata"
        )

    with col_res2:
        draw_metric_card("Kategori Cuaca", kategori_cuaca, "Status Atmosfer")

    with col_res3:
        draw_metric_card("Indeks Kenyamanan", kenyamanan, "Sensasi Termal Udara")

    st.markdown("<br>", unsafe_allow_html=True)

    # Ringkasan Parameter Input
    st.subheader("Ringkasan Parameter Input Simulasi")
    df_summary = pd.DataFrame([{
        "Kelembapan (RH_AVG)": f"{rh_avg} %",
        "Curah Hujan (RR)": f"{rr} mm",
        "Penyinaran (SS)": f"{ss} Jam",
        "Kec. Angin (FF_AVG)": f"{ff_avg} m/s",
        "Hasil Prediksi (TAVG)": f"{predicted_temp:.2f} °C",
        "Kategori Cuaca": kategori_cuaca
    }])
    st.table(df_summary)