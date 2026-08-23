import streamlit as st
from style import apply_custom_css, draw_metric_card, render_header

# Config Halaman
st.set_page_config(page_title="Tentang - BMKG", page_icon="ℹ️", layout="wide")
apply_custom_css()

# Header Halaman
render_header(
    title="Tentang Sistem Prediksi Temperatur BMKG",
    subtitle="Dokumentasi, metodologi Machine Learning, dan informasi variabel parameter iklim.",
)

# ------------------------------------------------------
# 1. TENTANG PROYEK
# ------------------------------------------------------
st.markdown(
    '<div class="section-header">1. Gambaran Umum Proyek</div>',
    unsafe_allow_html=True,
)

st.write(
    """
Dashboard ini dirancang untuk melakukan **monitoring, analisis visual, dan pemodelan prediksi temperatur harian ($T_{AVG}$)** berdasarkan data histori parameter iklim dari Badan Meteorologi, Klimatologi, dan Geofisika (BMKG). 

Dengan memanfaatkan algoritma **Machine Learning (Random Forest Regression)**, sistem ini memprediksi suhu udara **H+1 (satu hari ke depan)** berdasarkan kondisi kelembapan, curah hujan, tekanan udara, penyinaran matahari, kecepatan angin, dan suhu rata-rata pada hari sebelumnya (H-1). Dengan skema ini, model benar-benar melakukan *forecasting* menggunakan informasi yang tersedia sebelum hari yang diprediksi terjadi, bukan sekadar estimasi berdasarkan kondisi hari yang sama.
"""
)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------
# 2. DEFINISI PARAMETER IKLIM
# ------------------------------------------------------
st.markdown(
    '<div class="section-header">2. Variabel & Parameter Iklim</div>',
    unsafe_allow_html=True,
)

col_p1, col_p2 = st.columns(2)

with col_p1:
    st.markdown(
        """
    * **`TAVG` (Temperatur Rata-Rata)**: Suhu udara rata-rata harian yang diukur dalam satuan derajat Celsius (°C). Parameter ini menjadi variabel target (*label*) prediksi.
    * **`RH_AVG` (Kelembapan Rata-Rata)**: Persentase kelembapan udara relatif rata-rata harian (%).
    * **`RR` (Curah Hujan)**: Jumlah akumulasi curah hujan harian yang diukur dalam satuan milimeter (mm).
    * **`SS` (Penyinaran Matahari)**: Durasi lamanya sinar matahari memancar terang hingga permukaan bumi dalam sehari (Jam).
    """
    )

with col_p2:
    st.markdown(
        """
    * **`FF_AVG` (Kecepatan Angin)**: Kecepatan angin rata-rata harian yang diukur pada ketinggian standar (m/s).
    * **`MONTH` & `DAY`**: Komponen bulan dan tanggal pada hari yang diprediksi, digunakan model untuk menangkap pola musiman.
    * **`TAVG_LAG_1`**: Suhu rata-rata satu hari sebelumnya (H-1), fitur dengan korelasi tertinggi terhadap TAVG (≈0.78).
    * **`PRESSURE` (Tekanan Udara)**: Tekanan udara rata-rata permukaan (hPa/mb), diambil dari data H-1.

    Sejak pembaruan terbaru, model diarahkan untuk melakukan **forecast H+1**: seluruh parameter kelembapan, curah hujan, tekanan udara, penyinaran matahari, dan kecepatan angin di atas menggunakan nilai pada hari sebelumnya (H-1) untuk memprediksi TAVG pada hari berikutnya (H+1). Skema ini meningkatkan R² Score model Random Forest menjadi **0,7202** (dari skema estimasi hari yang sama sebesar 0,6751).
    """
    )

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------
# 3. METODOLOGI & PERFORMA MODEL
# ------------------------------------------------------
st.markdown(
    '<div class="section-header">3. Metodologi Pemodelan Machine Learning</div>',
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    draw_metric_card(
        "Algoritma Utama", "Random Forest", "Ensemble Regression"
    )
with c2:
    draw_metric_card(
        "Preprocessing", "Cleaning & Interpolasi Waktu", "Tanpa Scaling (Tree-Based)"
    )
with c3:
    draw_metric_card("Evaluasi Utama", "R² Score & RMSE", "Akurasi Prediksi")

st.markdown("<br>", unsafe_allow_html=True)

st.info(
    "💡 **Catatan**: Model Random Forest dipilih karena ketahanannya terhadap data non-linear serta kemampuannya menangani kompleksitas korelasi antar-parameter iklim secara efisien. "
    "Sebagai model berbasis pohon keputusan (*tree-based*), Random Forest membagi data berdasarkan nilai ambang batas fitur, sehingga tidak memerlukan normalisasi/scaling seperti pada model berbasis jarak (misalnya KNN atau regresi linear dengan regularisasi)."
)