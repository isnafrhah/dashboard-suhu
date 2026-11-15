import streamlit as st
from style import apply_custom_css, draw_metric_card, render_header

# Config Halaman Utama
st.set_page_config(
    page_title="CuacaTani Cabai – Pemantauan Cuaca & Prediksi Suhu", page_icon="🌶️", layout="wide"
)
apply_custom_css()

# Header Utama
render_header(
    title="CuacaTani Cabai – Pemantauan Cuaca Kawasan Temon",
    subtitle="Sistem Analisis Parameter Iklim Harian dan Estimasi Temperatur Udara Berbasis Machine Learning di Kulon Progo.",
)

# Landing Page Banner - Tampilan Clean White & Modern Accent
st.markdown(
    """
<div style="
    background-color: #FFFFFF;
    border: 1px solid #D6E3D9;
    border-left: 6px solid #174C3A;
    border-radius: 12px;
    padding: 1.8rem 2rem;
    box-shadow: 0 4px 6px -1px rgba(23, 76, 58, 0.05);
    margin-bottom: 2rem;
">
    <h2 style="color: #174C3A; margin-top: 0; font-size: 1.4rem; font-weight: 700; display: flex; align-items: center; gap: 8px;">
        Selamat Datang di Sistem Informasi CuacaTani Cabai
    </h2>
    <p style="color: #54665C; font-size: 0.95rem; margin-bottom: 1rem; line-height: 1.5;">
        Gunakan menu navigasi di sebelah kiri (Sidebar) untuk menjelajahi berbagai fitur yang tersedia:
    </p>
    <ul style="color: #29352F; font-size: 0.92rem; line-height: 1.8; margin-bottom: 0; padding-left: 1.2rem;">
        <li><b>Dataset</b>: Menampilkan dan mendownload tabel data parameter iklim historis utuh.</li>
        <li><b>Visualisasi Data</b>: Grafik tren temporal harian, heatmap korelasi, dan distribusi.</li>
        <li><b>Perbandingan Model</b>: Evaluasi metrik MAE, MSE, RMSE, dan R² antar algoritma.</li>
        <li><b>Prediksi</b>: Simulasi kalkulasi estimasi temperatur udara harian secara real-time.</li>
        <li><b>Tentang</b>: Dokumentasi lengkap variabel dan algoritma yang digunakan.</li>
    </ul>
</div>
""",
    unsafe_allow_html=True,
)

# Metrics Highlight
c1, c2, c3, c4 = st.columns(4)
with c1:
    draw_metric_card("Target Prediksi", "TAVG (°C)", "Suhu Udara Rata-Rata")
with c2:
    draw_metric_card("Fitur Input", "8 Indikator (H-1)", "RH, RR, Tekanan, SS, FF, Suhu (H-1), Bulan, Hari (H)")
with c3:
    draw_metric_card("Model Aktif", "Random Forest", "Model Terbaik dari 3 Kandidat")
with c4:
    draw_metric_card("Status Sistem", "Aktif / Ready", "Streamlit Dashboard")