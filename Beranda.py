import streamlit as st
from style import apply_custom_css, draw_metric_card, render_header

# Config Halaman Utama
st.set_page_config(
    page_title="Dashboard Prediksi Suhu BMKG", page_icon="🌤️", layout="wide"
)
apply_custom_css()

# Header Utama
render_header(
    title="Dashboard Analisis & Prediksi Suhu Udara BMKG YIA",
    subtitle="Sistem Analisis Parameter Iklim Harian dan Estimasi Temperatur Udara Berbasis Machine Learning di YIA.",
)

# Landing Page Banner - Tampilan Clean White & Modern Accent
st.markdown(
    """
<div style="
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 6px solid #2563EB;
    border-radius: 12px;
    padding: 1.8rem 2rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    margin-bottom: 2rem;
">
    <h2 style="color: #1E293B; margin-top: 0; font-size: 1.5rem; font-weight: 700; display: flex; align-items: center; gap: 8px;">
        Selamat Datang di Dashboard Suhu Udara BMKG YIA! 🌤️
    </h2>
    <p style="color: #475569; font-size: 1rem; margin-bottom: 1rem; line-height: 1.5;">
        Gunakan menu navigasi di sebelah kiri (Sidebar) untuk menjelajahi berbagai fitur yang tersedia:
    </p>
    <ul style="color: #334155; font-size: 0.95rem; line-height: 1.8; margin-bottom: 0; padding-left: 1.2rem;">
        <li><b> Dataset</b>: Menampilkan dan mendownload tabel data parameter iklim BMKG utuh.</li>
        <li><b> Visualisasi Data</b>: Grafik tren temporal harian, heatmap korelasi, dan distribusi.</li>
        <li><b> Perbandingan Model</b>: Evaluasi metrik MAE, MSE, RMSE, dan R² antar algoritma.</li>
        <li><b> Prediksi</b>: Simulasi kalkulasi estimasi temperatur udara harian secara real-time.</li>
        <li><b> Tentang</b>: Dokumentasi lengkap variabel dan algoritma yang digunakan.</li>
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
    draw_metric_card("Fitur Input", "4 Indikator", "RH, RR, SS, FF")
with c3:
    draw_metric_card("Model Aktif", "Random Forest", "Akurasi Optimum")
with c4:
    draw_metric_card("Status Sistem", "Aktif / Ready 🟢", "Streamlit Dashboard")