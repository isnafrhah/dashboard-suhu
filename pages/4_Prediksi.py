import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from style import apply_custom_css, render_header

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Analisis Klimatologi",
    page_icon="⛅",
    layout="wide"
)
apply_custom_css()

# ============================================================
# CSS CUSTOM
# ============================================================
st.markdown("""
<style>
div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label span,
.stRadio label div,
.stRadio label {
    color: #0F172A !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stDateInput"] label {
    color: #0F172A !important;
    font-weight: 500 !important;
}
.metric-card {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    text-align: left;
}
.metric-label {
    color: #475569 !important;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    margin-bottom: 5px;
}
.metric-value {
    color: #0F172A !important;
    font-size: 1.35rem;
    font-weight: 700;
}
div[data-testid="stTable"] table {
    color: #0F172A !important;
    background-color: #FFFFFF !important;
}
div[data-testid="stTable"] th {
    background-color: #F1F5F9 !important;
    color: #0F172A !important;
}
div[data-testid="stTable"] td {
    color: #0F172A !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
render_header(
    "Analisis & Proyeksi Klimatologi",
    "Prediksi suhu H+1 (hari berikutnya) berdasarkan kondisi cuaca satu hari sebelumnya."
)

# ============================================================
# LOAD MODEL FORECAST H+1 (satu-satunya model di halaman ini)
# ============================================================
MODEL_FORECAST_PATH = "model_rf_forecast.pkl"
DATA_PATH = "data_bmkg_fix.csv"

FORECAST_FEATURES = [
    "RH_AVG_LAG_1", "RR_LAG_1", "PRESSURE_LAG_1", "SS_LAG_1", "FF_AVG_LAG_1",
    "MONTH", "DAY", "TAVG_LAG_1"
]

METRICS_PATH = "hasil_model_forecast.csv"


@st.cache_data
def load_mae_forecast(path):
    """Ambil MAE Random Forest dari hasil evaluasi, untuk rentang ketidakpastian prediksi."""
    if not os.path.exists(path):
        return 0.5  # fallback aman kalau file tidak ada
    df = pd.read_csv(path)
    row = df[df["Model Algorithm"] == "Random Forest"]
    if row.empty:
        return 0.5
    return float(row["MAE (°C)"].iloc[0])


@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Gagal memuat model ({path}): {e}")
        return None


@st.cache_data
def load_last_row(path):
    """Ambil baris data aktual paling baru dari dataset hasil cleaning (fallback)."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["TANGGAL"])
    df = df.sort_values("TANGGAL")
    return df.iloc[-1]


# Koordinat Stasiun/Bandara YIA (Temon, Kulon Progo)
YIA_LAT = -7.9053
YIA_LON = 110.0573

KALIBRASI_PATH = "kalibrasi_openmeteo.json"


@st.cache_data
def load_kalibrasi():
    """
    Muat bias koreksi Open-Meteo vs data BMKG asli (hasil dari
    hitung_kalibrasi_openmeteo.py). Kalau file belum ada, tidak ada koreksi
    yang diterapkan (bias dianggap 0).
    """
    import json
    if not os.path.exists(KALIBRASI_PATH):
        return None
    try:
        with open(KALIBRASI_PATH) as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=3600)  # cache 1 jam, supaya tidak memanggil API berkali-kali
def ambil_cuaca_live():
    """
    Ambil kondisi cuaca hari ini secara otomatis dari Open-Meteo (API cuaca publik,
    gratis, tanpa API key) untuk koordinat YIA. Ini yang membuat mode Otomatis
    beneran real-time -- tidak bergantung pada dataset statis.
    """
    import requests

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": YIA_LAT,
        "longitude": YIA_LON,
        "daily": "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum,"
                 "wind_speed_10m_mean,sunshine_duration",
        "hourly": "surface_pressure",
        "timezone": "Asia/Jakarta",
        "wind_speed_unit": "ms",   # supaya satuan angin m/s, sama seperti dataset BMKG
        "forecast_days": 1
    }

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    daily = data["daily"]
    tekanan_per_jam = data["hourly"]["surface_pressure"]
    tekanan_rata2 = sum(tekanan_per_jam) / len(tekanan_per_jam)

    hasil = {
        "TANGGAL": pd.to_datetime(daily["time"][0]),
        "TAVG": daily["temperature_2m_mean"][0],
        "RH_AVG": daily["relative_humidity_2m_mean"][0],
        "RR": daily["precipitation_sum"][0] or 0.0,
        "FF_AVG": daily["wind_speed_10m_mean"][0],
        "SS": (daily["sunshine_duration"][0] or 0) / 3600,  # detik -> jam
        "PRESSURE": tekanan_rata2
    }

    # Terapkan koreksi bias (kalau file kalibrasi sudah dibuat) supaya angka
    # Open-Meteo lebih mendekati kondisi instrumen BMKG asli di YIA.
    # Bias = rata-rata (Open-Meteo - BMKG asli), jadi dikurangkan.
    bias = load_kalibrasi()
    if bias:
        for key in ["TAVG", "RH_AVG", "RR", "FF_AVG", "SS", "PRESSURE"]:
            if key in bias:
                hasil[key] = hasil[key] - bias[key]
        hasil["RR"] = max(0.0, hasil["RR"])   # curah hujan tidak boleh negatif
        hasil["SS"] = max(0.0, hasil["SS"])   # penyinaran tidak boleh negatif
        hasil["_terkalibrasi"] = True
    else:
        hasil["_terkalibrasi"] = False

    return hasil


def prediksi_h_plus_1(model, rh, rr, pressure, ss, ff, tavg, tanggal_x):
    """
    Prediksi suhu hari (tanggal_x + 1) berdasarkan kondisi cuaca pada tanggal_x.
    Semua parameter (rh, rr, pressure, ss, ff, tavg) adalah kondisi AKTUAL/INPUT
    pada tanggal_x -- fungsi ini yang mengubahnya jadi fitur *_LAG_1 untuk
    memprediksi hari setelahnya.
    """
    tanggal_besok = tanggal_x + timedelta(days=1)

    input_df = pd.DataFrame({
        "RH_AVG_LAG_1": [rh],
        "RR_LAG_1": [rr],
        "PRESSURE_LAG_1": [pressure],
        "SS_LAG_1": [ss],
        "FF_AVG_LAG_1": [ff],
        "MONTH": [tanggal_besok.month],
        "DAY": [tanggal_besok.day],
        "TAVG_LAG_1": [tavg]
    })

    if hasattr(model, "feature_names_in_"):
        input_df = input_df[list(model.feature_names_in_)]
    else:
        input_df = input_df[FORECAST_FEATURES]

    suhu_besok = float(model.predict(input_df)[0])
    return tanggal_besok, suhu_besok


def kategori_kenyamanan(suhu):
    if suhu < 24:
        return "Sejuk"
    elif suhu < 28.5:
        return "Optimal / Nyaman"
    elif suhu < 32:
        return "Hangat"
    else:
        return "Sangat Gerah"


def render_hasil(tanggal_x, tanggal_besok, tavg_x, suhu_besok, sumber_label, mae):
    selisih = suhu_besok - tavg_x
    if selisih > 0.05:
        tren_teks = "Cenderung Naik"
    elif selisih < -0.05:
        tren_teks = "Cenderung Turun"
    else:
        tren_teks = "Stabil"

    kenyamanan = kategori_kenyamanan(suhu_besok)

    st.markdown(f"### Proyeksi Suhu: {tanggal_besok.strftime('%d %B %Y')}")
    st.caption(f"Dihitung dari kondisi cuaca {sumber_label} tanggal {tanggal_x.strftime('%d %B %Y')}.")

    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Prediksi Suhu ({tanggal_besok.strftime('%d %b')})</div>
            <div class="metric-value">{suhu_besok:.2f} °C</div>
            <div style="color:#64748B; font-size:0.8rem; margin-top:4px;">
            MAE model: ±{mae:.2f} °C
            </div>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Tren vs {tanggal_x.strftime('%d %b')}</div>
            <div class="metric-value">{tren_teks}</div>
        </div>
        """, unsafe_allow_html=True)
    with p3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Indeks Kenyamanan</div>
            <div class="metric-value">{kenyamanan}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[tanggal_x.strftime("%d %b"), tanggal_besok.strftime("%d %b")],
        y=[tavg_x, suhu_besok],
        mode="lines+markers+text",
        line=dict(color="#ef4444", width=3),
        marker=dict(size=10),
        text=[f"{tavg_x:.2f}°C ({sumber_label})", f"{suhu_besok:.2f}°C (prediksi)"],
        textposition="top center",
        name="Proyeksi Suhu"
    ))
    fig.update_layout(
        title="Grafik Proyeksi Suhu H+1",
        xaxis_title="",
        yaxis_title="Temperatur (°C)",
        template="plotly_dark",
        height=320,
        margin=dict(t=40, b=20, l=20, r=20)
    )
    st.plotly_chart(fig, use_container_width=True)


model_forecast = load_model(MODEL_FORECAST_PATH)
last_row = load_last_row(DATA_PATH)
mae_forecast = load_mae_forecast(METRICS_PATH)

if model_forecast is None:
    st.error("Model forecast (model_rf_forecast.pkl) tidak ditemukan.")
    st.info("Pastikan file 'model_rf_forecast.pkl' berada di folder utama project.")
    st.stop()

# ============================================================
# SKENARIO ANALISIS (RADIO BUTTON)
# ============================================================
mode = st.radio(
    "Mode Prediksi:",
    [
        "Prediksi Otomatis (Live)",
        "Prediksi Manual (Opsional)"
    ],
    horizontal=True
)
st.caption(
    "Mode **Otomatis** mengambil kondisi cuaca hari ini secara langsung dari data cuaca publik "
    "untuk lokasi YIA — tidak perlu isi apapun. Mode **Prediksi Manual** untuk yang ingin bereksperimen "
    "dengan angka sendiri (misalnya petugas BMKG yang ingin mengecek skenario tertentu)."
)
st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# MODE 1: OTOMATIS (LIVE) - AMBIL DATA CUACA HARI INI DARI API PUBLIK
# ============================================================
if mode == "Prediksi Otomatis (Live)":

    data_hari_ini = None
    sumber_data = "live"

    try:
        data_hari_ini = ambil_cuaca_live()
    except Exception:
        data_hari_ini = None

    if data_hari_ini is None:
        # Fallback: kalau API cuaca gagal diakses (mis. tidak ada koneksi internet),
        # pakai data terakhir dari dataset supaya halaman tetap berfungsi.
        if last_row is not None:
            data_hari_ini = {
                "TANGGAL": last_row["TANGGAL"], "TAVG": last_row["TAVG"],
                "RH_AVG": last_row["RH_AVG"], "RR": last_row["RR"],
                "FF_AVG": last_row["FF_AVG"], "SS": last_row["SS"],
                "PRESSURE": last_row["PRESSURE"]
            }
            sumber_data = "dataset"
        else:
            st.error("Tidak bisa mengambil data cuaca — cek koneksi internet, atau dataset cadangan tidak ditemukan.")
            st.stop()

    tanggal_x = data_hari_ini["TANGGAL"]

    if sumber_data == "live":
        st.info(
            f"📅 Kondisi cuaca hari ini (**{tanggal_x.strftime('%d %B %Y')}**) diambil otomatis "
            f"dari data cuaca terkini untuk lokasi YIA. Prediksi di bawah untuk hari setelahnya."
        )
        if data_hari_ini.get("_terkalibrasi"):
            st.caption(
                "ℹ️ Data cuaca sumber pihak ketiga (Open-Meteo), sudah dikoreksi memakai bias "
                "historis terhadap data instrumen BMKG YIA — namun tetap bukan pengukuran "
                "langsung, sehingga hasil prediksi berpotensi sedikit berbeda dari kondisi aktual."
            )
        else:
            st.caption(
                "ℹ️ Data cuaca berasal dari sumber pihak ketiga (Open-Meteo), bukan alat ukur "
                "langsung BMKG YIA, sehingga hasil prediksi berpotensi sedikit berbeda dari kondisi "
                "aktual dibanding jika memakai data instrumen BMKG langsung."
            )
    else:
        st.warning(
            f"⚠️ Data cuaca live sedang tidak dapat diakses. Prediksi ditampilkan menggunakan data "
            f"cadangan tanggal **{tanggal_x.strftime('%d %B %Y')}**."
        )

    tanggal_besok, suhu_besok = prediksi_h_plus_1(
        model_forecast,
        rh=data_hari_ini["RH_AVG"], rr=data_hari_ini["RR"], pressure=data_hari_ini["PRESSURE"],
        ss=data_hari_ini["SS"], ff=data_hari_ini["FF_AVG"], tavg=data_hari_ini["TAVG"],
        tanggal_x=tanggal_x
    )

    render_hasil(tanggal_x, tanggal_besok, data_hari_ini["TAVG"], suhu_besok, sumber_label="data hari ini", mae=mae_forecast)

    st.markdown("---")
    st.markdown("#### Data Cuaca yang Dipakai Sebagai Dasar Prediksi")
    df_summary = pd.DataFrame({
        "Parameter Klimatologi": ["Kelembapan (RH)", "Curah Hujan (RR)", "Tekanan Udara",
                                    "Penyinaran (SS)", "Kecepatan Angin (FF)", "Suhu Rata-Rata"],
        "Nilai Aktual": [
            f"{data_hari_ini['RH_AVG']:.2f} %", f"{data_hari_ini['RR']:.2f} mm",
            f"{data_hari_ini['PRESSURE']:.2f} hPa", f"{data_hari_ini['SS']:.2f} Jam",
            f"{data_hari_ini['FF_AVG']:.2f} m/s", f"{data_hari_ini['TAVG']:.2f} °C"
        ]
    })
    st.table(df_summary)

# ============================================================
# MODE 2: INPUT MANUAL - USER PILIH TANGGAL & KONDISI SENDIRI
# ============================================================
else:
    st.caption(
        "Pilih tanggal, lalu isi kondisi cuaca pada tanggal tersebut. Sistem akan memprediksi "
        "suhu pada tanggal setelahnya (H+1) — cocok untuk mengecek data historis tertentu, "
        "atau simulasi 'bagaimana jika kondisi besok seperti ini'."
    )

    with st.form("input_form"):
        tanggal_x = st.date_input(
            "Tanggal Data (Hari X)",
            value=datetime.now().date(),
            max_value=datetime.now().date(),
            help="Maksimal hari ini — kondisi cuaca untuk tanggal ini harus sudah benar-benar "
                 "terjadi/terukur, bukan tanggal di masa depan yang datanya belum ada."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            rh = st.number_input("Kelembapan Rata-Rata (RH) [%]", min_value=0.0, max_value=100.0, value=80.00, step=0.1, format="%.2f")
            rr = st.number_input("Curah Hujan (RR) [mm]", min_value=0.0, value=0.00, step=0.1, format="%.2f")

        with col2:
            ss = st.number_input("Penyinaran Matahari (SS) [Jam]", min_value=0.0, max_value=24.0, value=6.00, step=0.1, format="%.2f")
            ff = st.number_input("Kecepatan Angin (FF) [m/s]", min_value=0.0, value=2.00, step=0.1, format="%.2f")

        with col3:
            pressure = st.number_input("Tekanan Udara [hPa / mb]", min_value=800.0, max_value=1100.0, value=1010.00, step=0.1, format="%.2f")
            tavg = st.number_input("Suhu Rata-Rata Hari Itu (°C)", value=27.50, step=0.1, format="%.2f")

        submitted = st.form_submit_button("Prediksi Hari Berikutnya", use_container_width=True)

        st.caption(
            "Semua nilai di atas dianggap sebagai kondisi cuaca pada 'Tanggal Data' yang dipilih. "
            "Model akan memprediksi suhu pada tanggal setelahnya (H+1). "
            "Angka default di atas hanyalah contoh — silahkan ubah sesuai kondisi yang ingin disimulasikan."
        )

    if submitted:
        st.divider()

        tanggal_x_dt = datetime.combine(tanggal_x, datetime.min.time())
        tanggal_besok, suhu_besok = prediksi_h_plus_1(
            model_forecast, rh=rh, rr=rr, pressure=pressure, ss=ss, ff=ff, tavg=tavg,
            tanggal_x=tanggal_x_dt
        )

        render_hasil(tanggal_x_dt, tanggal_besok, tavg, suhu_besok, sumber_label="input manual", mae=mae_forecast)

        st.markdown("---")
        st.caption(
            "Model Random Forest (forecast H+1) dilatih menggunakan data historis BMKG stasiun YIA "
            "(14 Jul 2024 - 12 Jul 2026). Fitur yang dipakai: kelembapan, curah hujan, tekanan udara, "
            "penyinaran matahari, kecepatan angin, dan suhu pada hari X, untuk memprediksi suhu hari X+1."
        )
