from pathlib import Path
from datetime import datetime, timedelta
import base64
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="CuacaTani Cabai – Informasi Cuaca & Prediksi Suhu untuk Petani Cabai",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data_bmkg_fix.csv"
MODEL_PATH = BASE_DIR / "model_rf_forecast.pkl"
METRICS_PATH = BASE_DIR / "hasil_model_forecast.csv"
PREDICTIONS_PATH = BASE_DIR / "hasil_prediksi_forecast.csv"
CALIBRATION_PATH = BASE_DIR / "kalibrasi_openmeteo.json"

# Koordinat resmi Stasiun Meteorologi Kelas II Yogyakarta (YIA), Temon, Kulon Progo
LATITUDE = -7.90746
LONGITUDE = 110.05448

FEATURES = [
    "RH_AVG_LAG_1", "RR_LAG_1", "PRESSURE_LAG_1", "SS_LAG_1",
    "FF_AVG_LAG_1", "MONTH", "DAY", "TAVG_LAG_1",
]

HARI_ID = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
BULAN_ID = [
    "", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember",
]

# =========================================================
# IKON SVG RESMI PARAMETER CUACA (TAJAM, ELEGAN, VEKTOR)
# =========================================================
ICON_THERMO = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2D6A4F" '
    'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"/>'
    '</svg>'
)
ICON_HUMIDITY = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2D6A4F" '
    'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/>'
    '</svg>'
)
ICON_RAIN = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2D6A4F" '
    'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"/>'
    '<path d="M16 14v6"/><path d="M8 14v6"/><path d="M12 16v6"/>'
    '</svg>'
)
ICON_WIND = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2D6A4F" '
    'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2"/>'
    '</svg>'
)
ICON_PRESSURE = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2D6A4F" '
    'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="10"/><path d="m14 14-4-4"/><path d="M12 6v2"/><path d="M6 12h2"/><path d="M18 12h-2"/>'
    '</svg>'
)
ICON_LOCATION = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2D6A4F" '
    'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/>'
    '</svg>'
)
ICON_CHIP = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2D6A4F" '
    'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>'
    '<path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>'
    '</svg>'
)



def format_indo_date(dt, with_day=True):
    try:
        day_str = f"{HARI_ID[dt.weekday()]}, " if with_day else ""
        return f"{day_str}{dt.day} {BULAN_ID[dt.month]} {dt.year}"
    except Exception:
        return str(dt)


@st.cache_data
def get_base64_image(file_path: Path) -> str:
    if not file_path.exists():
        return ""
    mime_type = "image/png" if file_path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def render_html(content: str):
    """Render HTML secara aman tanpa risiko markdown indented code-block parser bug."""
    if hasattr(st, "html"):
        st.html(content)
    else:
        st.markdown(content, unsafe_allow_html=True)





CHILI_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="32" height="32" fill="none">'
    '<!-- Daun Cabai (Leaf) -->'
    '<path d="M16 9C18 5.5 22.5 4 26 4.8C26.5 8.5 23.5 12 19.5 12.8C17.5 12.2 16.5 10.8 16 9Z" fill="#FAF6EB"/>'
    '<path d="M18 8.5C21 7.2 23.5 6 25 5" stroke="#174C3A" stroke-width="1.2" stroke-linecap="round"/>'
    '<!-- Tangkai Cabai (Stem) -->'
    '<path d="M14.5 11C14 7.5 15 4.5 17.5 2.5" stroke="#FAF6EB" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>'
    '<!-- Kelopak Cabai (Calyx) -->'
    '<path d="M9.5 11.5C11 13.5 13.5 13.8 15 13C16.5 13.8 19 13.5 20.5 11.5C19 10.5 11 10.5 9.5 11.5Z" fill="#FAF6EB"/>'
    '<!-- Buah Cabai (Chili Body) -->'
    '<path d="M10 12.5C8 17 8.5 22 12.5 25.5C14.5 27.5 17.5 28.5 19.8 27.8C20.6 27.5 20.5 26.2 19.5 25.5C17.2 23.8 16.8 20 17.5 16.5C17.8 15 18.8 13.8 19.5 12.5C16.5 13.2 13 13.2 10 12.5Z" fill="#FAF6EB"/>'
    '</svg>'
)
CHILI_ICON_BASE64 = f"data:image/svg+xml;base64,{base64.b64encode(CHILI_SVG.encode('utf-8')).decode('utf-8')}"


# =========================================================
# CSS TEMA FOREST GREEN + SAGE GREEN + WARM CREAM
# =========================================================
render_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

:root {
  /* 1. Forest Green (#174C3A): Sidebar, Tombol Utama, Elemen Navigasi */
  --forest-green:  #174C3A;
  --forest-dark:   #10382B;
  --forest-mid:    #1F5C48;
  --fg-dark:       #174C3A;
  --fg-mid:        #1F5C48;
  --fg-soft:       #2D725A;

  /* 2. Sage Green (#A8C3AD): Kartu Informasi Sekunder & Elemen Pendukung */
  --sage-main:     #A8C3AD;
  --sage-border:   #A8C3AD;
  --sage-soft:     #EBF2ED;
  --sage-bg:       #F0F6F2;
  --sage-light:    #D5E4D8;

  /* 3. Soft Cream (#F8F6EF): Latar Belakang Utama */
  --cream-main:    #F8F6EF;
  --cream-soft:    #F1EFE6;

  /* 4. Warm White (#FFFFFF): Kartu Prediksi, Formulir, Area Konten Utama */
  --warm-white:    #FFFFFF;
  --cream-card:    #FFFFFF;

  /* 5. Muted Gold (#D6AD55): Aksen Angka Suhu, Highlight, Informasi Penting */
  --gold:          #D6AD55;
  --gold-dark:     #B8892E;
  --gold-soft:     #E8D3A2;
  --gold-bg:       #FAF6EB;

  /* 6. Charcoal (#29352F): Teks Utama */
  --charcoal:      #29352F;
  --text-dark:     #29352F;
  --text-mid:      #3B4B43;
  --text-muted:    #54665C;

  /* Borders & Dividers */
  --border:        #D6E3D9;
  --border-strong: #A8C3AD;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
  overflow-x: hidden !important;
  max-width: 100vw !important;
}

.stApp {
  background: var(--cream-main);
  color: var(--charcoal);
  font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Header Streamlit: Transparan agar tidak menutupi judul & konten, tombol navigasi sidebar tetap dapat diakses di HP */
header[data-testid="stHeader"] {
  background: transparent !important;
  color: var(--forest-green) !important;
  height: 2.8rem !important;
  z-index: 99 !important;
  pointer-events: none !important;
}

header[data-testid="stHeader"] * {
  pointer-events: auto !important;
}

[data-testid="stDecoration"] {
  display: none !important;
}

/* Tombol buka sidebar di HP agar jelas, rapi, dan mudah disentuh */
[data-testid="stSidebarCollapsedControl"] {
  display: flex !important;
  visibility: visible !important;
  background: var(--warm-white) !important;
  border: 1.5px solid var(--sage-border) !important;
  border-radius: 10px !important;
  box-shadow: 0 3px 12px rgba(23, 76, 58, 0.16) !important;
  color: var(--forest-green) !important;
  top: 0.65rem !important;
  left: 0.75rem !important;
  min-width: 44px !important;
  min-height: 44px !important;
  padding: 8px 10px !important;
  align-items: center !important;
  justify-content: center !important;
  z-index: 99999 !important;
}

/* Sembunyikan footer Streamlit agar tidak menutupi tombol atau konten bawah di HP */
footer {
  display: none !important;
}

[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stSidebarCollapsedControl"] svg {
  color: var(--forest-green) !important;
  fill: var(--forest-green) !important;
  stroke: var(--forest-green) !important;
}

/* Sembunyikan SEMUA navigasi multipage bawaan Streamlit agar sidebar HANYA menampilkan 2 menu: Beranda & Tentang */
[data-testid="stSidebarNav"],
[data-testid="stSidebarNavItems"],
[data-testid="stSidebarNavSeparator"],
nav[data-testid="stSidebarNav"],
ul[data-testid="stSidebarNavItems"],
div[data-testid="stSidebarNav"],
section[data-testid="stSidebar"] nav,
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"],
section[data-testid="stSidebar"] ul:has(a[data-testid="stSidebarNavLink"]) {
  display: none !important;
  visibility: hidden !important;
  height: 0 !important;
  max-height: 0 !important;
  overflow: hidden !important;
  margin: 0 !important;
  padding: 0 !important;
  pointer-events: none !important;
}

/* Ruang konten dashboard yang rapi, proporsional, dan tidak terpotong */
.block-container {
  max-width: 1200px;
  padding-top: 1.6rem !important;
  padding-bottom: 3.5rem;
  padding-left: 1.5rem !important;
  padding-right: 1.5rem !important;
}

h1, h2, h3, h4 {
  font-family: 'Manrope', sans-serif;
  color: var(--forest-green);
  letter-spacing: -0.02em;
}

/* ======= SIDEBAR FOREST GREEN (#174C3A) ======= */
section[data-testid="stSidebar"] {
  background: var(--forest-green) !important;
  border-right: none;
}

section[data-testid="stSidebar"] * {
  color: #E2ECE5 !important;
}

section[data-testid="stSidebar"] > div {
  background: var(--forest-green) !important;
}

/* Identitas aplikasi di bagian paling atas sidebar */
[data-testid="stSidebarContent"],
section[data-testid="stSidebar"] > div:first-child,
div[data-testid="stSidebarUserContent"] {
  padding-top: 0.8rem !important;
  padding-left: 1.1rem !important;
  padding-right: 1.1rem !important;
}

[data-testid="stSidebarHeader"] {
  display: none !important;
  height: 0 !important;
  min-height: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
}

.sidebar-minimal-brand {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 2px 2px 14px 2px;
  margin-top: 0;
  border-bottom: 1px solid rgba(255,255,255,0.14);
  margin-bottom: 16px;
}

.sidebar-app-icon {
  width: 42px;
  height: 42px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(255,255,255,0.12);
  border: 1px solid rgba(255,255,255,0.22);
  border-radius: 12px;
  flex-shrink: 0;
  box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}

.sidebar-chili-img {
  width: 28px;
  height: 28px;
  object-fit: contain;
  display: block;
}

.sidebar-station-title {
  font-family: 'Manrope', sans-serif;
  font-size: 0.93rem;
  font-weight: 800;
  color: #FFFFFF !important;
  line-height: 1.25;
}

.sidebar-station-sub {
  font-size: 0.8rem;
  color: var(--gold) !important;
  font-weight: 600;
  letter-spacing: 0.02em;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {
  gap: 8px;
}

/* Judul MENU UTAMA: Murni teks tebal (bold), tanpa kotak, tanpa background, tanpa border/outline */
section[data-testid="stSidebar"] div[data-testid="stRadio"] > label,
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stWidgetLabel"] {
  background: transparent !important;
  background-color: transparent !important;
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
  border-radius: 0 !important;
  padding: 4px 2px 10px 2px !important;
  margin: 0 !important;
  min-height: auto !important;
  height: auto !important;
  display: block !important;
  cursor: default !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] > label p,
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] div[data-testid="stRadio"] > label span,
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stWidgetLabel"] span {
  color: #E2ECE5 !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-size: 0.8rem !important;
  font-weight: 800 !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  background: transparent !important;
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin: 0 !important;
}

/* Tombol opsi navigasi Beranda & Tentang (Tetap mempertahankan desain kotak/pill yang sekarang) */
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label,
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label {
  background: rgba(255,255,255,0.08) !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  border-radius: 10px !important;
  padding: 11px 16px !important;
  margin: 3px 0 !important;
  font-weight: 700 !important;
  font-size: 0.94rem !important;
  color: #FFFFFF !important;
  cursor: pointer;
  min-height: 44px !important;
  display: flex !important;
  align-items: center !important;
  transition: all 0.2s ease;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:hover,
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
  background: rgba(255,255,255,0.18) !important;
  border-color: rgba(255,255,255,0.32) !important;
  color: #FFFFFF !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked),
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
  background: var(--gold) !important;
  border: 1px solid var(--gold-soft) !important;
  color: var(--forest-green) !important;
  font-weight: 800 !important;
  box-shadow: 0 3px 12px rgba(0,0,0,0.25) !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) *,
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) * {
  color: var(--forest-green) !important;
}

.sidebar-menu-label {
  font-size: 0.78rem;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: rgba(255,255,255,0.65) !important;
  margin: 10px 0 6px 2px;
  display: block;
}

.sidebar-footer-note {
  font-size: 0.82rem;
  color: rgba(255,255,255,0.7) !important;
  margin-top: 36px;
  text-align: center;
  line-height: 1.6;
  border-top: 1px solid rgba(255,255,255,0.14);
  padding-top: 14px;
}

/* ======= HEADER CARD ======= */
.header-card {
  position: relative;
  overflow: hidden;
  border-radius: 20px;
  background: linear-gradient(105deg, #1F5C48 0%, #174C3A 55%, #10382B 100%);
  border: 1px solid rgba(255,255,255,0.12);
  padding: 26px 32px;
  margin-bottom: 24px;
  box-shadow: 0 8px 32px rgba(23, 76, 58, 0.24);
  display: flex;
  align-items: center;
  min-height: 170px;
}

.header-content {
  position: relative;
  z-index: 2;
  width: 100%;
}

.header-brand-line {
  margin-bottom: 7px;
}

.header-kicker {
  font-size: 0.7rem;
  font-weight: 800;
  letter-spacing: 0.14em;
  color: var(--gold);
  text-transform: uppercase;
  display: inline-block;
}

.header-title-row {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-top: 2px;
  margin-bottom: 8px;
}

.header-title {
  font-family: 'Manrope', sans-serif;
  font-size: clamp(1.2rem, 2.2vw, 1.65rem);
  font-weight: 800;
  color: #FFFFFF !important;
  margin: 0;
  line-height: 1.25;
}

.header-subtitle {
  font-size: 0.84rem;
  color: rgba(255,255,255,0.78);
  margin: 3px 0 0 0;
  font-weight: 500;
}

.header-badge-row {
  margin-top: 12px;
}

.header-pill {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  background: rgba(255,255,255,0.12);
  border: 1px solid rgba(255,255,255,0.22);
  padding: 5px 13px;
  border-radius: 999px;
  font-size: 0.74rem;
  font-weight: 600;
  color: rgba(255,255,255,0.92);
  backdrop-filter: blur(4px);
}

.dot-pulse {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--gold);
  box-shadow: 0 0 6px rgba(214, 173, 85, 0.7);
  animation: pulse-dot 2s infinite ease-in-out;
}

@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.4; transform: scale(0.85); }
}

/* ======= RESPONSIVITAS MULTI-SCREEN (DESKTOP, LAPTOP, TABLET, HP) ======= */
@media (max-width: 1200px) {
  .block-container {
    max-width: 100%;
    padding-left: 1.2rem !important;
    padding-right: 1.2rem !important;
  }
  .params-grid {
    gap: 10px;
  }
  .param-card {
    padding: 14px 14px;
  }
  .param-value {
    font-size: 1.35rem;
  }
}

@media (max-width: 992px) {
  .hero-main-flex {
    flex-direction: column;
    align-items: flex-start;
    gap: 18px;
  }
  .hero-range-box {
    width: 100%;
    min-width: unset;
  }
  .pred-content-row {
    flex-direction: column;
    align-items: stretch;
    gap: 16px;
  }
  .pred-temp-box {
    width: 100%;
    justify-content: center;
  }
  .pred-desc-box {
    min-width: unset;
    width: 100%;
  }
  /* 5 Parameter di tablet: 3 kolom di baris 1, 2 kolom di baris 2 */
  .params-grid {
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
  }
  .param-card-fifth {
    grid-column: span 2;
  }
  div[data-testid="stHorizontalBlock"] {
    flex-wrap: wrap !important;
    gap: 12px !important;
  }
}

@media (max-width: 768px) {
  /* Kolom dalam form input mandiri agar 100% full width, tidak terpotong tombol plus/minus */
  div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
    flex-direction: column !important;
    gap: 10px !important;
  }
  div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    flex: 1 1 100% !important;
    min-width: 100% !important;
    width: 100% !important;
  }
  .form-note-box {
    margin-top: 4px !important;
  }
  /* Kartu info besar agar mengambil lebar penuh di HP */
  div[data-testid="stHorizontalBlock"]:has(.info-card-full) > div[data-testid="column"] {
    flex: 1 1 100% !important;
    min-width: 100% !important;
  }
  .block-container {
    padding-top: 3.5rem !important; /* ruang agar tombol sidebar floating tidak menimpa header */
    padding-left: 1.1rem !important;
    padding-right: 1.1rem !important;
    padding-bottom: calc(5.5rem + env(safe-area-inset-bottom, 24px)) !important; /* Mencegah terpotong atau tertutup navigasi browser/floating bar HP */
  }
  .header-card {
    min-height: auto;
    padding: 20px 20px;
    border-radius: 18px;
    margin-bottom: 20px;
  }
  .header-content {
    width: 100%;
  }
  .header-kicker {
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    line-height: 1.4;
  }
  .header-title {
    font-size: clamp(1.25rem, 4vw, 1.48rem);
    line-height: 1.32;
    margin: 4px 0 8px 0;
  }
  .header-subtitle {
    font-size: 0.9rem;
    line-height: 1.55;
    color: rgba(255,255,255,0.9);
  }
  .header-pill {
    font-size: 0.78rem;
    padding: 5px 13px;
    margin-top: 6px;
  }
  .weather-hero-card {
    padding: 22px 18px;
    border-radius: 18px;
    margin-bottom: 22px;
  }
  .hero-top-row {
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
  }
  .hero-badge-pill {
    font-size: 0.78rem;
    padding: 5px 12px;
    font-weight: 800;
  }
  .hero-date-text {
    font-size: 0.88rem;
    font-weight: 600;
  }
  .hero-location-bar {
    padding: 10px 14px;
    margin-bottom: 16px;
    border-radius: 12px;
  }
  .hero-location-main {
    font-size: 0.92rem;
    font-weight: 800;
  }
  .hero-location-sub {
    font-size: 0.84rem;
    line-height: 1.52;
    color: var(--text-mid);
  }
  .hero-temp-section {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    gap: 16px !important;
    flex-wrap: nowrap !important;
    width: 100% !important;
  }
  .hero-svg-box {
    width: 82px !important;
    height: 82px !important;
    border-radius: 18px !important;
    flex-shrink: 0 !important;
  }
  .weather-illustration-img {
    width: 66px !important;
    height: 66px !important;
    max-width: 66px !important;
    max-height: 66px !important;
  }
  .hero-temp-big {
    font-size: clamp(2.7rem, 8.5vw, 3.4rem) !important;
    line-height: 1 !important;
  }
  .hero-cond-title {
    font-size: 1.22rem !important;
    font-weight: 800 !important;
    margin-top: 3px !important;
  }
  .hero-feel-text {
    font-size: 0.86rem !important;
    line-height: 1.45 !important;
    margin-top: 3px !important;
  }
  .hero-range-box {
    padding: 14px 16px !important;
    border-radius: 14px !important;
    margin-top: 8px !important;
  }
  .range-item {
    padding: 5px 0 !important;
  }
  .range-label {
    font-size: 0.84rem !important;
    font-weight: 600 !important;
  }
  .range-value {
    font-size: 1.05rem !important;
    font-weight: 800 !important;
  }
  /* Card Prediksi H+1: Ringkas & Menyorot Angka */
  .prediction-card {
    padding: 22px 18px;
    border-radius: 18px;
    margin-bottom: 22px;
  }
  .pred-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 14px;
    padding-bottom: 14px;
  }
  .pred-title {
    font-size: 1.1rem !important;
    line-height: 1.38 !important;
    font-weight: 800 !important;
  }
  .pred-trend-pill {
    align-self: flex-start;
    font-size: 0.84rem;
    padding: 6px 13px;
    font-weight: 700;
  }
  .pred-temp-box {
    width: 100% !important;
    justify-content: center !important;
    padding: 14px 22px !important;
    border-radius: 14px !important;
  }
  .pred-temp-val {
    font-size: clamp(2.8rem, 9vw, 3.5rem);
    line-height: 1;
  }
  .pred-temp-unit {
    font-size: 1.4rem;
  }
  .pred-desc-box {
    width: 100% !important;
    min-width: 0 !important;
    background: #FAF6EB;
    border: 1px solid #E8D3A2;
    border-radius: 14px;
    padding: 14px 16px;
  }
  .pred-desc-box p {
    font-size: 0.92rem !important;
    line-height: 1.68 !important;
    color: #29352F !important;
  }
  /* Benchmark Card di HP */
  .benchmark-card {
    padding: 18px 16px;
    border-radius: 16px;
    margin-bottom: 22px;
  }
  .benchmark-grid {
    grid-template-columns: repeat(2, 1fr) !important;
    gap: 10px !important;
  }
  .benchmark-item-fifth {
    grid-column: 1 / -1 !important;
  }
  section[data-testid="stSidebar"] {
    width: 84vw !important;
    max-width: 320px !important;
  }
}

@media (max-width: 600px) {
  /* 5 Parameter Cuaca rapi 2 kolom pada layar HP dengan padding lega */
  .params-grid {
    grid-template-columns: repeat(2, 1fr) !important;
    gap: 12px !important;
    margin-top: 10px !important;
    margin-bottom: 24px !important;
  }
  .param-card {
    padding: 14px 14px !important;
    border-radius: 14px !important;
  }
  .param-label {
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    font-weight: 700 !important;
  }
  .param-value {
    font-size: 1.42rem !important;
    margin: 6px 0 3px 0 !important;
    font-weight: 800 !important;
  }
  .param-sub {
    font-size: 0.8rem !important;
    line-height: 1.4 !important;
    color: var(--text-muted) !important;
  }
  /* Kartu ke-5 (Tekanan Udara) membentang selebar 2 kolom dengan susunan horizontal rapi */
  .param-card-fifth {
    grid-column: 1 / -1 !important;
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: space-between !important;
    padding: 14px 16px !important;
  }
  .param-card-fifth .param-value {
    margin: 0 !important;
    font-size: 1.48rem !important;
    font-weight: 800 !important;
  }
  .param-card-fifth .param-sub-desktop-hide {
    display: none !important;
  }
  .param-card-fifth .param-sub-mobile-show {
    display: block !important;
    font-size: 0.8rem !important;
    color: var(--text-muted) !important;
    margin-top: 2px !important;
  }
}

@media (max-width: 480px) {
  div.stButton > button,
  div.stDownloadButton > button,
  div[data-testid="stFormSubmitButton"] > button {
    width: 100% !important;
    padding: 0.8rem 1.2rem !important;
    font-size: 0.96rem !important;
    font-weight: 800 !important;
    min-height: 48px !important;
    border-radius: 10px !important;
  }
  div[data-testid="stNumberInput"] input,
  div[data-testid="stDateInput"] input {
    font-size: 16px !important; /* Mencegah auto-zoom browser iOS di HP */
    min-height: 44px !important;
  }
  /* Tabs pada layar kecil */
  div[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-size: 0.86rem !important;
    padding: 8px 14px !important;
    min-height: 44px !important;
  }
  div[data-testid="stCaptionContainer"] p {
    font-size: 0.86rem !important;
    line-height: 1.55 !important;
  }
}

@media (max-width: 340px) {
  .params-grid {
    grid-template-columns: 1fr !important;
    gap: 10px !important;
  }
  .param-card-fifth {
    flex-direction: column !important;
    align-items: flex-start !important;
  }
}

/* ======= HERO CUACA TERKINI (WARM WHITE #FFFFFF) ======= */
.weather-hero-card {
  background: var(--warm-white);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 26px 30px;
  box-shadow: 0 6px 26px rgba(23, 76, 58, 0.07);
  margin-bottom: 20px;
  position: relative;
}

.hero-top-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 18px;
  flex-wrap: wrap;
  gap: 10px;
}

.hero-badge-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--sage-soft);
  border: 1px solid var(--sage-border);
  padding: 4px 12px;
  border-radius: 999px;
  font-size: 0.74rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--forest-green);
}

.hero-date-text {
  font-size: 0.85rem;
  color: var(--text-muted);
  font-weight: 600;
}

.hero-main-flex {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 24px;
}

.hero-temp-section {
  display: flex;
  align-items: center;
  gap: 22px;
}

.weather-illustration-img {
  width: 90px;
  height: 90px;
  max-width: 90px;
  max-height: 90px;
  object-fit: contain;
  display: block;
  filter: drop-shadow(0 4px 10px rgba(23, 76, 58, 0.15));
}

.hero-svg-box {
  width: 110px;
  height: 110px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  background: var(--sage-soft);
  border: 1px solid var(--sage-border);
  border-radius: 20px;
}

.hero-temp-big {
  font-family: 'Manrope', sans-serif;
  font-size: clamp(3rem, 6.5vw, 4.6rem);
  font-weight: 800;
  line-height: 1;
  color: var(--forest-green);
  letter-spacing: -0.04em;
}

.hero-temp-unit {
  font-size: 0.45em;
  font-weight: 800;
  color: var(--gold);
  margin-left: 2px;
}

.hero-cond-title {
  font-size: 1.35rem;
  font-weight: 800;
  color: var(--charcoal);
  margin-top: 4px;
}

.hero-feel-text {
  font-size: 0.88rem;
  color: var(--text-muted);
  margin-top: 2px;
}

.hero-range-box {
  background: var(--sage-bg);
  border: 1px solid var(--sage-border);
  border-radius: 16px;
  padding: 16px 20px;
  min-width: 215px;
}

.range-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 4px 0;
}

.range-label {
  font-size: 0.8rem;
  color: var(--text-muted);
  font-weight: 600;
}

.range-value {
  font-family: 'Manrope', sans-serif;
  font-size: 1.05rem;
  font-weight: 800;
}

/* ======= LOCATION BAR (HERO) ======= */
.hero-location-bar {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  margin-bottom: 14px;
  padding: 9px 14px;
  background: var(--sage-bg);
  border: 1px solid var(--sage-border);
  border-radius: 10px;
  flex-wrap: wrap;
}
.hero-location-pin {
  flex-shrink: 0;
  margin-top: 1px;
  color: var(--forest-green);
  font-size: 1rem;
}
.hero-location-main {
  font-family: 'Manrope', sans-serif;
  font-size: 0.9rem;
  font-weight: 800;
  color: var(--forest-green);
  line-height: 1.3;
}
.hero-location-sub {
  font-size: 0.84rem;
  color: var(--text-muted);
  font-weight: 500;
  margin-top: 2px;
  line-height: 1.5;
}
@media (max-width: 600px) {
  .hero-location-bar {
    padding: 10px 14px;
  }
  .hero-location-main {
    font-size: 0.94rem;
  }
  .hero-location-sub {
    font-size: 0.86rem;
    line-height: 1.55;
  }
}

/* ======= SECTION TITLES ======= */
.section-title-wrap {
  margin: 26px 0 12px 0;
}

.section-kicker {
  font-size: 0.78rem;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--gold);
  margin-bottom: 3px;
}

.section-heading {
  font-size: 1.22rem;
  font-weight: 800;
  color: var(--forest-green);
  margin: 0;
}

/* ======= PARAMETER METRIC CARDS (SAGE GREEN BG #F0F6F2) ======= */
.param-card {
  background: var(--sage-bg);
  border: 1px solid var(--sage-border);
  border-top: 3.5px solid var(--forest-green);
  border-radius: 16px;
  padding: 16px 18px;
  box-shadow: 0 3px 12px rgba(23, 76, 58, 0.05);
  height: 100%;
  transition: all 0.25s ease;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

.param-card:hover {
  background: var(--warm-white);
  border-color: var(--forest-green);
  box-shadow: 0 6px 20px rgba(23, 76, 58, 0.12);
  transform: translateY(-2px);
}

.param-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.param-label {
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-muted);
}

.param-value {
  font-family: 'Manrope', sans-serif;
  font-size: 1.55rem;
  font-weight: 800;
  color: var(--forest-green);
  margin: 8px 0 2px 0;
}

.param-sub {
  font-size: 0.77rem;
  color: var(--text-muted);
}

.param-sub-desktop-hide {
  display: block;
}

.param-sub-mobile-show {
  display: none;
}

/* ======= 5 PARAMETER CUACA GRID (RESPONSIF KONSISTEN) ======= */
.params-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 14px;
  margin-top: 10px;
  margin-bottom: 24px;
}

.param-card-fifth {
  /* Default desktop style behaves like normal param-card */
}

/* ======= PREDIKSI H+1 CARD (WARM WHITE #FFFFFF + GOLD HIGHLIGHT) ======= */
.prediction-card {
  background: var(--warm-white);
  border: 1px solid var(--border);
  border-left: 6px solid var(--gold);
  border-radius: 18px;
  padding: 24px 28px;
  box-shadow: 0 6px 24px rgba(23, 76, 58, 0.07);
  margin-top: 8px;
  margin-bottom: 22px;
}

.pred-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 14px;
  margin-bottom: 18px;
}

.pred-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--gold-bg);
  color: var(--gold-dark);
  border: 1px solid var(--gold-soft);
  border-radius: 999px;
  padding: 4px 14px;
  font-size: 0.74rem;
  font-weight: 800;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.pred-title {
  font-size: 1.15rem;
  font-weight: 800;
  color: #174C3A;
  margin-top: 6px;
}

.pred-content-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 20px;
}

.pred-temp-box {
  display: flex;
  align-items: baseline;
  gap: 6px;
  background: var(--gold-bg);
  border: 1.5px solid var(--gold);
  border-radius: 14px;
  padding: 10px 22px;
  box-shadow: 0 2px 10px rgba(214, 173, 85, 0.16);
}

.pred-temp-val {
  font-family: 'Manrope', sans-serif;
  font-size: clamp(2.5rem, 5.2vw, 3.5rem);
  font-weight: 800;
  color: var(--gold-dark);
  line-height: 1;
}

.pred-temp-unit {
  font-size: 1.4rem;
  font-weight: 800;
  color: var(--gold);
}

.pred-desc-box {
  flex: 1 1 52%;
  min-width: 260px;
}

.pred-trend-pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 5px 14px;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 700;
}

.trend-warmer {
  background: #FEF2F2;
  color: #B91C1C;
  border: 1px solid #FECACA;
}

.trend-cooler {
  background: var(--sage-soft);
  color: var(--forest-green);
  border: 1px solid var(--sage-border);
}

.trend-equal {
  background: var(--gold-bg);
  color: var(--gold-dark);
  border: 1px solid var(--gold-soft);
}

/* ======= BENCHMARK BMKG CARD ======= */
.benchmark-card {
  background: var(--warm-white);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 20px 24px;
  margin-bottom: 22px;
  box-shadow: 0 3px 12px rgba(23, 76, 58, 0.04);
}

.benchmark-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 12px;
}

.benchmark-item-fifth {
  /* Default behavior */
}

/* Form note styling */
.form-note-box {
  background: #F0F6F2;
  border: 1px solid #A8C3AD;
  border-radius: 10px;
  padding: 10px 14px;
  margin-top: 26px;
  font-size: 0.83rem;
  color: #174C3A;
  line-height: 1.5;
}

/* ======= TABS RESPONSIF ======= */
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: 6px;
  background-color: var(--sage-soft);
  padding: 6px 6px;
  border-radius: 12px;
  border: 1px solid var(--sage-border);
  overflow-x: auto !important;
  flex-wrap: nowrap !important;
  -webkit-overflow-scrolling: touch;
  scrollbar-width: none;
}

div[data-testid="stTabs"] [data-baseweb="tab-list"]::-webkit-scrollbar {
  display: none;
}

div[data-testid="stTabs"] button[data-baseweb="tab"] {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-size: 0.88rem !important;
  font-weight: 600 !important;
  color: var(--charcoal) !important;
  padding: 8px 16px !important;
  border-radius: 8px !important;
  background: transparent !important;
  border: none !important;
  white-space: nowrap !important;
  flex-shrink: 0 !important;
  transition: all 0.2s ease !important;
}

div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
  color: var(--forest-green) !important;
  background: rgba(23, 76, 58, 0.08) !important;
}

div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--forest-green) !important;
  background: var(--warm-white) !important;
  box-shadow: 0 2px 10px rgba(23, 76, 58, 0.1) !important;
  border: 1px solid var(--sage-border) !important;
  border-bottom: 3px solid var(--forest-green) !important;
  font-weight: 800 !important;
}

/* Pastikan grafik Plotly dan DataFrame responsif dan tidak menyebabkan horizontal scroll halaman */
.js-plotly-plot, .plot-container {
  max-width: 100% !important;
}
div[data-testid="stPlotlyChart"] {
  width: 100% !important;
  overflow-x: auto !important;
}
div[data-testid="stDataFrame"] {
  border-radius: 12px !important;
  overflow-x: auto !important;
  max-width: 100% !important;
  border: 1px solid var(--border) !important;
  background: var(--warm-white) !important;
}

/* Form Container */
div[data-testid="stForm"] {
  background: var(--warm-white) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 18px !important;
  padding: 24px 26px !important;
  box-shadow: 0 4px 20px rgba(23, 76, 58, 0.05) !important;
}

/* Primary Action Buttons */
div.stButton > button,
div.stDownloadButton > button,
div[data-testid="stFormSubmitButton"] > button {
  background: var(--forest-green) !important;
  color: #FFFFFF !important;
  border: 1px solid var(--forest-dark) !important;
  border-radius: 10px !important;
  padding: 0.65rem 1.4rem !important;
  font-weight: 800 !important;
  font-size: 0.95rem !important;
  letter-spacing: 0.02em !important;
  box-shadow: 0 4px 14px rgba(23, 76, 58, 0.28) !important;
  transition: all 0.2s ease !important;
}

div.stButton > button:hover,
div.stDownloadButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
  background: var(--forest-dark) !important;
  border-color: #082119 !important;
  box-shadow: 0 6px 20px rgba(23, 76, 58, 0.4) !important;
  transform: translateY(-2px) !important;
}

div.stButton > button *,
div.stDownloadButton > button *,
div[data-testid="stFormSubmitButton"] > button * {
  color: #FFFFFF !important;
}

/* Inputs */
div[data-testid="stNumberInput"] input,
div[data-testid="stDateInput"] input {
  background-color: var(--warm-white) !important;
  color: var(--charcoal) !important;
  border: 1.5px solid var(--sage-border) !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
}

div[data-testid="stNumberInput"] input:focus,
div[data-testid="stDateInput"] input:focus {
  border-color: var(--forest-green) !important;
  box-shadow: 0 0 0 3px rgba(23, 76, 58, 0.18) !important;
}

div[data-testid="stNumberInput"] label,
div[data-testid="stDateInput"] label {
  color: var(--charcoal) !important;
  font-weight: 700 !important;
  font-size: 0.88rem !important;
}

/* SVG KEYFRAME ANIMATIONS */
@keyframes rotate-rays {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
.anim-sun-rays {
  animation: rotate-rays 22s linear infinite;
}

@keyframes pulse-sun {
  0%, 100% { transform: scale(1); opacity: 0.95; }
  50% { transform: scale(1.08); opacity: 1; }
}
.anim-sun-pulse {
  animation: pulse-sun 3.5s ease-in-out infinite;
}

@keyframes cloud-float {
  0%, 100% { transform: translate(0, 0); }
  50% { transform: translate(5px, -3px); }
}
.anim-cloud-float {
  animation: cloud-float 4.5s ease-in-out infinite;
}

@keyframes cloud-back {
  0%, 100% { transform: translate(0, 0); }
  50% { transform: translate(-4px, 2px); }
}
.anim-cloud-back {
  animation: cloud-back 5.5s ease-in-out infinite;
}

@keyframes cloud-front {
  0%, 100% { transform: translate(0, 0); }
  50% { transform: translate(5px, -2px); }
}
.anim-cloud-front {
  animation: cloud-front 4.2s ease-in-out infinite;
}

@keyframes fog-drift-1 {
  0%, 100% { transform: translateX(0); opacity: 0.85; }
  50% { transform: translateX(8px); opacity: 0.45; }
}
.anim-fog-1 {
  animation: fog-drift-1 3.5s ease-in-out infinite;
}

@keyframes fog-drift-2 {
  0%, 100% { transform: translateX(0); opacity: 0.7; }
  50% { transform: translateX(-6px); opacity: 0.35; }
}
.anim-fog-2 {
  animation: fog-drift-2 4s ease-in-out infinite;
}

@keyframes fog-drift-3 {
  0%, 100% { transform: translateX(0); opacity: 0.6; }
  50% { transform: translateX(5px); opacity: 0.85; }
}
.anim-fog-3 {
  animation: fog-drift-3 3.2s ease-in-out infinite;
}

@keyframes rain-drop-anim {
  0% { transform: translateY(-8px) translateX(0); opacity: 0; }
  35% { opacity: 1; }
  100% { transform: translateY(18px) translateX(-5px); opacity: 0; }
}
.drop-1 { animation: rain-drop-anim 1.2s linear infinite; }
.drop-2 { animation: rain-drop-anim 1.2s linear infinite 0.3s; }
.drop-3 { animation: rain-drop-anim 1.2s linear infinite 0.6s; }
.drop-4 { animation: rain-drop-anim 1.2s linear infinite 0.9s; }
.rain-1 { animation: rain-drop-anim 0.85s linear infinite; }
.rain-2 { animation: rain-drop-anim 0.85s linear infinite 0.2s; }
.rain-3 { animation: rain-drop-anim 0.85s linear infinite 0.4s; }
.rain-4 { animation: rain-drop-anim 0.85s linear infinite 0.6s; }
.rain-5 { animation: rain-drop-anim 0.85s linear infinite 0.75s; }

@keyframes thunder-flash-anim {
  0%, 82%, 94%, 100% { opacity: 0; }
  85%, 89% { opacity: 1; filter: drop-shadow(0 0 10px #facc15); }
}
.anim-lightning {
  animation: thunder-flash-anim 3.2s infinite ease-in-out;
}
</style>
""")


# =========================================================
# FUNGSI RENDER HEADER KONSISTEN (BERANDA & TENTANG)
# =========================================================
def render_page_header(title: str, subtitle: str, badge_text: str):
    """Menampilkan header bersih dan profesional tanpa logo maupun foto gedung."""
    render_html(f"""
    <div class="header-card">
        <div class="header-content">
            <div class="header-brand-line">
                <span class="header-kicker">CUACATANI CABAI · SISTEM INFORMASI CUACA UNTUK PETANI CABAI</span>
            </div>
            <div class="header-title-row">
                <div>
                    <h1 class="header-title">{title}</h1>
                    <p class="header-subtitle">{subtitle}</p>
                </div>
            </div>
            <div class="header-badge-row">
                <span class="header-pill">
                    <span class="dot-pulse"></span> {badge_text}
                </span>
            </div>
        </div>
    </div>
    """)


# =========================================================
# FUNGSI ILUSTRASI CUACA DINAMIS SOFT 2D / SEMI-FLAT (DATA URI)
# =========================================================
def _svg_to_data_uri(svg_code: str, desc: str = "Ilustrasi Cuaca") -> str:
    """Mengonversi string SVG ke Data URI Base64 di dalam tag <img> agar kebal dari sanitasi HTML Streamlit."""
    b64 = base64.b64encode(svg_code.strip().encode("utf-8")).decode("utf-8")
    return f'<img src="data:image/svg+xml;base64,{b64}" class="weather-illustration-img" alt="{desc}" />'


def _svg_cerah() -> str:
    """1. CERAH: Matahari bersinar semi-flat hangat dengan berkas sinar emas dan kilau halus."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" width="120" height="120">'
        '<defs>'
        '<linearGradient id="sg" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#FCE182"/>'
        '<stop offset="60%" stop-color="#EBB338"/>'
        '<stop offset="100%" stop-color="#D4971A"/>'
        '</linearGradient>'
        '<radialGradient id="sh" cx="50%" cy="50%" r="50%">'
        '<stop offset="0%" stop-color="#FCE182" stop-opacity="0.45"/>'
        '<stop offset="70%" stop-color="#EBB338" stop-opacity="0.15"/>'
        '<stop offset="100%" stop-color="#D4971A" stop-opacity="0"/>'
        '</radialGradient>'
        '</defs>'
        '<circle cx="60" cy="60" r="46" fill="url(#sh)"/>'
        '<g stroke="#EBB338" stroke-width="4.5" stroke-linecap="round">'
        '<line x1="60" y1="12" x2="60" y2="24"/>'
        '<line x1="60" y1="96" x2="60" y2="108"/>'
        '<line x1="12" y1="60" x2="24" y2="60"/>'
        '<line x1="96" y1="60" x2="108" y2="60"/>'
        '<line x1="26" y1="26" x2="35" y2="35"/>'
        '<line x1="85" y1="85" x2="94" y2="94"/>'
        '<line x1="26" y1="94" x2="35" y2="85"/>'
        '<line x1="85" y1="35" x2="94" y2="26"/>'
        '</g>'
        '<circle cx="60" cy="60" r="26" fill="url(#sg)"/>'
        '<ellipse cx="53" cy="53" rx="14" ry="11" fill="#FFFFFF" opacity="0.25"/>'
        '</svg>'
    )
    return _svg_to_data_uri(svg, "Cuaca Cerah")


def _svg_cerah_berawan() -> str:
    """2. CERAH BERAWAN: Matahari hangat berpadu dengan awan tumpuk soft white/sage."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" width="120" height="120">'
        '<defs>'
        '<linearGradient id="sg_cb" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#FCE182"/>'
        '<stop offset="100%" stop-color="#EBB338"/>'
        '</linearGradient>'
        '<linearGradient id="cg_cb" x1="0%" y1="0%" x2="0%" y2="100%">'
        '<stop offset="0%" stop-color="#FFFFFF"/>'
        '<stop offset="70%" stop-color="#F2F8F1"/>'
        '<stop offset="100%" stop-color="#D7E8D5"/>'
        '</linearGradient>'
        '<linearGradient id="cb_cb" x1="0%" y1="0%" x2="0%" y2="100%">'
        '<stop offset="0%" stop-color="#E4EFE3"/>'
        '<stop offset="100%" stop-color="#B8CFB6"/>'
        '</linearGradient>'
        '</defs>'
        '<circle cx="78" cy="40" r="22" fill="url(#sg_cb)"/>'
        '<g stroke="#EBB338" stroke-width="3.5" stroke-linecap="round">'
        '<line x1="78" y1="12" x2="78" y2="18"/>'
        '<line x1="102" y1="40" x2="108" y2="40"/>'
        '<line x1="95" y1="24" x2="100" y2="19"/>'
        '</g>'
        '<path d="M48 74h46a14 14 0 0 0 4-27.4 20 20 0 0 0-38-5.6 15 15 0 0 0-12 33z" fill="url(#cb_cb)" opacity="0.85"/>'
        '<path d="M30 82h56a17 17 0 0 0 4-33.5 24 24 0 0 0-45-7 18 18 0 0 0-15 40.5z" fill="url(#cg_cb)"/>'
        '</svg>'
    )
    return _svg_to_data_uri(svg, "Cuaca Cerah Berawan")


def _svg_berawan() -> str:
    """3. BERAWAN: Awan tebal bertumpuk mendominasi (nuansa sage-slate dan soft cream)."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" width="120" height="120">'
        '<defs>'
        '<linearGradient id="cb_ov" x1="0%" y1="0%" x2="0%" y2="100%">'
        '<stop offset="0%" stop-color="#BACFBA"/>'
        '<stop offset="100%" stop-color="#7C997A"/>'
        '</linearGradient>'
        '<linearGradient id="cf_ov" x1="0%" y1="0%" x2="0%" y2="100%">'
        '<stop offset="0%" stop-color="#FFFFFF"/>'
        '<stop offset="65%" stop-color="#EDF5EC"/>'
        '<stop offset="100%" stop-color="#CBE0C9"/>'
        '</linearGradient>'
        '</defs>'
        '<path d="M46 64h48a15 15 0 0 0 3.5-29.6 22 22 0 0 0-41.5-6.4 17 17 0 0 0-10 36z" fill="url(#cb_ov)"/>'
        '<path d="M26 86h62a19 19 0 0 0 4-37.5 26 26 0 0 0-49-7.5 20 20 0 0 0-17 45z" fill="url(#cf_ov)"/>'
        '</svg>'
    )
    return _svg_to_data_uri(svg, "Cuaca Berawan")


def _svg_berkabut() -> str:
    """4. BERKABUT: Awan rendah soft sage dengan garis kabut horizontal berlapis."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" width="120" height="120">'
        '<defs>'
        '<linearGradient id="fg_s" x1="0%" y1="0%" x2="0%" y2="100%">'
        '<stop offset="0%" stop-color="#FFFFFF"/>'
        '<stop offset="60%" stop-color="#E2EEE0"/>'
        '<stop offset="100%" stop-color="#BACFBA"/>'
        '</linearGradient>'
        '</defs>'
        '<path d="M34 52h52a16 16 0 0 0 4-31.5 22 22 0 0 0-42-6 17 17 0 0 0-14 37.5z" fill="url(#fg_s)"/>'
        '<g stroke-linecap="round">'
        '<line x1="20" y1="66" x2="100" y2="66" stroke="#40916C" stroke-width="4.5" opacity="0.9"/>'
        '<line x1="26" y1="78" x2="94" y2="78" stroke="#52B788" stroke-width="4.5" opacity="0.8"/>'
        '<line x1="34" y1="90" x2="86" y2="90" stroke="#74C69D" stroke-width="4.5" opacity="0.7"/>'
        '</g>'
        '</svg>'
    )
    return _svg_to_data_uri(svg, "Udara Berkabut")


def _svg_hujan() -> str:
    """5. HUJAN: Awan sage-slate dengan tetesan air hujan teal-sage teratur."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" width="120" height="120">'
        '<defs>'
        '<linearGradient id="rc_s" x1="0%" y1="0%" x2="0%" y2="100%">'
        '<stop offset="0%" stop-color="#E2EEE0"/>'
        '<stop offset="50%" stop-color="#B2CBB0"/>'
        '<stop offset="100%" stop-color="#739471"/>'
        '</linearGradient>'
        '</defs>'
        '<path d="M30 62h58a17 17 0 0 0 4-33.5 24 24 0 0 0-46-6.5 19 19 0 0 0-16 40z" fill="url(#rc_s)"/>'
        '<g stroke-linecap="round" stroke-width="3.5">'
        '<line x1="38" y1="72" x2="33" y2="88" stroke="#2D6A4F"/>'
        '<line x1="51" y1="74" x2="46" y2="90" stroke="#40916C"/>'
        '<line x1="64" y1="72" x2="59" y2="88" stroke="#2D6A4F"/>'
        '<line x1="77" y1="74" x2="72" y2="90" stroke="#40916C"/>'
        '<line x1="90" y1="72" x2="85" y2="88" stroke="#2D6A4F"/>'
        '</g>'
        '</svg>'
    )
    return _svg_to_data_uri(svg, "Cuaca Hujan")


def _svg_badai() -> str:
    """6. BADAI: Awan gelap forest-slate dengan kilatan petir emas tajam."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" width="120" height="120">'
        '<defs>'
        '<linearGradient id="st_s" x1="0%" y1="0%" x2="0%" y2="100%">'
        '<stop offset="0%" stop-color="#496558"/>'
        '<stop offset="55%" stop-color="#2D463B"/>'
        '<stop offset="100%" stop-color="#182D24"/>'
        '</linearGradient>'
        '</defs>'
        '<path d="M28 60h62a18 18 0 0 0 4-35.5 25 25 0 0 0-48-7 20 20 0 0 0-18 42.5z" fill="url(#st_s)"/>'
        '<polygon points="59,56 46,76 57,76 48,100 75,71 63,71" fill="#FACC15"/>'
        '<g stroke="#52B788" stroke-width="3.2" stroke-linecap="round">'
        '<line x1="36" y1="72" x2="31" y2="88"/>'
        '<line x1="80" y1="72" x2="75" y2="88"/>'
        '<line x1="92" y1="74" x2="87" y2="90"/>'
        '</g>'
        '</svg>'
    )
    return _svg_to_data_uri(svg, "Cuaca Badai Petir")


def _svg_default_neutral() -> str:
    """7. DEFAULT NETRAL: Komposisi awan sage halus dan matahari hangat jika data cuaca tidak tersedia."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" width="120" height="120">'
        '<defs>'
        '<linearGradient id="df_s" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#FCE182"/>'
        '<stop offset="100%" stop-color="#EBB338"/>'
        '</linearGradient>'
        '<linearGradient id="df_c" x1="0%" y1="0%" x2="0%" y2="100%">'
        '<stop offset="0%" stop-color="#FFFFFF"/>'
        '<stop offset="60%" stop-color="#EFF6EE"/>'
        '<stop offset="100%" stop-color="#D5E6D3"/>'
        '</linearGradient>'
        '</defs>'
        '<circle cx="75" cy="46" r="20" fill="url(#df_s)"/>'
        '<path d="M32 80h54a17 17 0 0 0 4-33.5 24 24 0 0 0-45-7 18 18 0 0 0-13 40.5z" fill="url(#df_c)"/>'
        '</svg>'
    )
    return _svg_to_data_uri(svg, "Kondisi Cuaca")


def render_weather_svg(code: int) -> str:
    """Menghasilkan ilustrasi cuaca visual (Data URI Image) yang pasti muncul dan tidak terpengaruh sanitasi HTML."""
    if code is None:
        return _svg_default_neutral()

    try:
        code = int(code)
    except (ValueError, TypeError):
        return _svg_default_neutral()

    if code == 0:
        return _svg_cerah()
    elif code in [1, 2]:
        return _svg_cerah_berawan()
    elif code == 3:
        return _svg_berawan()
    elif code in [45, 48]:
        return _svg_berkabut()
    elif code in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82]:
        return _svg_hujan()
    elif code in [95, 96, 99]:
        return _svg_badai()
    return _svg_default_neutral()


def get_weather_desc(code: int) -> str:
    """Mengembalikan label teks kondisi cuaca aktual yang bersesuaian dengan ilustrasi cuaca."""
    if code is None:
        return "Cuaca Yogyakarta"
    try:
        code = int(code)
    except (ValueError, TypeError):
        return "Cuaca Yogyakarta"

    if code == 0:
        return "Cerah"
    elif code in [1, 2]:
        return "Cerah Berawan"
    elif code == 3:
        return "Berawan"
    elif code in [45, 48]:
        return "Berkabut"
    elif code in [51, 53, 55, 56, 57]:
        return "Hujan Gerimis"
    elif code in [61, 63, 65, 66, 67, 80, 81, 82]:
        return "Hujan"
    elif code in [95, 96, 99]:
        return "Badai Petir"
    return "Cerah Berawan"


def get_temp_feel_label(temp: float) -> str:
    if temp < 24.0:
        return "Sejuk"
    elif temp < 28.5:
        return "Cenderung Nyaman"
    elif temp < 32.0:
        return "Cenderung Hangat"
    return "Terasa Panas / Terik"


# =========================================================
# FUNGSI PEMUATAN DATA & MODEL
# =========================================================
@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_PATH, parse_dates=["TANGGAL"])
    df = df.dropna(subset=["TANGGAL"]).sort_values("TANGGAL").reset_index(drop=True)
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_PATH)


@st.cache_data
def load_predictions():
    df = pd.read_csv(PREDICTIONS_PATH, parse_dates=["TANGGAL"])
    return df.dropna(subset=["TANGGAL"]).sort_values("TANGGAL").reset_index(drop=True)


@st.cache_data
def load_calibration():
    if not CALIBRATION_PATH.exists():
        return None
    try:
        with open(CALIBRATION_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if "bias" in data:
            if data.get("latitude") == LATITUDE and data.get("longitude") == LONGITUDE:
                return data.get("bias", {})
            return None
        if isinstance(data, dict) and "TAVG" in data:
            return data
        return None
    except Exception:
        return None


@st.cache_data(ttl=1800)
def get_live_weather():
    """Mengambil kondisi cuaca terkini dan harian dari Open-Meteo pada koordinat YIA."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,surface_pressure,is_day",
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,wind_speed_10m_mean,sunshine_duration,weather_code",
        "hourly": "surface_pressure",
        "timezone": "Asia/Jakarta",
        "wind_speed_unit": "ms",
        "forecast_days": 1,
    }
    response = requests.get(url, params=params, timeout=12)
    response.raise_for_status()
    payload = response.json()

    daily = payload["daily"]
    current = payload.get("current", {})
    hourly_pres = [v for v in payload.get("hourly", {}).get("surface_pressure", []) if v is not None]
    pres_mean = float(np.mean(hourly_pres)) if hourly_pres else current.get("surface_pressure", 1010.5)

    weather_code = current.get("weather_code")
    if weather_code is None:
        weather_code_list = daily.get("weather_code") or [0]
        weather_code = weather_code_list[0] if weather_code_list else 0

    t_mean = daily["temperature_2m_mean"][0] if daily.get("temperature_2m_mean") else current.get("temperature_2m", 28.0)
    t_curr = current.get("temperature_2m", t_mean)
    t_feel = current.get("apparent_temperature", t_curr)
    t_max = daily.get("temperature_2m_max", [t_curr + 2.5])[0]
    t_min = daily.get("temperature_2m_min", [t_curr - 2.5])[0]

    rh = current.get("relative_humidity_2m", daily.get("relative_humidity_2m_mean", [80.0])[0])
    rr = daily.get("precipitation_sum", [current.get("precipitation", 0.0)])[0] or 0.0
    ff = current.get("wind_speed_10m", daily.get("wind_speed_10m_mean", [2.5])[0])
    ss = (daily.get("sunshine_duration", [0.0])[0] or 0.0) / 3600

    # Terapan kalibrasi secara konsisten ke seluruh parameter temperatur
    bias = load_calibration()
    if bias and "TAVG" in bias:
        t_bias = float(bias["TAVG"])
        t_curr -= t_bias
        t_feel -= t_bias
        t_max -= t_bias
        t_min -= t_bias
        t_mean -= t_bias
        calibrated = True
    else:
        calibrated = False

    if bias:
        if "RH_AVG" in bias:
            rh -= float(bias["RH_AVG"])
        if "RR" in bias:
            rr -= float(bias["RR"])
        if "FF_AVG" in bias:
            ff -= float(bias["FF_AVG"])
        if "SS" in bias:
            ss -= float(bias["SS"])
        if "PRESSURE" in bias:
            pres_mean -= float(bias["PRESSURE"])

    # PEMASTIAN KONSISTENSI FISIK MUTLAK
    t_max = max(t_max, t_curr, t_mean)
    t_min = min(t_min, t_curr, t_mean)

    rr = max(0.0, rr)
    ss = max(0.0, ss)
    rh = max(0.0, min(100.0, rh))

    return {
        "TANGGAL": pd.to_datetime(daily["time"][0]),
        "TAVG": float(t_mean),
        "TEMP_CURRENT": float(t_curr),
        "TEMP_FEEL": float(t_feel),
        "TEMP_MAX": float(t_max),
        "TEMP_MIN": float(t_min),
        "RH_AVG": float(rh),
        "RR": float(rr),
        "FF_AVG": float(ff),
        "SS": float(ss),
        "PRESSURE": float(pres_mean),
        "WEATHER_CODE": int(weather_code) if weather_code is not None else 0,
        "SOURCE": "live",
        "_calibrated": calibrated,
    }


def predict_tomorrow(model, weather: dict):
    """Menghitung estimasi suhu udara besok (H+1) dengan model Random Forest berdasarkan data cuaca live/stasiun."""
    tomorrow = weather["TANGGAL"] + timedelta(days=1)
    x = pd.DataFrame([{
        "RH_AVG_LAG_1": float(weather["RH_AVG"]),
        "RR_LAG_1": float(weather["RR"]),
        "PRESSURE_LAG_1": float(weather["PRESSURE"]),
        "SS_LAG_1": float(weather["SS"]),
        "FF_AVG_LAG_1": float(weather["FF_AVG"]),
        "MONTH": int(tomorrow.month),
        "DAY": int(tomorrow.day),
        "TAVG_LAG_1": float(weather["TAVG"]),
    }])
    if hasattr(model, "feature_names_in_"):
        x = x[list(model.feature_names_in_)]
    else:
        x = x[FEATURES]
    return tomorrow, float(model.predict(x)[0])


def predict_manual_input(model, rh: float, rr: float, pressure: float, ss: float, ff: float, tavg: float, tanggal_x: datetime):
    """Menghitung estimasi suhu udara H+1 dari input parameter cuaca manual menggunakan model Random Forest."""
    tomorrow = tanggal_x + timedelta(days=1)
    x = pd.DataFrame([{
        "RH_AVG_LAG_1": float(rh),
        "RR_LAG_1": float(rr),
        "PRESSURE_LAG_1": float(pressure),
        "SS_LAG_1": float(ss),
        "FF_AVG_LAG_1": float(ff),
        "MONTH": int(tomorrow.month),
        "DAY": int(tomorrow.day),
        "TAVG_LAG_1": float(tavg),
    }])
    if hasattr(model, "feature_names_in_"):
        x = x[list(model.feature_names_in_)]
    else:
        x = x[FEATURES]
    return tomorrow, float(model.predict(x)[0])


# =========================================================
# SIDEBAR MINIMALIS: NAMA APLIKASI & 2 MENU
# =========================================================
with st.sidebar:
    render_html(f"""
    <div class="sidebar-minimal-brand">
        <div class="sidebar-app-icon" title="CuacaTani Cabai">
            <img src="{CHILI_ICON_BASE64}" class="sidebar-chili-img" alt="Ikon CuacaTani Cabai" />
        </div>
        <div>
            <div class="sidebar-station-title">CUACATANI CABAI</div>
            <div class="sidebar-station-sub">Info Cuaca untuk Petani Cabai</div>
        </div>
    </div>
    """)

    menu = st.radio(
        "MENU UTAMA",
        ["Beranda", "Tentang"],
        label_visibility="visible",
    )

    render_html("""
    <div class="sidebar-footer-note">
        Kawasan Temon, Kulon Progo<br>
        Daerah Istimewa Yogyakarta
    </div>
    """)


# =========================================================
# HALAMAN 1: BERANDA
# =========================================================
if menu == "Beranda":
    # 1. HEADER BERANDA
    render_page_header(
        title="CuacaTani Cabai – Informasi Cuaca & Prediksi Suhu",
        subtitle="Sistem pemantauan kondisi cuaca harian dan perkiraan suhu esok hari untuk membantu petani cabai di Kawasan Temon, Kulon Progo",
        badge_text="Pemantauan Cuaca Petani Cabai · Prediksi Suhu H+1",
    )

    # 2. LOAD DATASET & MODEL
    try:
        df = load_dataset()
        model = load_model()
        metrics = load_metrics()

        # Ambil data cuaca (Open-Meteo dengan fallback aman ke rekaman historis lokal)
        try:
            weather = get_live_weather()
            source_mode = "live"
        except Exception:
            source_mode = "dataset"
            last = df.iloc[-1]
            tavg = float(last["TAVG"])
            rh = float(last["RH_AVG"])
            rr = float(last["RR"])
            ss = float(last["SS"])
            ff = float(last["FF_AVG"])
            pres = float(last["PRESSURE"])

            # Estimasi kondisi cuaca dari data curah hujan & penyinaran
            if rr >= 5.0:
                w_code = 63
            elif rr > 0.0:
                w_code = 51
            elif ss >= 6.0:
                w_code = 0
            elif ss >= 3.0:
                w_code = 1
            else:
                w_code = 3

            t_min = round(tavg - 2.4, 1)
            t_max = round(tavg + 3.1, 1)
            t_curr = round(tavg + 0.5, 1)
            # Pastikan konsistensi fisik
            t_max = max(t_max, t_curr)
            t_min = min(t_min, t_curr)

            weather = {
                "TANGGAL": pd.to_datetime(last["TANGGAL"]),
                "TAVG": tavg,
                "TEMP_CURRENT": t_curr,
                "TEMP_FEEL": t_curr + 1.2 if rh > 75 else t_curr,
                "TEMP_MAX": t_max,
                "TEMP_MIN": t_min,
                "RH_AVG": rh,
                "RR": rr,
                "FF_AVG": ff,
                "SS": ss,
                "PRESSURE": pres,
                "WEATHER_CODE": w_code,
                "SOURCE": "dataset",
                "_calibrated": False,
            }
            st.info(
                "ℹ️ **Informasi Layanan:** Data live Open-Meteo sedang dalam mode offline. "
                "Menampilkan kondisi berdasarkan data pengamatan stasiun terakhir."
            )

        # Hitung prediksi suhu esok hari (H+1)
        tomorrow, predicted_tavg = predict_tomorrow(model, weather)
        weather_cond_text = get_weather_desc(weather.get("WEATHER_CODE"))
        feel_desc = get_temp_feel_label(weather["TEMP_FEEL"])

        # 3. KARTU SHOWCASE CUACA TERKINI (FOKUS UTAMA - DATA AKTUAL)
        svg_illustration = render_weather_svg(weather.get("WEATHER_CODE"))
        source_badge_text = "CUACA AKTUAL TERKINI (OPEN-METEO)" if source_mode == "live" else "CUACA AKTUAL (CATATAN LOKAL)"

        render_html(f"""
        <div class="weather-hero-card">
            <div class="hero-top-row">
                <span class="hero-badge-pill">
                    <span style="color:#40916C;">●</span> {source_badge_text}
                </span>
                <span class="hero-date-text">
                    Hari Ini: {format_indo_date(weather['TANGGAL'])}
                </span>
            </div>
            <div class="hero-location-bar">
                <span class="hero-location-pin"></span>
                <div>
                    <div class="hero-location-main">Kawasan Temon, Kulon Progo</div>
                    <div class="hero-location-sub">
                        Kabupaten Kulon Progo, D.I. Yogyakarta &nbsp;·&nbsp;
                        Data cuaca aktual diambil dari titik stasiun sekitar lahan pertanian Temon untuk membantu pemantauan harian petani cabai.
                    </div>
                </div>
            </div>
            <div class="hero-main-flex">
                <div class="hero-temp-section">
                    <div class="hero-svg-box">
                        {svg_illustration}
                    </div>
                    <div>
                        <div class="hero-temp-big">
                            {weather['TEMP_CURRENT']:.1f}<span class="hero-temp-unit">°C</span>
                        </div>
                        <div class="hero-cond-title">{weather_cond_text}</div>
                        <div class="hero-feel-text">
                            Suhu terasa: <b>{weather['TEMP_FEEL']:.1f}°C</b> · {feel_desc}
                        </div>
                    </div>
                </div>
                <div class="hero-range-box">
                    <div style="font-size:0.8rem; font-weight:800; color:#174C3A; letter-spacing:0.09em; text-transform:uppercase; margin-bottom:8px;">
                        VARIASI SUHU HARI INI
                    </div>
                    <div class="range-item">
                        <span class="range-label">▲ Suhu Tertinggi</span>
                        <span class="range-value" style="color:#b91c1c;">{weather['TEMP_MAX']:.1f} °C</span>
                    </div>
                    <div class="range-item" style="border-top:1px solid #D5E4D8; padding-top:6px; margin-top:2px;">
                        <span class="range-label">▼ Suhu Terendah</span>
                        <span class="range-value" style="color:#2D725A;">{weather['TEMP_MIN']:.1f} °C</span>
                    </div>
                    <div class="range-item" style="border-top:1px solid #D5E4D8; padding-top:6px; margin-top:2px;">
                        <span class="range-label">● Rata-rata Hari Ini</span>
                        <span class="range-value" style="color:#174C3A;">{weather['TAVG']:.1f} °C</span>
                    </div>
                </div>
            </div>
        </div>
        """)

        # 4. GRID PARAMETER CUACA HARI INI (TAMPILAN UTAMA - RESPONSIF)
        render_html(f"""
        <div class="section-title-wrap">
            <div class="section-kicker">DATA CUACA AKTUAL</div>
            <h3 class="section-heading">Kondisi Cuaca Hari Ini di Kawasan Temon</h3>
        </div>
        <div class="params-grid">
            <div class="param-card">
                <span class="param-label">Suhu Udara</span>
                <div class="param-value">{weather['TAVG']:.1f} <span style="font-size:0.85rem; font-weight:600;">°C</span></div>
                <div class="param-sub">Suhu rata-rata harian</div>
            </div>
            <div class="param-card">
                <span class="param-label">Kelembapan Udara</span>
                <div class="param-value">{weather['RH_AVG']:.0f} %</div>
                <div class="param-sub">Kandungan uap air di udara</div>
            </div>
            <div class="param-card">
                <span class="param-label">Curah Hujan</span>
                <div class="param-value">{weather['RR']:.1f} <span style="font-size:0.85rem; font-weight:600;">mm</span></div>
                <div class="param-sub">Jumlah air hujan hari ini</div>
            </div>
            <div class="param-card">
                <span class="param-label">Kecepatan Angin</span>
                <div class="param-value">{weather['FF_AVG']:.1f} <span style="font-size:0.85rem; font-weight:600;">m/s</span></div>
                <div class="param-sub">Rata-rata laju hembusan angin</div>
            </div>
            <div class="param-card param-card-fifth">
                <div>
                    <span class="param-label">Tekanan Udara</span>
                    <div class="param-sub param-sub-mobile-show">Tekanan udara permukaan</div>
                </div>
                <div class="param-value">{weather['PRESSURE']:.1f} <span style="font-size:0.85rem; font-weight:600;">hPa</span></div>
                <div class="param-sub param-sub-desktop-hide">Tekanan udara permukaan</div>
            </div>
        </div>
        """)

        # 5. SISTEM PREDIKSI SUHU H+1: DUA MODE (OTOMATIS & MANUAL)
        render_html("""
        <div class="section-title-wrap">
            <div class="section-kicker">HASIL PREDIKSI (BUKAN DATA AKTUAL)</div>
            <h3 class="section-heading">Perkiraan Suhu Rata-Rata Esok Hari (H+1)</h3>
        </div>
        """)

        tab_auto, tab_manual = st.tabs([
            "Prediksi Otomatis (Data Cuaca Terkini)",
            "Prediksi Manual (Input Mandiri)",
        ])

        # Ambil metrik MAE & R2 untuk transparansi akurasi model
        mae_row = metrics[metrics["Model Algorithm"].astype(str).str.contains("Random Forest", case=False)]
        mae_val = float(mae_row["MAE (°C)"].iloc[0]) if not mae_row.empty else 0.404
        r2_val = float(mae_row["R² Score"].iloc[0]) if not mae_row.empty else 0.7202

        # -----------------------------------------------------------
        # TAB 1: PREDIKSI OTOMATIS (DATA CUACA TERKINI)
        # -----------------------------------------------------------
        with tab_auto:
            st.caption(
                "Mode **Prediksi Otomatis** menggunakan data kondisi cuaca hari ini dari Open-Meteo untuk memperkirakan "
                "rata-rata suhu udara esok hari (H+1) dengan model Random Forest."
            )

            # Evaluasi selisih suhu vs suhu rata-rata hari ini
            delta_temp = predicted_tavg - weather['TAVG']
            if delta_temp > 0.15:
                trend_class = "trend-warmer"
                trend_icon = "▲"
                trend_label = f"+{delta_temp:.2f} °C (Cenderung lebih hangat dari hari ini)"
            elif delta_temp < -0.15:
                trend_class = "trend-cooler"
                trend_icon = "▼"
                trend_label = f"{delta_temp:.2f} °C (Cenderung lebih sejuk dari hari ini)"
            else:
                trend_class = "trend-equal"
                trend_icon = "■"
                trend_label = "±0.00 °C (Relatif sama dengan hari ini)"

            render_html(f"""
            <div class="prediction-card">
                <div class="pred-header">
                    <div>
                        <div class="pred-badge">HASIL PREDIKSI OTOMATIS (H+1) · RANDOM FOREST</div>
                        <div class="pred-title">
                            Perkiraan Suhu Rata-rata Esok Hari: {format_indo_date(tomorrow)}
                        </div>
                    </div>
                    <div class="pred-trend-pill {trend_class}">
                        {trend_icon} {trend_label}
                    </div>
                </div>
                <div class="pred-content-row">
                    <div class="pred-temp-box">
                        <span class="pred-temp-val">{predicted_tavg:.2f}</span>
                        <span class="pred-temp-unit">°C</span>
                    </div>
                    <div class="pred-desc-box">
                        <p style="font-size:0.92rem; color:#29352F; line-height:1.65; margin:0;">
                            Suhu rata-rata esok hari diperkirakan oleh model <b>Random Forest</b> 
                            berdasarkan data cuaca hari ini (kelembapan, curah hujan, tekanan udara, angin, dan penyinaran matahari).
                            Tingkat akurasi model pada data pengujian: <b>R² = {r2_val:.4f}</b> dengan rata-rata selisih <b>MAE = ±{mae_val:.2f} °C</b>.
                        </p>
                    </div>
                </div>
            </div>
            """)

            # Benchmark rekaman historis observasi lokal sebagai pembanding akurasi
            last_bmkg = df.iloc[-1]
            last_bmkg_date = pd.to_datetime(last_bmkg["TANGGAL"])

            render_html(f"""
            <div class="benchmark-card">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; margin-bottom:14px; border-bottom:1px solid #D6E3D9; padding-bottom:10px;">
                    <div>
                        <b style="color:#174C3A; font-size:1rem;">Data Cuaca Terakhir yang Tercatat pada Dataset</b>
                        <div style="font-size:0.8rem; color:#54665C;">Tanggal Pengamatan: {format_indo_date(last_bmkg_date)}</div>
                    </div>
                    <div style="font-size:0.8rem; background:#FAF6EB; border:1px solid #E8D3A2; padding:4px 12px; border-radius:8px; color:#B8892E; font-weight:700;">
                        Model: Random Forest (300 Estimators)
                    </div>
                </div>
                <div class="benchmark-grid">
                    <div style="background:#F0F6F2; padding:12px 14px; border-radius:12px; border:1px solid #A8C3AD;">
                        <span style="font-size:0.8rem; color:#2D725A; font-weight:800; letter-spacing:0.04em; text-transform:uppercase;">SUHU ACUAN</span>
                        <div style="font-size:1.25rem; font-weight:800; color:#174C3A; margin-top:2px;">{last_bmkg['TAVG']:.1f} °C</div>
                    </div>
                    <div style="background:#F0F6F2; padding:12px 14px; border-radius:12px; border:1px solid #A8C3AD;">
                        <span style="font-size:0.8rem; color:#2D725A; font-weight:800; letter-spacing:0.04em; text-transform:uppercase;">KELEMBAPAN</span>
                        <div style="font-size:1.25rem; font-weight:800; color:#174C3A; margin-top:2px;">{last_bmkg['RH_AVG']:.0f} %</div>
                    </div>
                    <div style="background:#F0F6F2; padding:12px 14px; border-radius:12px; border:1px solid #A8C3AD;">
                        <span style="font-size:0.8rem; color:#2D725A; font-weight:800; letter-spacing:0.04em; text-transform:uppercase;">CURAH HUJAN</span>
                        <div style="font-size:1.25rem; font-weight:800; color:#174C3A; margin-top:2px;">{last_bmkg['RR']:.1f} mm</div>
                    </div>
                    <div style="background:#F0F6F2; padding:12px 14px; border-radius:12px; border:1px solid #A8C3AD;">
                        <span style="font-size:0.8rem; color:#2D725A; font-weight:800; letter-spacing:0.04em; text-transform:uppercase;">PENYINARAN</span>
                        <div style="font-size:1.25rem; font-weight:800; color:#174C3A; margin-top:2px;">{last_bmkg['SS']:.1f} Jam</div>
                    </div>
                    <div class="benchmark-item-fifth" style="background:#F0F6F2; padding:12px 14px; border-radius:12px; border:1px solid #A8C3AD;">
                        <span style="font-size:0.8rem; color:#2D725A; font-weight:800; letter-spacing:0.04em; text-transform:uppercase;">TEKANAN UDARA</span>
                        <div style="font-size:1.25rem; font-weight:800; color:#174C3A; margin-top:2px;">{last_bmkg['PRESSURE']:.1f} hPa</div>
                    </div>
                </div>
            </div>
            """)

        # -----------------------------------------------------------
        # TAB 2: PREDIKSI MANUAL (INPUT MANDIRI)
        # -----------------------------------------------------------
        with tab_manual:
            st.caption(
                "Mode **Prediksi Manual** memungkinkan Anda memasukkan angka parameter cuaca secara mandiri "
                "untuk memperkirakan suhu esok hari (H+1) menggunakan model Random Forest yang sama."
            )

            with st.form("form_prediksi_manual"):
                col_tgl, col_note = st.columns([1, 2])
                with col_tgl:
                    tgl_manual = st.date_input(
                        "Tanggal Pengamatan (Hari X)",
                        value=datetime.now().date(),
                        max_value=datetime.now().date(),
                        help="Tanggal terjadinya kondisi cuaca yang dimasukkan. Model akan memproyeksikan suhu pada hari X+1."
                    )
                with col_note:
                    render_html("""
                    <div class="form-note-box">
                        <b>Petunjuk:</b> Masukkan kondisi cuaca pada Hari X. Nilai awal di bawah merupakan angka contoh panduan dan dapat Anda ubah sesuai kebutuhan.
                    </div>
                    """)

                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    in_rh = st.number_input(
                        "Kelembapan Rata-Rata (RH) [%]",
                        min_value=0.0, max_value=100.0, value=80.00, step=0.1, format="%.2f",
                        help="Kelembapan udara relatif rata-rata harian"
                    )
                    in_rr = st.number_input(
                        "Curah Hujan (RR) [mm]",
                        min_value=0.0, value=0.00, step=0.1, format="%.2f",
                        help="Akumulasi curah hujan harian"
                    )

                with m_col2:
                    in_ss = st.number_input(
                        "Penyinaran Matahari (SS) [Jam]",
                        min_value=0.0, max_value=24.0, value=6.00, step=0.1, format="%.2f",
                        help="Durasi lama penyinaran sinar matahari harian"
                    )
                    in_ff = st.number_input(
                        "Kecepatan Angin (FF) [m/s]",
                        min_value=0.0, value=2.00, step=0.1, format="%.2f",
                        help="Kecepatan angin rata-rata harian"
                    )

                with m_col3:
                    in_pres = st.number_input(
                        "Tekanan Udara [hPa / mb]",
                        min_value=800.0, max_value=1100.0, value=1010.00, step=0.1, format="%.2f",
                        help="Tekanan udara permukaan rata-rata harian"
                    )
                    in_tavg = st.number_input(
                        "Suhu Rata-Rata Hari Itu (TAVG) [°C]",
                        value=27.50, step=0.1, format="%.2f",
                        help="Temperatur udara rata-rata pada tanggal pengamatan"
                    )

                submit_manual = st.form_submit_button("Hitung Prediksi Suhu (H+1)", use_container_width=True)

            # Simpan nilai input ke variabel lokal agar tetap tersedia setelah form ditutup
            _tgl_manual = tgl_manual
            _in_rh = in_rh
            _in_rr = in_rr
            _in_pres = in_pres
            _in_ss = in_ss
            _in_ff = in_ff
            _in_tavg = in_tavg

            if submit_manual:
                # Eksekusi kalkulasi prediksi manual dengan model asli (hanya saat tombol ditekan)
                try:
                    tgl_x_dt = datetime.combine(_tgl_manual, datetime.min.time())
                    tgl_besok_m, pred_suhu_m = predict_manual_input(
                        model=model,
                        rh=_in_rh,
                        rr=_in_rr,
                        pressure=_in_pres,
                        ss=_in_ss,
                        ff=_in_ff,
                        tavg=_in_tavg,
                        tanggal_x=tgl_x_dt
                    )

                    # Evaluasi selisih suhu prediksi manual vs suhu input hari X
                    diff_manual = pred_suhu_m - _in_tavg
                    if diff_manual > 0.15:
                        trend_m_class = "trend-warmer"
                        trend_m_icon = "▲"
                        trend_m_label = f"+{diff_manual:.2f} °C (Cenderung lebih hangat dari suhu input)"
                    elif diff_manual < -0.15:
                        trend_m_class = "trend-cooler"
                        trend_m_icon = "▼"
                        trend_m_label = f"{diff_manual:.2f} °C (Cenderung lebih sejuk dari suhu input)"
                    else:
                        trend_m_class = "trend-equal"
                        trend_m_icon = "■"
                        trend_m_label = "±0.00 °C (Relatif sama dengan suhu input)"

                    # Status kenyamanan suhu
                    if pred_suhu_m < 24.0:
                        kenyamanan_label = "Sejuk"
                    elif pred_suhu_m < 28.5:
                        kenyamanan_label = "Optimal / Nyaman"
                    elif pred_suhu_m < 32.0:
                        kenyamanan_label = "Cenderung Hangat"
                    else:
                        kenyamanan_label = "Sangat Gerah"

                    render_html(f"""
                    <div class="prediction-card" style="border-left-color: #D6AD55; margin-top: 8px;">
                        <div class="pred-header">
                            <div>
                                <div class="pred-badge" style="background:#FAF6EB; color:#B8892E; border-color:#E8D3A2;">
                                    HASIL PREDIKSI MANUAL (H+1) · RANDOM FOREST
                                </div>
                                <div class="pred-title">
                                    Perkiraan Suhu Rata-rata Esok Hari: {format_indo_date(tgl_besok_m)}
                                </div>
                                <div style="font-size:0.8rem; color:#54665C; margin-top:2px;">
                                    Berdasarkan data input tanggal {format_indo_date(tgl_x_dt)} · Indeks Kenyamanan Suhu: <b>{kenyamanan_label}</b>
                                </div>
                            </div>
                            <div class="pred-trend-pill {trend_m_class}">
                                {trend_m_icon} {trend_m_label}
                            </div>
                        </div>
                        <div class="pred-content-row">
                            <div class="pred-temp-box">
                                <span class="pred-temp-val">{pred_suhu_m:.2f}</span>
                                <span class="pred-temp-unit">°C</span>
                            </div>
                            <div class="pred-desc-box">
                                <p style="font-size:0.92rem; color:#29352F; line-height:1.65; margin:0;">
                                    Hasil perkiraan dihitung dari 8 fitur Machine Learning:
                                    kelembapan ({_in_rh:.1f}%), curah hujan ({_in_rr:.1f} mm),
                                    tekanan ({_in_pres:.1f} hPa), penyinaran ({_in_ss:.1f} jam),
                                    kecepatan angin ({_in_ff:.1f} m/s), penanggalan ({tgl_besok_m.day}/{tgl_besok_m.month}),
                                    dan temperatur acuan ({_in_tavg:.1f} °C). Rata-rata selisih model: <b>±{mae_val:.2f} °C</b>.
                                </p>
                                <div style="margin-top:10px; display:flex; flex-wrap:wrap; gap:6px;">
                                    <span style="background:#F0F6F2; color:#174C3A; border:1px solid #A8C3AD; border-radius:6px; padding:3px 10px; font-size:0.78rem; font-weight:700;">RH: {_in_rh:.1f}%</span>
                                    <span style="background:#F0F6F2; color:#174C3A; border:1px solid #A8C3AD; border-radius:6px; padding:3px 10px; font-size:0.78rem; font-weight:700;">RR: {_in_rr:.1f} mm</span>
                                    <span style="background:#F0F6F2; color:#174C3A; border:1px solid #A8C3AD; border-radius:6px; padding:3px 10px; font-size:0.78rem; font-weight:700;">P: {_in_pres:.1f} hPa</span>
                                    <span style="background:#F0F6F2; color:#174C3A; border:1px solid #A8C3AD; border-radius:6px; padding:3px 10px; font-size:0.78rem; font-weight:700;">SS: {_in_ss:.1f} jam</span>
                                    <span style="background:#F0F6F2; color:#174C3A; border:1px solid #A8C3AD; border-radius:6px; padding:3px 10px; font-size:0.78rem; font-weight:700;">FF: {_in_ff:.1f} m/s</span>
                                    <span style="background:#F0F6F2; color:#174C3A; border:1px solid #A8C3AD; border-radius:6px; padding:3px 10px; font-size:0.78rem; font-weight:700;">TAVG: {_in_tavg:.1f} °C</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    """)

                    # Grafik visualisasi tren dari tanggal input ke tanggal prediksi
                    y_min_val = min(_in_tavg, pred_suhu_m)
                    y_max_val = max(_in_tavg, pred_suhu_m)
                    diff_y = abs(pred_suhu_m - _in_tavg)
                    y_buffer = max(2.2, diff_y * 1.5)
                    y_range = [y_min_val - y_buffer * 0.6, y_max_val + y_buffer * 1.2]

                    fig_manual = go.Figure()
                    fig_manual.add_trace(go.Scatter(
                        x=[tgl_x_dt.strftime("%d %b %Y"), tgl_besok_m.strftime("%d %b %Y")],
                        y=[_in_tavg, pred_suhu_m],
                        mode="lines+markers+text",
                        line=dict(color="#174C3A", width=3.5),
                        marker=dict(size=14, color=["#A8C3AD", "#D6AD55"], line=dict(color="#174C3A", width=2.5)),
                        text=[f"Hari X: {_in_tavg:.2f} °C", f"Prediksi H+1: {pred_suhu_m:.2f} °C"],
                        textposition=["top center", "top center"],
                        textfont=dict(family="Plus Jakarta Sans", size=13, color="#174C3A"),
                        cliponaxis=False,
                    ))
                    fig_manual.update_layout(
                        title=dict(
                            text="Proyeksi Perubahan Suhu: Hari X ke Hari X+1",
                            font=dict(family="Manrope", size=15, color="#174C3A"),
                            x=0.01,
                            y=0.96,
                            xanchor="left",
                            yanchor="top",
                        ),
                        xaxis_title="Timeline Pengamatan & Prakiraan",
                        yaxis_title="Temperatur Udara (°C)",
                        yaxis=dict(
                            range=y_range,
                            showgrid=True,
                            gridcolor="#EBF2ED",
                            zeroline=False,
                        ),
                        xaxis=dict(
                            showgrid=False,
                            range=[-0.35, 1.35],
                        ),
                        template="plotly_white",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=340,
                        margin=dict(t=65, b=35, l=25, r=20),
                    )
                    st.plotly_chart(fig_manual, use_container_width=True)

                except Exception as e_manual:
                    st.error(f"⚠️ Gagal menghitung prediksi: {e_manual}. Periksa kembali nilai input yang dimasukkan.")

            else:
                # Tampilan panduan sebelum tombol ditekan (bukan placeholder kosong)
                render_html("""
                <div style="background:#FFFFFF; border:2px dashed #A8C3AD; border-radius:14px; box-shadow:0 4px 16px rgba(23,76,58,0.04);
                            padding:22px 24px; text-align:center; margin-top:8px; color:#4A6E55;">
                    <div style="font-size:1.5rem; margin-bottom:6px;">🌡️</div>
                    <div style="font-weight:700; font-size:0.95rem; color:#174C3A; margin-bottom:4px;">
                        Siap Menghitung Perkiraan Suhu H+1
                    </div>
                    <div style="font-size:0.83rem; color:#54665C; line-height:1.6;">
                        Isi parameter cuaca pada formulir di atas, lalu tekan tombol
                        <b style="color:#174C3A;">⚡ Hitung Prediksi Suhu (H+1)</b>.<br>
                        Nilai awal merupakan <b>angka contoh panduan</b> — bukan data aktual. Silakan ubah sesuai kebutuhan Anda.
                    </div>
                </div>
                """)

        st.caption(
            "*Keterangan:* Seluruh perkiraan suhu esok hari (H+1) dihitung menggunakan model Random Forest "
            "berdasarkan data pengamatan cuaca kawasan Temon, Kulon Progo untuk membantu pemantauan kondisi cuaca petani cabai."
        )


    except Exception as e:
        st.error(f"Terjadi kendala saat memuat data Beranda: {e}")


# =========================================================
# HALAMAN 2: TENTANG (DOKUMENTASI, PROFIL, DATASET, VIZ, MODEL)
# =========================================================
else:
    # HEADER TENTANG: KONSISTEN SECARA VISUAL DENGAN JUDUL SESUAI HALAMAN
    render_page_header(
        title="Tentang CuacaTani Cabai",
        subtitle="Dokumentasi Sistem Pemantauan Cuaca, Sumber Data, Metodologi Prediksi Suhu, dan Evaluasi Model",
        badge_text="Dokumentasi Sistem & Sumber Data",
    )

    # KOTAK INFORMASI LOKASI & SISTEM
    c_info1, c_info2 = st.columns(2)
    with c_info1:
        render_html("""
        <div class="param-card info-card-full" style="background:#FFFFFF; border:1px solid #D6E3D9; border-top:4px solid #174C3A; margin-bottom:16px;">
            <span class="param-label" style="color:#174C3A;">TUJUAN & WILAYAH SISTEM</span>
            <div style="font-size:0.95rem; color:#29352F; font-weight:800; margin-top:8px;">
                Pemantauan Cuaca untuk Petani Cabai
            </div>
            <div style="font-size:0.84rem; color:#3B4B43; line-height:1.6; margin-top:4px;">
                Membantu petani cabai di Kawasan Temon, Kulon Progo, D.I. Yogyakarta memantau kondisi cuaca harian dan perkiraan suhu esok hari.<br>
                <b>Titik Koordinat:</b> -7.90746° LS, 110.05448° BT · <b>Elevasi:</b> 11 mdpl
            </div>
        </div>
        """)
    with c_info2:
        render_html("""
        <div class="param-card info-card-full" style="background:#FFFFFF; border:1px solid #D6E3D9; border-top:4px solid #D6AD55; margin-bottom:16px;">
            <span class="param-label" style="color:#B8892E;">SISTEM PREDIKSI SUHU</span>
            <div style="font-size:0.95rem; color:#29352F; font-weight:800; margin-top:8px;">
                Machine Learning Forecasting H+1
            </div>
            <div style="font-size:0.84rem; color:#3B4B43; line-height:1.6; margin-top:4px;">
                Model Aktif: <b>Random Forest Regressor (300 estimators)</b><br>
                <b>Metrik Akurasi:</b> R² = 0.7202 · MAE = ±0.404 °C pada data pengujian
            </div>
        </div>
        """)

    # 4 TAB LENGKAP: DATASET, VISUALISASI, PERBANDINGAN MODEL, METODE & SUMBER DATA
    tab_dataset, tab_viz, tab_model, tab_method = st.tabs([
        "Dataset Pengamatan",
        "Visualisasi Historis",
        "Perbandingan Model",
        "Sumber Data & Metodologi",
    ])

    # ---------------- TAB 1: DATASET ----------------
    with tab_dataset:
        st.markdown("### Dataset Pengamatan Cuaca Historis Kawasan Temon")
        st.write(
            "Tabel di bawah menampilkan arsip data historis cuaca harian yang diunduh dari pengamatan BMKG di sekitar Temon. "
            "Data ini digunakan sebagai dasar pengolahan data serta pelatihan dan pengujian model prediksi suhu."
        )

        try:
            df = load_dataset()
            k1, k2, k3 = st.columns(3)
            with k1:
                render_html(f"""
                <div class="param-card" style="background:#FFFFFF; border:1px solid #D6E3D9; border-top:4px solid #174C3A;">
                    <span class="param-label" style="color:#174C3A;">Jumlah Data</span>
                    <div class="param-value" style="color:#174C3A;">{len(df):,}</div>
                    <div class="param-sub">Baris pengamatan harian</div>
                </div>
                """)
            with k2:
                render_html(f"""
                <div class="param-card" style="background:#FFFFFF; border:1px solid #D6E3D9; border-top:4px solid #A8C3AD;">
                    <span class="param-label" style="color:#2D725A;">Jumlah Parameter</span>
                    <div class="param-value" style="color:#174C3A;">{len(df.columns)}</div>
                    <div class="param-sub">Kolom variabel iklim</div>
                </div>
                """)
            with k3:
                rentang_awal = df["TANGGAL"].min().strftime("%d %b %Y")
                rentang_akhir = df["TANGGAL"].max().strftime("%d %b %Y")
                render_html(f"""
                <div class="param-card" style="background:#FFFFFF; border:1px solid #D6E3D9; border-top:4px solid #D6AD55;">
                    <span class="param-label" style="color:#B8892E;">Rentang Waktu</span>
                    <div class="param-value" style="font-size:1.25rem; color:#B8892E;">{rentang_awal} – {rentang_akhir}</div>
                    <div class="param-sub">Data historis pengamatan lokal</div>
                </div>
                """)

            st.markdown("<br>", unsafe_allow_html=True)
            df_display = df.copy()
            df_display["TANGGAL"] = df_display["TANGGAL"].dt.strftime("%Y-%m-%d")
            st.dataframe(df_display, use_container_width=True, hide_index=True)

            st.download_button(
                label="Unduh Dataset Lengkap (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="dataset_cuaca_historis_temon.csv",
                mime="text/csv",
            )
        except Exception as err:
            st.error(f"Gagal memuat dataset: {err}")

    # ---------------- TAB 2: VISUALISASI DATA ----------------
    with tab_viz:
        st.markdown("### Visualisasi & Analisis Tren Historis")
        st.write("Eksplorasi grafik tren parameter cuaca harian dan korelasi antar-variabel iklim pada dataset historis.")

        try:
            df = load_dataset()
            numeric_cols = [c for c in ["TAVG", "RH_AVG", "RR", "SS", "FF_AVG", "PRESSURE"] if c in df.columns]
            selected_params = st.multiselect(
                "Pilih Parameter Cuaca untuk Ditampilkan:",
                numeric_cols,
                default=["TAVG", "RH_AVG"] if "RH_AVG" in numeric_cols else numeric_cols[:1],
            )

            if selected_params:
                fig_trend = px.line(
                    df,
                    x="TANGGAL",
                    y=selected_params,
                    labels={"value": "Nilai Parameter", "TANGGAL": "Tanggal Pengamatan", "variable": "Parameter Cuaca"},
                    color_discrete_sequence=["#174C3A", "#D6AD55", "#2D725A", "#8D6B28", "#5B8266", "#3D5A4C"],
                )
                fig_trend.update_layout(
                    title=dict(
                        text="Grafik Tren Historis Parameter Iklim Harian Kawasan Temon, Kulon Progo",
                        font=dict(family="Manrope", size=14, color="#174C3A")
                    ),
                    template="plotly_white",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=390,
                    margin=dict(l=10, r=10, t=40, b=10),
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(showgrid=True, gridcolor="#EAF3E8"),
                    yaxis=dict(showgrid=True, gridcolor="#EAF3E8"),
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("Pilih minimal satu parameter pada menu di atas untuk menampilkan grafik garis.")

            st.markdown("#### Matriks Korelasi Antar-Parameter Iklim")
            st.caption("Nilai mendekati +1 atau -1 mengindikasikan hubungan linier yang kuat antar-dua parameter.")
            corr = df[numeric_cols].corr().round(2)
            fig_corr = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale=["#FFFFFF", "#D5E4D8", "#A8C3AD", "#2D725A", "#174C3A"],
            )
            fig_corr.update_layout(
                title=dict(
                    text="Matriks Korelasi Pearson Antar-Parameter Iklim Harian",
                    font=dict(family="Manrope", size=14, color="#174C3A")
                ),
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=430,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        except Exception as err:
            st.error(f"Gagal menampilkan visualisasi data: {err}")

    # ---------------- TAB 3: PERBANDINGAN MODEL ----------------
    with tab_model:
        st.markdown("### Evaluasi & Komparasi Model Machine Learning")
        st.write("Perbandingan metrik performa algoritma regresi yang diuji untuk estimasi suhu udara harian di Kulon Progo.")

        try:
            metrics = load_metrics()
            best_model_row = metrics.loc[metrics["R² Score"].idxmax()]

            m1, m2, m3 = st.columns(3)
            with m1:
                render_html(f"""
                <div class="param-card" style="background:#FFFFFF; border:1px solid #D6E3D9; border-top:4px solid #D6AD55;">
                    <span class="param-label" style="color:#B8892E;">Model Terbaik</span>
                    <div class="param-value" style="font-size:1.45rem; color:#B8892E;">{best_model_row['Model Algorithm']}</div>
                    <div class="param-sub">Akurasi R² tertinggi</div>
                </div>
                """)
            with m2:
                render_html(f"""
                <div class="param-card" style="background:#FFFFFF; border:1px solid #D6E3D9; border-top:4px solid #174C3A;">
                    <span class="param-label" style="color:#174C3A;">Koefisien Determinasi (R²)</span>
                    <div class="param-value" style="color:#174C3A;">{best_model_row['R² Score']:.4f}</div>
                    <div class="param-sub">Kemampuan prediksi variansi</div>
                </div>
                """)
            with m3:
                render_html(f"""
                <div class="param-card" style="background:#FFFFFF; border:1px solid #D6E3D9; border-top:4px solid #A8C3AD;">
                    <span class="param-label" style="color:#2D725A;">Rata-rata Selisih (MAE)</span>
                    <div class="param-value" style="color:#174C3A;">{best_model_row['MAE (°C)']:.3f} <span style="font-size:0.95rem;">°C</span></div>
                    <div class="param-sub">Mean Absolute Error</div>
                </div>
                """)

            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(metrics, use_container_width=True, hide_index=True)

            # Bar Chart Perbandingan Metrik
            metrik_dipilih = st.selectbox("Pilih Metrik untuk Grafik Perbandingan:", ["R² Score", "MAE (°C)", "RMSE (°C)"])
            metrics_sorted = metrics.sort_values(by=metrik_dipilih, ascending=(metrik_dipilih != "R² Score"))

            fig_bar = px.bar(
                metrics_sorted,
                x="Model Algorithm",
                y=metrik_dipilih,
                color="Model Algorithm",
                color_discrete_sequence=["#174C3A", "#A8C3AD", "#D6AD55"],
                text_auto=".4f" if metrik_dipilih == "R² Score" else ".3f",
            )
            fig_bar.update_layout(
                title=dict(
                    text=f"Komparasi Nilai {metrik_dipilih} Antar-Algoritma Machine Learning",
                    font=dict(family="Manrope", size=14, color="#174C3A")
                ),
                xaxis_title="Algoritma Model",
                yaxis_title=f"Nilai {metrik_dipilih}",
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                height=360,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#EAF3E8"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Grafik Aktual vs Prediksi
            st.markdown("#### Validasi Garis Aktual vs Prediksi Model")
            pred_df = load_predictions()
            pilihan_algoritma = st.selectbox(
                "Pilih Algoritma Model:",
                ["Linear Regression", "Decision Tree", "Random Forest"],
                index=2,
            )

            if pilihan_algoritma in pred_df.columns and "Aktual (TAVG)" in pred_df.columns:
                fig_val = go.Figure()
                fig_val.add_trace(go.Scatter(
                    x=pred_df["TANGGAL"],
                    y=pred_df["Aktual (TAVG)"],
                    mode="lines",
                    name="Data Aktual Observasi",
                    line=dict(color="#174C3A", width=2.8),
                ))
                fig_val.add_trace(go.Scatter(
                    x=pred_df["TANGGAL"],
                    y=pred_df[pilihan_algoritma],
                    mode="lines",
                    name=f"Hasil Prediksi ({pilihan_algoritma})",
                    line=dict(color="#D6AD55", width=2.6, dash="dash"),
                ))
                fig_val.update_layout(
                    title=dict(
                        text=f"Validasi Garis Waktu: Data Aktual Observasi vs Prediksi ({pilihan_algoritma})",
                        font=dict(family="Manrope", size=14, color="#174C3A")
                    ),
                    template="plotly_white",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=390,
                    xaxis_title="Tanggal Observasi",
                    yaxis_title="Suhu Udara (°C)",
                    hovermode="x unified",
                    margin=dict(l=10, r=10, t=40, b=10),
                    xaxis=dict(showgrid=True, gridcolor="#EAF3E8"),
                    yaxis=dict(showgrid=True, gridcolor="#EAF3E8"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_val, use_container_width=True)
        except Exception as err:
            st.error(f"Gagal memuat perbandingan model: {err}")

    # ---------------- TAB 4: SUMBER DATA & METODOLOGI ----------------
    with tab_method:
        render_html("""
        <div class="param-card" style="background:#FFFFFF; border:1px solid #D6E3D9; border-left:5px solid #174C3A; margin-bottom:20px;">
            <h3 style="color:#174C3A; margin-top:0;">1. Tujuan & Fungsi Aplikasi CuacaTani Cabai</h3>
            <p style="font-size:0.92rem; line-height:1.7; color:#29352F;">
                <b>CuacaTani Cabai</b> dirancang sebagai sistem informasi pemantauan cuaca yang ditujukan khusus bagi <b>petani cabai</b> 
                di kawasan Temon, Kabupaten Kulon Progo, D.I. Yogyakarta. Tanaman cabai sangat sensitif terhadap perubahan suhu udara, 
                kelembapan tinggi, dan curah hujan ekstrem. Aplikasi ini hadir untuk membantu petani:
            </p>
            <ul style="font-size:0.9rem; line-height:1.8; color:#29352F; padding-left:20px; margin:0 0 10px 0;">
                <li><b>Memantau cuaca aktual hari ini:</b> Memperoleh informasi langsung tentang suhu, kelembapan, curah hujan, angin, dan tekanan udara.</li>
                <li><b>Mengetahui perkiraan suhu esok hari (H+1):</b> Mendapatkan gambaran awal apakah suhu esok hari cenderung lebih panas, sejuk, atau stabil guna mengantisipasi kondisi lahan.</li>
            </ul>
            <div style="background:#FAF6EB; border:1px solid #E8D3A2; padding:12px 16px; border-radius:10px; font-size:0.88rem; color:#8D6B28; line-height:1.55;">
                ⚠️ <b>Keterangan Status Aplikasi:</b> CuacaTani Cabai adalah sistem informasi pendukung dan <b>bukan aplikasi resmi BMKG</b>.
            </div>
        </div>

        <div class="param-card" style="background:#FFFFFF; border:1px solid #D6E3D9; border-left:5px solid #2D725A; margin-bottom:20px;">
            <h3 style="color:#174C3A; margin-top:0;">2. Sumber Data & Perbedaan Fungsinya</h3>
            <p style="font-size:0.92rem; line-height:1.7; color:#29352F;">
                Aplikasi ini memanfaatkan <b>dua sumber data yang berbeda</b> dengan fungsi masing-masing agar sistem berjalan optimal:
            </p>
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap:14px; margin-top:12px;">
                <div style="background:#F0F6F2; border:1px solid #A8C3AD; border-radius:12px; padding:16px 16px;">
                    <div style="font-size:0.82rem; font-weight:800; color:#174C3A; letter-spacing:0.06em; text-transform:uppercase;">
                        SUMBER 1: DATA HISTORIS BMKG
                    </div>
                    <div style="font-size:0.94rem; font-weight:700; color:#174C3A; margin-top:4px;">
                        Dataset Pembelajaran Model (Offline)
                    </div>
                    <div style="font-size:0.88rem; color:#29352F; line-height:1.65; margin-top:8px;">
                        • <b>Asal Data:</b> Data rekaman pengamatan stasiun cuaca resmi BMKG di sekitar Temon.<br>
                        • <b>Jumlah:</b> 583 baris data harian (15 Juli 2024 – 12 Juli 2026).<br>
                        • <b>Fungsi:</b> Khusus digunakan untuk <b>pengolahan data (preprocessing), analisis korelasi, serta melatih dan menguji model Machine Learning</b>.
                    </div>
                </div>
                <div style="background:#FAF6EB; border:1px solid #E8D3A2; border-radius:12px; padding:16px 16px;">
                    <div style="font-size:0.82rem; font-weight:800; color:#8D6B28; letter-spacing:0.06em; text-transform:uppercase;">
                        SUMBER 2: OPEN-METEO API
                    </div>
                    <div style="font-size:0.94rem; font-weight:700; color:#8D6B28; margin-top:4px;">
                        Data Cuaca Berjalan Terkini (Online)
                    </div>
                    <div style="font-size:0.88rem; color:#29352F; line-height:1.65; margin-top:8px;">
                        • <b>Asal Data:</b> Layanan Open-Meteo Weather API pada titik koordinat Temon (-7.90746, 110.05448).<br>
                        • <b>Sifat:</b> Data cuaca aktual yang diperbarui otomatis saat aplikasi dibuka.<br>
                        • <b>Fungsi:</b> Menyediakan <b>tampilan cuaca hari ini</b> serta menjadi <b>input otomatis</b> untuk menghitung perkiraan suhu esok hari (H+1).
                    </div>
                </div>
            </div>
            <div style="background:#F0F6F2; border-left:4px solid #174C3A; padding:12px 16px; border-radius:8px; margin-top:14px; font-size:0.88rem; color:#174C3A; line-height:1.6;">
                <b>Penegasan:</b> Data historis BMKG berfungsi sebagai <i>otak/fondasi pelatihan model</i>, sedangkan Open-Meteo API berfungsi sebagai <i>penyedia data cuaca berjalan</i>. Pemisahan ini memastikan sistem tetap dapat melakukan prediksi suhu esok hari secara mandiri.
            </div>
        </div>

        <div class="param-card" style="background:#FFFFFF; border:1px solid #D6E3D9; border-left:5px solid #D6AD55; margin-bottom:20px;">
            <h3 style="color:#174C3A; margin-top:0;">3. Proses Pengolahan Data & Model Random Forest (H+1)</h3>
            <p style="font-size:0.92rem; line-height:1.7; color:#29352F;">
                Prediksi suhu esok hari disusun melalui tahapan pengolahan data terstruktur:
            </p>
            <ol style="font-size:0.92rem; line-height:1.8; color:#29352F; padding-left:20px; margin:0 0 10px 0;">
                <li><b>Pembersihan Data (Data Cleaning):</b> Memeriksa kelengkapan data historis, menangani nilai anomali, dan merapikan format tanggal.</li>
                <li><b>Rekayasa Fitur Lag-1 (Time Lag):</b> Membentuk fitur penundaan waktu di mana data cuaca hari ini (Hari H) digunakan sebagai masukan untuk memprediksi suhu rata-rata hari berikutnya (Hari H+1).</li>
                <li><b>Pembagian Data:</b> Data dibagi menjadi data latih (training) dan data uji (testing) untuk mengukur performa prediksi secara objektif.</li>
                <li><b>Pelatihan Model Random Forest:</b> Menggabungkan 300 pohon keputusan (decision trees) untuk memprediksi suhu dengan akurasi tinggi (R² = 0.7202, MAE = ±0.404 °C).</li>
            </ol>
        </div>

        <div class="param-card" style="background:#FFFFFF; border:1px solid #D6E3D9; border-left:5px solid #A8C3AD;">
            <h3 style="color:#174C3A; margin-top:0;">4. Kamus Variabel Cuaca</h3>
            <ul style="font-size:0.92rem; line-height:1.8; color:#29352F; padding-left:20px; margin:0;">
                <li><b>TAVG (°C):</b> Suhu rata-rata harian udara permukaan (variabel yang diproyeksikan untuk H+1).</li>
                <li><b>RH_AVG (%):</b> Kelembapan udara relatif rata-rata harian (kandungan uap air).</li>
                <li><b>RR (mm):</b> Jumlah akumulasi curah hujan harian.</li>
                <li><b>PRESSURE (hPa):</b> Tekanan udara permukaan rata-rata harian.</li>
                <li><b>SS (Jam):</b> Durasi lama penyinaran sinar matahari harian.</li>
                <li><b>FF_AVG (m/s):</b> Kecepatan angin rata-rata harian.</li>
                <li><b>MONTH & DAY:</b> Fitur bulan dan tanggal untuk mengenali pola pergantian musim hujan dan kemarau di Kulon Progo.</li>
            </ul>
        </div>
        """)
