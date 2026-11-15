import glob
import os
from pathlib import Path
import pandas as pd
import streamlit as st
from style import apply_custom_css, draw_metric_card, render_header

# Config Halaman
st.set_page_config(
    page_title="Dataset Cuaca – CuacaTani Cabai",
    page_icon="📊",
    layout="wide"
)
apply_custom_css()

# Header Halaman
render_header(
    title="Dataset Parameter Cuaca Historis Kawasan Temon",
    subtitle="Eksplorasi arsip data pengamatan iklim harian di Kawasan Temon, Kulon Progo yang digunakan sebagai dasar pengolahan data dan pelatihan model Machine Learning.",
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data_bmkg_fix.csv"


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    if "TANGGAL" not in df.columns:
        raise ValueError("Kolom TANGGAL tidak ditemukan.")

    df = df.drop_duplicates()
    df["TANGGAL"] = pd.to_datetime(df["TANGGAL"], errors="coerce")
    df = df.dropna(subset=["TANGGAL"])
    df = df.sort_values("TANGGAL").reset_index(drop=True)

    return df


# Memuat data ke variabel df
df = load_data()

# Validasi keamanan agar df tidak kosong
if df is not None and not df.empty:
    if "TANGGAL" in df.columns and pd.api.types.is_datetime64_any_dtype(df["TANGGAL"]):
        min_date = df["TANGGAL"].min().strftime("%d %B %Y")
        max_date = df["TANGGAL"].max().strftime("%d %B %Y")
        df_display = df.copy()
        df_display["TANGGAL"] = df_display["TANGGAL"].dt.strftime("%Y-%m-%d")
    else:
        min_date = str(df["TANGGAL"].min()) if "TANGGAL" in df.columns else "-"
        max_date = str(df["TANGGAL"].max()) if "TANGGAL" in df.columns else "-"
        df_display = df.copy()
else:
    st.error("⚠️ Dataset gagal dimuat atau file data_bmkg_fix.csv tidak ditemukan.")
    st.stop()

# ------------------------------------------------------
# 1. INFORMASI DATASET
# ------------------------------------------------------
st.markdown(
    '<div class="section-header">1. Ringkasan Informasi Dataset</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    draw_metric_card(
        "Total Baris Data", f"{len(df):,} Baris", "Pengamatan Harian"
    )
with col2:
    draw_metric_card(
        "Total Parameter", f"{len(df.columns)} Kolom", "Indikator Iklim"
    )
with col3:
    draw_metric_card("Rentang Waktu", f"{min_date}", f"s/d {max_date}")

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------
# 2. FILTER DATASET
# ------------------------------------------------------
st.markdown(
    '<div class="section-header">2. Filter & Pencarian Data</div>',
    unsafe_allow_html=True,
)

c1, c2 = st.columns([2, 1])
with c1:
    search_keyword = st.text_input(
        "🔎 Cari Tanggal atau Nilai Parameter (misal: 2024-07 atau 27.5):",
        placeholder="Ketik tanggal atau angka yang ingin dicari...",
    )
with c2:
    row_count = st.slider(
        "Jumlah Baris Yang Ditampilkan:",
        min_value=5,
        max_value=max(10, len(df)),
        value=10,
    )

# Filter Logic
if search_keyword:
    filtered_df = df_display[
        df_display.astype(str).apply(
            lambda row: row.str.contains(search_keyword, case=False).any(),
            axis=1,
        )
    ]
    filtered_df = filtered_df.reset_index(drop=True)
else:
    filtered_df = df_display

# ------------------------------------------------------
# 3. PREVIEW DATASET
# ------------------------------------------------------
st.markdown(
    '<div class="section-header">3. Tabel Data Cuaca Historis</div>',
    unsafe_allow_html=True,
)

# Keterangan info bahwa tabel dapat digeser ke samping pada layar HP
st.caption("💡 *Tabel data dapat digeser secara horizontal (geser ke samping) pada layar HP atau tablet.*")

st.dataframe(
    filtered_df.head(row_count),
    use_container_width=True,
    hide_index=True
)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------
# 4. OPSI UNDUH DATASET
# ------------------------------------------------------
st.markdown(
    '<div class="section-header">4. Opsi Unduh Dataset</div>',
    unsafe_allow_html=True,
)

col_dn1, col_dn2 = st.columns(2)

with col_dn1:
    csv_filtered = (
        filtered_df.head(row_count).to_csv(index=False).encode("utf-8")
    )
    st.download_button(
        label=f"📥 Unduh Data Tampilan ({min(row_count, len(filtered_df))} Baris)",
        data=csv_filtered,
        file_name="dataset_cuaca_temon_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_dn2:
    csv_full = df_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Unduh Seluruh Dataset ({len(df)} Baris)",
        data=csv_full,
        file_name="dataset_cuaca_temon_full.csv",
        mime="text/csv",
        use_container_width=True,
    )
