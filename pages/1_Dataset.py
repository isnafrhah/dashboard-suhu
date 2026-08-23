import glob
import os
import pandas as pd
import streamlit as st
from style import apply_custom_css, draw_metric_card, render_header

# Config Halaman
st.set_page_config(page_title="Dataset - BMKG", page_icon="📊", layout="wide")
apply_custom_css()

# ======================================================
# INLINE CSS UNTUK LABEL SEARCH & SLIDER AGAR JELAS/TAJAM
# ======================================================
st.markdown(
    """
    <style>
    div[data-testid="stTextInput"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stWidgetLabel"] p,
    .stTextInput label,
    .stSlider label {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        opacity: 1 !important;
    }
    div[data-testid="stTextInput"] input {
        color: #0F172A !important;
        background-color: #FFFFFF !important;
    }
    div[data-testid="stTextInput"] input::placeholder {
        color: #94A3B8 !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Header Halaman
render_header(
    title="Dataset Suhu Udara & Parameter Iklim BMKG YIA",
    subtitle="Eksplorasi data historis parameter iklim harian yang digunakan untuk pemodelan prediksi temperatur.",
)


@st.cache_data
def load_data():
    # Nama file hasil cleaning dari notebook (sinkron dengan halaman Visualisasi Data)
    file_path = "data_bmkg_fix.csv"
        
    combined_df = None
    try:
        if os.path.exists(file_path):
            if file_path.endswith('.csv'):
                combined_df = pd.read_csv(file_path)
            else:
                combined_df = pd.read_excel(file_path)
        else:
            # Cari otomatis jika file spesifik di atas tidak ada
            excel_files = glob.glob("*.xlsx") + glob.glob("**/*.xlsx", recursive=True)
            csv_files = [f for f in glob.glob("*.csv") + glob.glob("**/*.csv", recursive=True) if "filtered" not in f and "full" not in f]
            
            if excel_files:
                combined_df = pd.read_excel(excel_files[0])
            elif csv_files:
                combined_df = pd.read_csv(csv_files[0])
    except Exception:
        combined_df = None

    # Jika file berhasil dimuat, bersihkan duplikat & format tanggal
    if combined_df is not None and not combined_df.empty:
        combined_df = combined_df.drop_duplicates()

        # Deteksi kolom Tanggal secara otomatis
        date_col = next(
            (c for c in combined_df.columns if "tanggal" in c.lower() or "date" in c.lower()),
            None,
        )
        if date_col:
            combined_df.rename(columns={date_col: "TANGGAL"}, inplace=True)
            combined_df["TANGGAL"] = pd.to_datetime(combined_df["TANGGAL"], errors="coerce")
            combined_df = combined_df.dropna(subset=["TANGGAL"])
            combined_df = combined_df.sort_values(by="TANGGAL", ascending=True)
            combined_df = combined_df.reset_index(drop=True)

        return combined_df

    # Fallback Data Dummy jika file tidak ditemukan sama sekali
    dates = pd.date_range(start="2024-07-14", end="2026-07-22", freq="D")
    return pd.DataFrame(
        {
            "TANGGAL": dates,
            "RH_AVG": [80.0 + i % 5 for i in range(len(dates))],
            "RR": [0.0 + (i % 10) * 2 for i in range(len(dates))],
            "SS": [6.0 + (i % 4) * 0.5 for i in range(len(dates))],
            "FF_AVG": [2.5 + (i % 3) * 0.2 for i in range(len(dates))],
            "TAVG": [26.0 + (i % 5) * 0.4 for i in range(len(dates))],
        }
    )


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
    st.error("⚠️ Dataset gagal dimuat atau kosong.")
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
        "Total Baris Data", f"{len(df)} Records", "Pengamatan Harian"
    )
with col2:
    draw_metric_card(
        "Total Kolom", f"{len(df.columns)} Parameter", "Indikator Iklim"
    )
with col3:
    draw_metric_card("Rentang Waktu", f"{min_date}", f"s/d {max_date}")

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------
# 2. FILTER DATASET
# ------------------------------------------------------
st.markdown(
    '<div class="section-header">2. Filter & Tampilkan Data</div>',
    unsafe_allow_html=True,
)

c1, c2 = st.columns([2, 1])
with c1:
    search_keyword = st.text_input(
        "🔎 Cari Tanggal atau Nilai Data (misal: 2024-07-14 atau 2025-04):",
        placeholder="Ketik tanggal atau kata kunci...",
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
    '<div class="section-header">3. Tabel Preview Dataset</div>',
    unsafe_allow_html=True,
)

st.dataframe(filtered_df.head(row_count), use_container_width=True)

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
        label=f"Unduh Data Tampilan ({min(row_count, len(filtered_df))} Baris)",
        data=csv_filtered,
        file_name="dataset_bmkg_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_dn2:
    csv_full = df_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Unduh Seluruh Dataset ({len(df)} Baris)",
        data=csv_full,
        file_name="dataset_bmkg_full.csv",
        mime="text/csv",
        use_container_width=True,
    )