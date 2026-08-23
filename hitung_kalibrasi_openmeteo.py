"""
Script ini dijalankan SEKALI SAJA (bukan bagian dashboard), untuk menghitung
selisih rata-rata (bias) antara data Open-Meteo dan data BMKG asli, dari
periode yang datanya sudah kamu punya (data_bmkg_fix.csv).

Cara pakai:
1. Jalankan script ini dari folder utama project (yang ada data_bmkg_fix.csv)
2. Pastikan komputer/laptop ada koneksi internet
3. Hasilnya akan tersimpan sebagai kalibrasi_openmeteo.json
4. File JSON itu dipakai otomatis oleh 4_Prediksi.py (lihat bagian bawah pesan ini)

Logikanya: ambil data Open-Meteo utk tanggal yang SAMA dengan data BMKG kamu,
lalu hitung rata-rata (Open-Meteo - BMKG asli) untuk tiap parameter. Kalau
misal Open-Meteo rata-rata melaporkan kelembapan 3% lebih tinggi dari BMKG,
maka setiap kali dashboard ambil data Open-Meteo, angka itu dikurangi 3%
dulu sebelum masuk ke model -- supaya lebih mendekati kondisi asli YIA.
"""
import requests
import pandas as pd
import json
import time

YIA_LAT = -7.9053
YIA_LON = 110.0573

# Baca data BMKG asli kamu
df_bmkg = pd.read_csv("data_bmkg_fix.csv", parse_dates=["TANGGAL"])
print(f"Data BMKG asli: {len(df_bmkg)} baris, {df_bmkg['TANGGAL'].min().date()} s/d {df_bmkg['TANGGAL'].max().date()}")

start_date = df_bmkg["TANGGAL"].min().strftime("%Y-%m-%d")
end_date = df_bmkg["TANGGAL"].max().strftime("%Y-%m-%d")

# Ambil data Open-Meteo untuk periode & lokasi yang SAMA
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": YIA_LAT,
    "longitude": YIA_LON,
    "start_date": start_date,
    "end_date": end_date,
    "daily": "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum,"
             "wind_speed_10m_mean,sunshine_duration",
    "hourly": "surface_pressure",
    "timezone": "Asia/Jakarta",
    "wind_speed_unit": "ms",
}

print("Mengambil data historis Open-Meteo... (bisa beberapa detik)")
resp = requests.get(url, params=params, timeout=60)
resp.raise_for_status()
data = resp.json()

daily = data["daily"]
df_om = pd.DataFrame({
    "TANGGAL": pd.to_datetime(daily["time"]),
    "TAVG_OM": daily["temperature_2m_mean"],
    "RH_AVG_OM": daily["relative_humidity_2m_mean"],
    "RR_OM": [v or 0.0 for v in daily["precipitation_sum"]],
    "FF_AVG_OM": daily["wind_speed_10m_mean"],
    "SS_OM": [(v or 0) / 3600 for v in daily["sunshine_duration"]],
})

# Rata-ratakan tekanan per jam jadi per hari
df_pressure_hourly = pd.DataFrame({
    "datetime": pd.to_datetime(data["hourly"]["time"]),
    "PRESSURE": data["hourly"]["surface_pressure"]
})
df_pressure_hourly["TANGGAL"] = df_pressure_hourly["datetime"].dt.normalize()
df_pressure_daily = df_pressure_hourly.groupby("TANGGAL")["PRESSURE"].mean().reset_index()
df_pressure_daily.columns = ["TANGGAL", "PRESSURE_OM"]

df_om = df_om.merge(df_pressure_daily, on="TANGGAL", how="left")

# Gabungkan dengan data BMKG asli berdasarkan tanggal yang sama
df_gabung = df_bmkg.merge(df_om, on="TANGGAL", how="inner")
print(f"Jumlah tanggal yang bisa dibandingkan (ada di kedua sumber): {len(df_gabung)}")

# Hitung bias rata-rata: (Open-Meteo - BMKG asli)
bias = {
    "RH_AVG": float((df_gabung["RH_AVG_OM"] - df_gabung["RH_AVG"]).mean()),
    "RR": float((df_gabung["RR_OM"] - df_gabung["RR"]).mean()),
    "PRESSURE": float((df_gabung["PRESSURE_OM"] - df_gabung["PRESSURE"]).mean()),
    "SS": float((df_gabung["SS_OM"] - df_gabung["SS"]).mean()),
    "FF_AVG": float((df_gabung["FF_AVG_OM"] - df_gabung["FF_AVG"]).mean()),
    "TAVG": float((df_gabung["TAVG_OM"] - df_gabung["TAVG"]).mean()),
}

print("\n=== Bias rata-rata (Open-Meteo - BMKG asli) ===")
for k, v in bias.items():
    print(f"  {k:10s}: {v:+.3f}")

with open("kalibrasi_openmeteo.json", "w") as f:
    json.dump(bias, f, indent=2)

print("\n✓ kalibrasi_openmeteo.json berhasil disimpan")
print("Taruh file ini di folder utama project (sejajar dengan data_bmkg_fix.csv)")