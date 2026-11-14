"""
Hitung bias Open-Meteo terhadap data historis BMKG pada koordinat Stasiun
Meteorologi Kelas II Yogyakarta (YIA).

Jalankan dari folder project:
    python hitung_kalibrasi_openmeteo.py

File hasil hanya akan dipakai dashboard jika metadata koordinatnya cocok.
"""
import json
import requests
import pandas as pd

LATITUDE = -7.90746
LONGITUDE = 110.05448

df_bmkg = pd.read_csv("data_bmkg_fix.csv", parse_dates=["TANGGAL"])
df_bmkg = df_bmkg.dropna(subset=["TANGGAL"]).sort_values("TANGGAL")
start_date = df_bmkg["TANGGAL"].min().strftime("%Y-%m-%d")
end_date = df_bmkg["TANGGAL"].max().strftime("%Y-%m-%d")

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": start_date,
    "end_date": end_date,
    "daily": "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum,"
             "wind_speed_10m_mean,sunshine_duration",
    "hourly": "surface_pressure",
    "timezone": "Asia/Jakarta",
    "wind_speed_unit": "ms",
}
response = requests.get(url, params=params, timeout=60)
response.raise_for_status()
payload = response.json()
daily = payload["daily"]

df_om = pd.DataFrame({
    "TANGGAL": pd.to_datetime(daily["time"]),
    "TAVG_OM": daily["temperature_2m_mean"],
    "RH_AVG_OM": daily["relative_humidity_2m_mean"],
    "RR_OM": [v or 0.0 for v in daily["precipitation_sum"]],
    "FF_AVG_OM": daily["wind_speed_10m_mean"],
    "SS_OM": [(v or 0.0) / 3600 for v in daily["sunshine_duration"]],
})
hourly = pd.DataFrame({
    "datetime": pd.to_datetime(payload["hourly"]["time"]),
    "PRESSURE_OM": payload["hourly"]["surface_pressure"],
})
hourly["TANGGAL"] = hourly["datetime"].dt.normalize()
pressure_daily = hourly.groupby("TANGGAL")["PRESSURE_OM"].mean().reset_index()
df_om = df_om.merge(pressure_daily, on="TANGGAL", how="left")

joined = df_bmkg.merge(df_om, on="TANGGAL", how="inner")
if joined.empty:
    raise RuntimeError("Tidak ada tanggal yang cocok antara dataset BMKG dan Open-Meteo.")

bias = {
    "TAVG": float((joined["TAVG_OM"] - joined["TAVG"]).mean()),
    "RH_AVG": float((joined["RH_AVG_OM"] - joined["RH_AVG"]).mean()),
    "RR": float((joined["RR_OM"] - joined["RR"]).mean()),
    "PRESSURE": float((joined["PRESSURE_OM"] - joined["PRESSURE"]).mean()),
    "SS": float((joined["SS_OM"] - joined["SS"]).mean()),
    "FF_AVG": float((joined["FF_AVG_OM"] - joined["FF_AVG"]).mean()),
}
result = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "matched_days": int(len(joined)),
    "bias": bias,
}
with open("kalibrasi_openmeteo.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print(f"Kalibrasi selesai untuk {len(joined)} hari.")
print(f"Koordinat: {LATITUDE}, {LONGITUDE}")
for name, value in bias.items():
    print(f"{name:10s}: {value:+.3f}")
print("File kalibrasi_openmeteo.json berhasil diperbarui.")
