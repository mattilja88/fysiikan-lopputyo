import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import streamlit as st
from streamlit_folium import st_folium
from scipy.signal import butter, filtfilt
from math import radians, cos, sin, asin, sqrt

st.title("Fysiikan loppuprojekti")

df_linear = pd.read_csv("./data/linear.csv")
df_gps = pd.read_csv("./data/location.csv")

# Lyhyt perustelu eri akseleille:
st.caption(
    "Käytän eri akseleita, koska puhelimen asento projisoi kävelyliikkeen eri suuntiin: "
    "x-komponentissa askel näkyy aikatason suodatetussa signaalissa selkeinä nollanylityksinä, "
    "kun taas z-komponentissa askelrytmi näkyy tehospektrissä selkeimpänä dominoivana taajuuspiikkinä."
)

# - Askelmäärä suodatetusta kiihtyvyysdatasta (X-akseli)

t_lin = df_linear["Time (s)"]
data_x = df_linear["Linear Acceleration x (m/s^2)"]

T_tot = df_linear["Time (s)"].max()
n = len(df_linear["Time (s)"])
fs = n / T_tot
nyq = fs / 2

order = 3
cutoff = 1 / 0.4  # 2.5 Hz

def butter_lowpass_filter(series, cutoff_hz, nyq_hz, order_n):
    normal_cutoff = cutoff_hz / nyq_hz
    b, a = butter(order_n, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, series)

data_filt = butter_lowpass_filter(data_x, cutoff, nyq, order)

# Nollanylitykset (1 ylitys = 1 askel tämän datan tapauksessa)
steps_filtered = 0.0
for i in range(n - 1):
    if data_filt[i] / data_filt[i + 1] < 0:
        steps_filtered += 1

# Suodatettu kiihtyvyysdata -kuvaaja
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(t_lin, data_filt, label="Suodatettu kiihtyvyys (x)")
ax1.set_xlabel("Aika [s]")
ax1.set_ylabel("Acceleration x (m/s^2)")
ax1.set_title("Suodatettu kiihtyvyysdata (askelmäärän määrittämiseen)")
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)

#  - Askelmäärä Fourier-analyysillä (Z-akseli) – tyyli kuten esimerkissä

signal = df_linear["Linear Acceleration z (m/s^2)"].to_numpy()
t = df_linear["Time (s)"].to_numpy()

N = len(signal)
dt = (t.max() - t.min()) / N

fourier = np.fft.fft(signal, N)
psd = fourier * np.conj(fourier) / N
freq = np.fft.fftfreq(N, dt)
L = np.arange(1, int(N / 2))

fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(freq[L], psd[L].real)
ax2.set_xlabel("Taajuus [Hz] = [1/s]")
ax2.set_ylabel("Teho")
ax2.set_title("Valitun kiihtyvyyskomponentin tehospektritiheys (z)")
ax2.set_xlim(0, 10)
ax2.grid(True)
st.pyplot(fig2)

# etsitään maksimia kävelylle järkevältä alueelta
mask_f = (freq > 0.7) & (freq < 3.0)
f_max = freq[mask_f][psd[mask_f].real == np.max(psd[mask_f].real)][0]
T = 1 / f_max
steps_fft = f_max * np.max(t)


#  - GPS: poista alku (epätarkkuus + paikallaan olo)

acc_limit = 10
speed_threshold = 0.5

mask = (df_gps["Horizontal Accuracy (m)"] < acc_limit) & (df_gps["Velocity (m/s)"] > speed_threshold)
if mask.any():
    i0 = df_gps.index[mask][0]
    df_gps = df_gps.loc[i0:].reset_index(drop=True)
else:
    st.warning("GPS-suodatus: ei löytynyt kohtaa joka täyttää rajat. Käytetään koko GPS-dataa.")

# Keskinopeus
mean_speed_mps = df_gps["Velocity (m/s)"].mean() if "Velocity (m/s)" in df_gps.columns else float("nan")
mean_speed_kmh = mean_speed_mps * 3.6 if not np.isnan(mean_speed_mps) else float("nan")

# Haversine-matka
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

df_gps = df_gps.copy()
df_gps["Distance_calc_km"] = np.zeros(len(df_gps))

for i in range(len(df_gps) - 1):
    lon1 = df_gps["Longitude (°)"][i]
    lon2 = df_gps["Longitude (°)"][i + 1]
    lat1 = df_gps["Latitude (°)"][i]
    lat2 = df_gps["Latitude (°)"][i + 1]
    df_gps.loc[i + 1, "Distance_calc_km"] = haversine(lon1, lat1, lon2, lat2)

df_gps["total_distance_km"] = df_gps["Distance_calc_km"].cumsum()
total_distance_km = df_gps["total_distance_km"].iloc[-1]

# Askelpituudet
step_length_m = (total_distance_km * 1000) / steps_filtered if steps_filtered != 0 else float("nan")
step_length_Fourier = (total_distance_km * 1000) / steps_fft if steps_fft != 0 else float("nan")

# - Tulokset (numerot)

st.subheader("Tulokset (numerot)")

st.write("Askelmäärä (suodatettu data, x):", f"{steps_filtered:.0f}")
st.write("Askelmäärä (Fourier, z):", f"{steps_fft:.0f}")

st.write("Keskinopeus:", f"{mean_speed_kmh:.2f} km/h")
st.write("Kuljettu matka:", f"{total_distance_km:.3f} km")

st.write("Askelpituudet:")
st.write(
    f"{step_length_m:.2f} m (suodatettu), "
    f"{step_length_Fourier:.2f} m (Fourier), "
    f"{((step_length_Fourier + step_length_m) / 2):.2f} m (keskiarvo)"
)

st.write(f"Dominoiva taajuus (z): {f_max:.3f} Hz")
st.write(f"Jaksonaika (askelaika): {T:.3f} s")

# - Reitti kartalla

st.subheader("Reitti kartalla")

lat_center = df_gps["Latitude (°)"].mean()
lon_center = df_gps["Longitude (°)"].mean()

my_map = folium.Map(location=[lat_center, lon_center], zoom_start=15)
folium.PolyLine(df_gps[["Latitude (°)", "Longitude (°)"]], color="red", weight=3).add_to(my_map)

st_folium(my_map, width=900, height=650)
