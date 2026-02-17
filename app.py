import os
import joblib
import pandas as pd
import numpy as np
import requests

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------- App + CORS ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # demo; restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
session = requests.Session()

# ---------------- Load Model Bundle ----------------
BUNDLE_PATH = os.path.join(BASE_DIR, "aqi_bundle.pkl")
bundle = joblib.load(BUNDLE_PATH)

reg_model = bundle["reg_model"]
clf_model = bundle["clf_model"]
FEATURES = bundle["features"]
PROB_THR = bundle.get("spike_prob_threshold", 0.30)

print("LOADED BUNDLE FROM:", BUNDLE_PATH)
print("FEATURE COUNT:", len(FEATURES))
print("FIRST 5 FEATURES:", FEATURES[:5])

def risk_label(prob: float) -> str:
    if prob >= 0.60:
        return "HIGH"
    if prob >= PROB_THR:
        return "MEDIUM"
    return "LOW"

# ---------------- Load City Dataset ----------------
DATA_PATH = os.path.join(BASE_DIR, "city_day.csv")
aqi_df = pd.read_csv(DATA_PATH)

# Standardize column names
if "PM2.5" in aqi_df.columns and "PM2_5" not in aqi_df.columns:
    aqi_df = aqi_df.rename(columns={"PM2.5": "PM2_5"})

aqi_df["Date"] = pd.to_datetime(aqi_df["Date"])
aqi_df = aqi_df.sort_values(["City", "Date"]).reset_index(drop=True)

# Precompute medians per city for fallback
NUM_COLS = ["PM10", "PM2_5", "NO2", "NO", "SO2", "CO", "O3", "AQI"]
city_medians = (
    aqi_df.groupby("City")[NUM_COLS]
    .median(numeric_only=True)
    .to_dict(orient="index")
)

# ---------------- Weather Helpers ----------------
def geocode_city_openmeteo(city: str, country_code="IN"):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 1, "language": "en", "format": "json", "country_code": country_code}
    r = session.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    if "results" not in js or not js["results"]:
        return None
    best = js["results"][0]
    return float(best["latitude"]), float(best["longitude"])

def fetch_openmeteo_today(lat: float, lon: float, timezone="Asia/Kolkata"):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"],
        "hourly": ["relative_humidity_2m"],
        "timezone": timezone,
    }
    r = session.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()

    daily = js["daily"]
    weather = {
        "temperature_2m_max": float(daily["temperature_2m_max"][0]),
        "temperature_2m_min": float(daily["temperature_2m_min"][0]),
        "precipitation_sum": float(daily["precipitation_sum"][0]),
        "wind_speed_10m_max": float(daily["wind_speed_10m_max"][0]),
    }

    hum = js.get("hourly", {}).get("relative_humidity_2m", None)
    weather["relative_humidity_2m_mean"] = float(np.mean(hum[:24])) if hum else 60.0
    return weather

# ---------------- Feature Builder (City only) ----------------
DEFAULT_CONSTRUCTION_INTENSITY = 0.45

def build_features_for_city(city: str) -> pd.DataFrame:
    if city not in aqi_df["City"].unique():
        raise ValueError(f"City '{city}' not found in dataset")

    df_c = aqi_df[aqi_df["City"] == city].copy().sort_values("Date")
    df_c = df_c.dropna(subset=["AQI"])

    if len(df_c) < 8:
        raise ValueError(f"Not enough AQI history for '{city}' (need >= 8 days)")

    latest = df_c.iloc[-1]
    last8 = df_c.iloc[-8:]
    aqi_vals = last8["AQI"].values

    # Lags
    AQI_lag_1 = float(aqi_vals[-2])
    AQI_lag_3 = float(aqi_vals[-4])
    AQI_lag_7 = float(aqi_vals[-8])

    # Rolling + diffs
    last7 = aqi_vals[-7:]
    AQI_roll7_mean = float(np.mean(last7))
    AQI_roll7_std  = float(np.std(last7, ddof=1))
    AQI_change_1d  = float(aqi_vals[-1] - aqi_vals[-2])
    AQI_change_3d  = float(aqi_vals[-1] - aqi_vals[-4])

    med = city_medians[city]

    def val(col: str) -> float:
        v = latest.get(col, np.nan)
        if pd.isna(v):
            return float(med[col])
        return float(v)

    # Weather for today
    latlon = geocode_city_openmeteo(city)
    if latlon:
        lat, lon = latlon
        weather = fetch_openmeteo_today(lat, lon)
    else:
        # fallback for demo if geocode fails
        weather = {
            "temperature_2m_max": 32.0,
            "temperature_2m_min": 24.0,
            "relative_humidity_2m_mean": 60.0,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 10.0,
        }

    row = {
        "PM10": val("PM10"),
        "PM2_5": val("PM2_5"),
        "NO2": val("NO2"),
        "NO": val("NO"),
        "SO2": val("SO2"),
        "CO": val("CO"),
        "O3": val("O3"),

        "construction_intensity": DEFAULT_CONSTRUCTION_INTENSITY,

        "temperature_2m_max": weather["temperature_2m_max"],
        "temperature_2m_min": weather["temperature_2m_min"],
        "relative_humidity_2m_mean": weather["relative_humidity_2m_mean"],
        "precipitation_sum": weather["precipitation_sum"],
        "wind_speed_10m_max": weather["wind_speed_10m_max"],

        "AQI_lag_1": AQI_lag_1,
        "AQI_lag_3": AQI_lag_3,
        "AQI_lag_7": AQI_lag_7,
        "AQI_roll7_mean": AQI_roll7_mean,
        "AQI_roll7_std": AQI_roll7_std,
        "AQI_change_1d": AQI_change_1d,
        "AQI_change_3d": AQI_change_3d,
    }

    # Enforce exact feature order expected by model
    return pd.DataFrame([row])[FEATURES]

# ---------------- API Schemas ----------------
class CityRequest(BaseModel):
    city: str

# ---------------- Routes ----------------
@app.get("/")
def home():
    return {"message": "AQI Change + Spike Risk API Running (City-only mode)"}

@app.get("/debug/features")
def debug_features():
    return {"features": FEATURES, "n_features": len(FEATURES)}

@app.post("/predict_city")
def predict_city(req: CityRequest):
    try:
        X = build_features_for_city(req.city)

        pred_change = float(reg_model.predict(X)[0])
        spike_prob  = float(clf_model.predict_proba(X)[0][1])
        spike_flag  = int(spike_prob >= PROB_THR)

        return {
            "city": req.city,
            "predicted_aqi_change": round(pred_change, 3),
            "spike_probability": round(spike_prob, 3),
            "risk_level": risk_label(spike_prob),
            "spike_flag": spike_flag,
            "construction_intensity_used": DEFAULT_CONSTRUCTION_INTENSITY
        }
    except Exception as e:
        return {"error": str(e)}
