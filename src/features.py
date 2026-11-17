"""Feature engineering for a day-ahead (issued 00:00) hourly demand forecast.

Feature groups (used later to aggregate SHAP values):
  temporal     - calendar, holidays, Ramadan
  weather      - temperature, humidity, wind, irradiance
  consumption  - demand lags >= 24 h and rolling stats of the previous day/week
  peak_load    - previous-day peak / trough / load factor

Weather at the target hour is the *observed* value, i.e. a perfect weather
forecast. This is standard in load-forecasting studies; `weather_mode="lagged"`
swaps it for yesterday's weather so the effect can be measured.
"""
import numpy as np
import pandas as pd
import holidays
from hijridate import Gregorian

import config

Y = config.TARGET_ZONE


def ramadan_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """Ramadan (Hijri month 9) from the Umm al-Qura calendar; Morocco can differ by ~1 day."""
    days = pd.Series(index.normalize().unique())
    is_ram = {d: Gregorian(d.year, d.month, d.day).to_hijri().month == 9 for d in days}
    return np.array([is_ram[d] for d in index.normalize()])


def add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    ma = holidays.country_holidays("MA", years=sorted(set(idx.year)))
    out = pd.DataFrame(index=idx)
    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["month"] = idx.month
    out["dayofyear"] = idx.dayofyear
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["is_holiday"] = np.array([d in ma for d in idx.date]).astype(int)
    out["is_ramadan"] = ramadan_mask(idx).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    return out


def add_consumption(y: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=y.index)
    for lag in (24, 48, 72, 168, 336):
        out[f"lag_{lag}h"] = y.shift(lag)
    out["same_hour_mean_7d"] = pd.concat([y.shift(24 * k) for k in range(1, 8)], axis=1).mean(axis=1)
    # previous calendar day stats, aligned to every hour of the target day
    daily = y.resample("D")
    prev = pd.DataFrame({
        "prev_day_mean": daily.mean(),
        "prev_day_max": daily.max(),
        "prev_day_min": daily.min(),
    }).shift(1)
    prev["prev_week_mean"] = daily.mean().rolling(7).mean().shift(1)
    prev = prev.reindex(y.index.normalize())
    prev.index = y.index
    out["prev_day_mean"] = prev["prev_day_mean"]
    out["prev_week_mean"] = prev["prev_week_mean"]
    out["lag24_vs_prev_week"] = out["lag_24h"] / out["prev_week_mean"]
    return out, prev


def add_peak(prev: pd.DataFrame, consumption: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=prev.index)
    out["prev_day_peak"] = prev["prev_day_max"]
    out["prev_day_trough"] = prev["prev_day_min"]
    out["prev_day_load_factor"] = prev["prev_day_mean"] / prev["prev_day_max"]
    out["lag24_peak_ratio"] = consumption["lag_24h"] / prev["prev_day_max"]
    return out


def add_weather(df: pd.DataFrame, mode: str = "observed") -> pd.DataFrame:
    cols = ["temp_c", "humidity_pct", "wind_ms", "ghi_wm2", "dhi_wm2"]
    w = df[cols].copy()
    if mode == "lagged":
        w = w.shift(24)
    elif mode != "observed":
        raise ValueError(mode)
    w["temp_prev_day_mean"] = df["temp_c"].resample("D").mean().shift(1).reindex(df.index.normalize()).values
    w["cooling_deg_h"] = (w["temp_c"] - 22).clip(lower=0)
    return w


def build_features(hourly: pd.DataFrame, weather_mode: str = "observed"):
    y = hourly[Y]
    temporal = add_temporal(hourly)
    consumption, prev = add_consumption(y)
    peak = add_peak(prev, consumption)
    weather = add_weather(hourly, weather_mode)
    X = pd.concat([temporal, weather, consumption, peak], axis=1)
    groups = {c: "temporal" for c in temporal} | {c: "weather" for c in weather} \
        | {c: "consumption" for c in consumption} | {c: "peak_load" for c in peak}
    frame = pd.concat([X, y.rename("target")], axis=1).dropna()
    return frame.drop(columns="target"), frame["target"], groups
