"""Load the Tetouan SCADA export and turn it into a clean hourly frame."""
import pandas as pd

import config

RENAME = {
    "DateTime": "timestamp",
    "Temperature": "temp_c",
    "Humidity": "humidity_pct",
    "Wind Speed": "wind_ms",
    "general diffuse flows": "ghi_wm2",   # global horizontal irradiance proxy
    "diffuse flows": "dhi_wm2",
    "Zone 1 Power Consumption": "zone1_kw",
    "Zone 2  Power Consumption": "zone2_kw",
    "Zone 3  Power Consumption": "zone3_kw",
}


def load_raw(path=config.RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path).rename(columns=RENAME)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y %H:%M")
    df = df.set_index("timestamp").sort_index()
    if df.index.duplicated().any():
        raise ValueError("duplicate timestamps in raw file")
    return df


def to_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """Mean of six 10-min readings = average kW over the hour = kWh in that hour."""
    counts = raw[config.TARGET_ZONE].resample(config.RESAMPLE).count()
    hourly = raw.resample(config.RESAMPLE).mean()
    hourly["n_readings"] = counts
    incomplete = int((counts < 6).sum())
    hourly = hourly.dropna(subset=[config.TARGET_ZONE])
    hourly.attrs["incomplete_hours"] = incomplete
    return hourly


def quality_report(raw: pd.DataFrame, hourly: pd.DataFrame) -> dict:
    full = pd.date_range(raw.index.min(), raw.index.max(), freq="10min")
    return {
        "raw_rows": int(len(raw)),
        "raw_start": str(raw.index.min()),
        "raw_end": str(raw.index.max()),
        "missing_10min_slots": int(len(full.difference(raw.index))),
        "missing_values": int(raw.isna().sum().sum()),
        "hourly_rows": int(len(hourly)),
        "hours_with_<6_readings": hourly.attrs.get("incomplete_hours", 0),
    }
