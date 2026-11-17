"""Central configuration. Every number used by the pipeline lives here.

The simulation settings were fixed before the first run and not tuned to hit
any target number. A sensitivity grid is reported alongside the primary setting
so the result can't be read as cherry-picked.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "data" / "raw" / "tetouan_power_2017.csv"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

SEED = 42

# --- data -------------------------------------------------------------------
TARGET_ZONE = "zone1_kw"          # Quads feeder, the largest of the three
RESAMPLE = "1h"                   # 10-min SCADA -> hourly energy (kWh/h == mean kW)

# --- forecasting --------------------------------------------------------------
# Day-ahead forecast issued at 00:00 for the 24 hours of that day, so every
# consumption feature must be lagged by >= 24 hours.
MIN_LAG_H = 24
TEST_FRACTION = 0.20              # last 20% of days held out (chronological)
CV_SPLITS = 5                     # TimeSeriesSplit inside the training period

RF_PARAM_GRID = {
    "n_estimators": [400],
    "max_depth": [None, 20],
    "min_samples_leaf": [3, 5, 10],
    "max_features": [0.33, 0.5, 1.0],
}

# --- off-grid simulation --------------------------------------------------------
# The feeder's real load *shape* is rescaled to a village-scale microgrid.
# The simulation is linear in scale, so percentages don't depend on this choice.
SIM_MEAN_LOAD_KW = 100.0
PV_ENERGY_SHARE = 0.50            # PV sized to supply 50% of annual load energy
PV_SYSTEM_LOSS = 0.14             # inverter, soiling, wiring
PV_TEMP_COEFF = -0.004            # per deg C above 25 C cell temp
PV_NOCT_RISE = 20.0               # cell temp rise at 800 W/m2
BATTERY_HOURS = 4.0               # usable kWh = 4 h of mean load
BATTERY_C_RATE = 0.5              # max power = 0.5 x capacity per hour
BATTERY_RT_EFF = 0.90             # round-trip efficiency
BATTERY_SOC_MIN = 0.10
GENSET_MAX_FRAC_OF_PEAK = 1.0     # genset nameplate = training-period peak
SPINNING_RESERVE = 0.10           # committed headroom as fraction of forecast load

SENS_BATTERY_HOURS = [2.0, 4.0, 6.0]
SENS_RESERVE = [0.0, 0.05, 0.10, 0.20]
PARETO_RESERVES = [round(x * 0.025, 3) for x in range(0, 25)]   # 0 .. 60 %

# --- palette (validated reference palette, light mode) ------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e0"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
