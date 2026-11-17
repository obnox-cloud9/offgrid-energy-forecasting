"""Sanity tests: no look-ahead leakage, clean split, physically valid simulation.

    python -m pytest -q tests/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import data, features, model, simulate  # noqa: E402

HOURLY = data.to_hourly(data.load_raw())


def test_dataset_size_and_completeness():
    assert len(HOURLY) >= 3000
    assert HOURLY[config.TARGET_ZONE].isna().sum() == 0


def test_no_target_leakage_into_forecast_day():
    """Forecast is issued at 00:00 of day D, so day D's features may only use data
    from before D. Changing demand at hour t must leave every feature of t's own
    calendar day untouched (and every earlier hour)."""
    X0, _, groups = features.build_features(HOURLY)
    for t in HOURLY.index[[3000, 5000, 7007]]:
        _check_perturbation(X0, groups, t)


def _check_perturbation(X0, groups, t):
    h2 = HOURLY.copy()
    h2.loc[t, config.TARGET_ZONE] *= 3
    X1, _, _ = features.build_features(h2)
    window = X0.index < t.normalize() + pd.Timedelta(days=1)
    cons = [c for c, g in groups.items() if g in ("consumption", "peak_load")]
    pd.testing.assert_frame_equal(X0.loc[window, cons], X1.loc[window, cons])
    # and it *does* show up later, so the test can actually fail
    assert not X0[cons].equals(X1[cons])


def test_split_is_chronological():
    X, y, _ = features.build_features(HOURLY)
    X_tr, X_te, *_ = model.chrono_split(X, y)
    assert X_tr.index.max() < X_te.index.min()
    assert set(X_tr.index.normalize()).isdisjoint(set(X_te.index.normalize()))


def test_simulation_physics():
    idx = pd.date_range("2017-06-01", periods=72, freq="h")
    demand = pd.Series(100 + 40 * np.sin(np.arange(72) / 24 * 2 * np.pi), index=idx)
    weather = pd.DataFrame({"ghi_wm2": np.clip(600 * np.sin((np.arange(72) % 24 - 6) / 12 * np.pi), 0, None),
                            "temp_c": 25.0}, index=idx)
    s = simulate.size_system(demand, weather)
    pv = simulate.pv_output(weather, s.pv_kwp)
    perfect = simulate.run(demand, demand, pv, s, reserve=0.0)
    assert perfect["unmet"].sum() < 1e-6                      # perfect plan -> no shedding
    assert (perfect["soc"] >= s.soc_min_kwh - 1e-9).all() and (perfect["soc"] <= s.batt_kwh + 1e-9).all()
    low = simulate.run(demand, demand * 0.7, pv, s, reserve=0.0)
    assert low["unmet"].sum() > 0                             # under-forecast -> shedding
    assert (low["genset"] <= low["committed"] + 1e-9).all()
