"""Off-grid PV + battery + diesel genset simulation with day-ahead genset commitment.

Why a forecast matters here
---------------------------
A diesel genset in an isolated grid has to be scheduled ahead (crew, fuel,
warm-up, minimum run time). Each morning the operator commits a genset
capacity for every hour of the day. In real time the genset can deliver up to
the committed capacity, the battery covers what it can, and whatever is left
is *unmet demand* (load shedding).

Planning (identical for every strategy, only the demand forecast differs):
  1. Simulate the day on *forecast* demand, starting from the true battery
     state of charge at 00:00, to get the genset energy the plan needs each hour.
  2. Commit  c_t = min(G_max, planned_genset_t + reserve * forecast_t).

Real time: actual demand -> PV first -> battery -> genset up to c_t -> unmet.
PV uses observed irradiance for every strategy, so differences come only
from the demand forecast.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config


@dataclass
class System:
    pv_kwp: float
    batt_kwh: float
    batt_kw: float
    eff_one_way: float
    soc_min_kwh: float
    genset_kw: float


def pv_output(weather: pd.DataFrame, kwp: float) -> np.ndarray:
    g = weather["ghi_wm2"].clip(lower=0).values
    t_cell = weather["temp_c"].values + g / 800.0 * config.PV_NOCT_RISE
    derate = 1 + config.PV_TEMP_COEFF * (t_cell - 25)
    return kwp * g / 1000.0 * derate * (1 - config.PV_SYSTEM_LOSS)


def size_system(train_load_kw: pd.Series, train_weather: pd.DataFrame, battery_hours=config.BATTERY_HOURS):
    pv_per_kwp = pv_output(train_weather, 1.0)
    kwp = config.PV_ENERGY_SHARE * train_load_kw.sum() / pv_per_kwp.sum()
    cap = battery_hours * train_load_kw.mean()
    return System(
        pv_kwp=kwp,
        batt_kwh=cap,
        batt_kw=cap * config.BATTERY_C_RATE,
        eff_one_way=np.sqrt(config.BATTERY_RT_EFF),
        soc_min_kwh=cap * config.BATTERY_SOC_MIN,
        genset_kw=config.GENSET_MAX_FRAC_OF_PEAK * train_load_kw.max(),
    )


def _step(demand, pv, soc, gen_cap, s: System):
    """One hour of dispatch. Returns (soc, genset_kwh, unmet_kwh, curtailed_kwh)."""
    net = demand - pv
    if net <= 0:   # surplus PV charges the battery
        charge = min(-net, s.batt_kw, (s.batt_kwh - soc) / s.eff_one_way)
        return soc + charge * s.eff_one_way, 0.0, 0.0, -net - charge
    discharge = min(net, s.batt_kw, max(soc - s.soc_min_kwh, 0) * s.eff_one_way)
    soc -= discharge / s.eff_one_way
    rest = net - discharge
    gen = min(rest, gen_cap)
    return soc, gen, rest - gen, 0.0


def plan_day(forecast, pv, soc0, s: System, reserve):
    soc, plan = soc0, np.zeros(len(forecast))
    for h in range(len(forecast)):
        soc, gen, unmet, _ = _step(forecast[h], pv[h], soc, np.inf, s)
        plan[h] = gen
    return np.minimum(s.genset_kw, plan + reserve * forecast)


def run(demand: pd.Series, forecast: pd.Series, pv: np.ndarray, s: System, reserve=config.SPINNING_RESERVE):
    d, f = demand.values, forecast.values
    soc = 0.5 * s.batt_kwh
    rec = {k: np.zeros(len(d)) for k in ("committed", "genset", "unmet", "curtailed", "soc")}
    day_id = demand.index.normalize()
    for day in day_id.unique():
        idx = np.where(day_id == day)[0]
        commit = plan_day(f[idx], pv[idx], soc, s, reserve)
        for j, i in enumerate(idx):
            soc, gen, unmet, curt = _step(d[i], pv[i], soc, commit[j], s)
            rec["committed"][i], rec["genset"][i] = commit[j], gen
            rec["unmet"][i], rec["curtailed"][i], rec["soc"][i] = unmet, curt, soc
    out = pd.DataFrame(rec, index=demand.index)
    out["demand"], out["pv"] = d, pv
    return out


def summarize(sim: pd.DataFrame) -> dict:
    total = sim["demand"].sum()
    return {
        "unmet_kwh": float(sim["unmet"].sum()),
        "unmet_pct_of_demand": float(sim["unmet"].sum() / total * 100),
        "hours_with_shortfall": int((sim["unmet"] > 1e-6).sum()),
        "genset_committed_kwh": float(sim["committed"].sum()),
        "genset_generated_kwh": float(sim["genset"].sum()),
        "pv_curtailed_kwh": float(sim["curtailed"].sum()),
    }
