# Off-Grid Energy Demand Forecasting and Analysis

Day-ahead hourly demand forecasting for an isolated PV + battery + diesel microgrid. The project covers a Random Forest forecaster, SHAP explanations, and a dispatch simulation that measures how much better forecasts reduce load shedding.

Everything runs on **real metered data**: one year of 10-minute SCADA load, temperature, humidity, wind and solar irradiance from Tétouan, Morocco (UCI dataset 849). Every number below comes from `results/metrics.json`, which the pipeline writes. Nothing here is typed in by hand.

```
pip install -r requirements.txt
python run_pipeline.py          # full run, ~7 min on 2 CPU cores
python run_pipeline.py --fast   # ~40 s smoke test with a single RF config
python -m pytest -q tests/      # leakage, split and simulation-physics tests
```

---

## Headline results (held-out test period: 22 Oct – 30 Dec 2017, 1,680 hours)

| | Measured |
|---|---|
| Hourly observations used | **8,400** (8,736 hourly rows; first 14 days consumed by lag features) |
| Random Forest R² (test) | **0.966**, with a 95% day-block bootstrap CI of 0.956–0.975 |
| Random Forest MAE / MAPE | 830 kW / **2.84%** |
| Skill vs seasonal-naive forecast (MAE) | **−15.0%** error |
| R² with weather lagged 24 h (no perfect-weather assumption) | 0.966 |
| Same recipe on Zone 2 / Zone 3 | R² 0.957 / 0.833 |
| Unmet demand reduction, RF vs seasonal-naive planning (10% reserve) | **63.0%** (658 → 244 kWh) |
| Same comparison at an equal diesel commitment budget | **64.9%** |
| Reduction across 12 battery-size × reserve settings | −6.7% to 70.2% (median 57.4%) |

### Model comparison (test set)

| Model | R² | MAE (kW) | MAPE | Skill vs naive |
|---|---|---|---|---|
| Seasonal naive (same hour yesterday) | 0.941 | 976 | 3.38% | 0% |
| Weekly naive (same hour last week) | 0.936 | 1,082 | 3.74% | −10.8% |
| 7-day same-hour mean | 0.948 | 992 | 3.46% | −1.6% |
| Ridge regression (same features) | 0.969 | 829 | 2.93% | +15.1% |
| Random Forest, raw target | 0.958 | 896 | 3.14% | +8.2% |
| **Random Forest, level-normalized (final)** | **0.966** | **830** | **2.84%** | **+15.0%** |

**Reading R² honestly.** Hourly load has a strong daily rhythm, so even "copy yesterday" scores R² = 0.94. R² alone flatters every model here. The meaningful number is **skill over the naive forecast: 15% less error**. Ridge matches the Random Forest on this feature set, and that's reported rather than hidden. The forest earns its place through SHAP interpretability and its handling of holiday and weekday interactions, not through a large accuracy margin.

---

## Pipeline

```
data/raw (10-min SCADA) ──► src/data.py      hourly resample, quality report
                         ──► src/features.py  31 leak-free day-ahead features
                         ──► src/model.py     chronological split, baselines, RF + TimeSeriesSplit grid search
                         ──► src/explain.py   exact TreeSHAP, grouped drivers
                         ──► src/simulate.py  PV + battery + genset dispatch with day-ahead commitment
                         ──► src/plots.py     16 figures in results/figures/
run_pipeline.py orchestrates; config.py holds every parameter.
```

### 1. EDA (`figures/01–05`)
- Demand peaks at **20:00** (evening lighting and residential load); the shape is consistent across weekdays.
- Summer (Jul–Aug) demand is **1.17×** winter (Jan–Feb).
- Temperature vs demand: r = 0.44 hourly, **0.73 daily**. Cooling load drives the seasonal level.
- Data quality: 52,416 rows, zero gaps, zero missing values.

### 2. Features (31; forecast issued 00:00 for the next 24 h)
| Group | Features |
|---|---|
| temporal | hour, weekday, month, day of year (+ sin/cos), weekend, Moroccan public holidays (`holidays` lib), Ramadan (Hijri calendar) |
| weather | temperature, humidity, wind, GHI, DHI, previous-day mean temperature, cooling degree-hours |
| consumption | demand lags 24/48/72/168/336 h, 7-day same-hour mean, previous-day and previous-week means |
| peak_load | previous-day peak, trough, load factor, lag-24 / previous-day peak ratio |

**No leakage.** Every demand-derived feature uses only data from before the forecast day. `tests/test_pipeline.py` perturbs demand at several hours and asserts that no feature on that day changes.

### 3. Model
- **Chronological split on whole days:** train 15 Jan – 21 Oct (6,720 h), test 22 Oct – 30 Dec (1,680 h).
- **Hyperparameters** chosen by 5-fold `TimeSeriesSplit` inside the training period only (18 configs). Best: 400 trees, max_depth 20, min_samples_leaf 10, max_features 0.33. CV R² 0.913, train R² 0.985.
- **Level normalization.** Trees can't extrapolate, and the test period (winter) sits below most of the training range. The forest predicts `demand / 7-day same-hour mean`, so the level comes from the last week of data and the trees learn the shape. This lifted test R² from 0.958 to 0.966 and doubled skill over naive (8% → 15%).

### 4. SHAP (`figures/10–13`, `results/shap_*.csv`)
Exact TreeSHAP on 600 test hours. The ratio model's SHAP values are rescaled to kW per row, so they add up exactly to the kW prediction (max additivity error 1e-10 kW).

| Driver group | Share of mean \|SHAP\| | Top features |
|---|---|---|
| temporal | 41.9% | weekday (14.0%), weekend, holiday |
| consumption | 25.5% | previous-week mean (11.8%), previous-day mean, lag 168 h |
| peak_load | 19.9% | previous-day peak (11.2%), trough, load factor |
| weather | 12.7% | wind, diffuse irradiance, global irradiance |

Because the model explains deviations from the 7-day level, temperature's seasonal effect is already inside that level. That's why weather ranks fourth here even though it correlates 0.73 with daily demand.

### 5. Off-grid capacity planning simulation (`figures/14–16`)

**System** (sized on training data only, fixed before the first run):
PV 299 kWp (supplies 50% of annual load energy) · battery 400 kWh / 200 kW, 90% round-trip efficiency, 10% minimum state of charge · diesel genset 156 kW (training peak) · load rescaled to a 100 kW average.

**Operating rule.** Each midnight the operator commits genset capacity hour by hour. The plan simulates the day on *forecast* demand from the true battery state, then adds a spinning reserve (10% of forecast). In real time, demand is served by PV, then battery, then genset up to the committed capacity; anything left is unmet. PV uses observed irradiance for every strategy, so the only difference between strategies is the demand forecast.

| Strategy (10% reserve) | Unmet (kWh) | Shortfall hours | Genset committed (MWh) |
|---|---|---|---|
| Seasonal naive (yesterday's profile) | 657.9 | 60 | 117.5 |
| Static weekday×hour profile from training | 46.6 | 12 | 138.6 |
| **Random Forest** | **243.7** | **22** | **116.6** |
| Perfect forecast (oracle bound) | 0.0 | 0 | 117.2 |

- RF vs seasonal naive: **−63.0% unmet demand while committing 0.8% less diesel**. At an exactly matched diesel budget the reduction is 64.9%.
- The static profile sheds less load only because it over-commits **19% more diesel** (it's trained on hotter months). At equal budget, RF beats it by 61% at 10% reserve. The Pareto chart (`figures/15`) shows the static profile overtaking RF once the budget passes about 135 MWh (reserve above ~20%). That's the high-cost, low-shedding corner.
- Sensitivity (`results/simulation_sensitivity.csv`): the gain is 52–70% for reserves of 5–20%, and **−5% to −7% at zero reserve**. A mean forecast with no headroom under-commits about half the time, whatever the model. Battery size barely matters because shortfalls happen at the evening peak, after the battery has already been planned into the day.

---

## Resume claim verification

| Bullet | Status | Measured |
|---|---|---|
| 3,000+ observations | ✅ holds | 8,400 hourly observations |
| Random Forest R² of 0.89 | ⚠️ not reproduced as written | **0.966** (CI 0.956–0.975). Higher than claimed, but the seasonal-naive baseline already reaches 0.941 |
| 27% reduction in simulated unmet demand | ⚠️ not reproduced as written | **63.0%** at the primary setting; −6.7% to 70.2% across settings |
| SHAP isolates weather, consumption and peak-load drivers | ✅ holds | 4 driver groups quantified, top features per group |

Suggested bullets that match this code exactly:

- Forecasted day-ahead hourly demand for an off-grid PV–battery–diesel microgrid using real SCADA load and weather data across **8,400 observations**
- **Random Forest** reached **R² of 0.966** (MAPE 2.8%) on a held-out winter period, cutting error **15%** versus a seasonal-naive baseline
- Drove day-ahead genset commitment with demand forecasts, reducing simulated unmet energy demand by **63%** at equal diesel budget
- Interpreted predictions using **SHAP analysis**, attributing demand to calendar (42%), consumption (25%), peak-load (20%) and weather (13%) drivers

---

## How this compares to existing public projects

Several GitHub projects use the same Tétouan data, for example `Kevinbui16/Electricity-Consumption-Prediction`, `salahmak/ML-project` and `chandanc5525/EnergyConsumption_PredictionModel`. The common pattern is to use `train_test_split` (a random shuffle) on 10-minute rows and predict consumption from same-timestamp features. That lets the model interpolate between neighbouring 10-minute readings, which produces very high scores that don't reflect forecasting skill. This project adds:

1. a strict day-ahead setup with a chronological split and a leakage test
2. naive baselines and a skill score, so R² is put in context
3. level normalization so the forest generalizes across seasons
4. exact, grouped SHAP in kW
5. an off-grid dispatch simulation that turns forecast quality into reliability and diesel figures, with sensitivity and equal-budget checks

## Limitations
- The feeder is a city distribution zone, rescaled to village size; real off-grid loads are spikier.
- Weather at the target hour is observed (a perfect weather forecast). The lagged-weather variant scores the same R², so the result doesn't depend on it.
- PV comes from a simple irradiance and temperature model, not a full PV simulator. Unmet demand comes only from day-ahead under-commitment; outages and genset failures aren't modelled.
- One year of data means a single winter test period.

## Layout
```
config.py            all parameters
run_pipeline.py      end-to-end run
src/                 data, features, model, explain, simulate, plots
tests/               leakage / split / physics tests
data/raw/            Tetouan CSV (CC BY 4.0) + data/README.md with source and checksum
results/             metrics.json, CSVs, trained model, run_log.txt, figures/
```
