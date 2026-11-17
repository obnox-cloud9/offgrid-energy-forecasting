# Data

`raw/tetouan_power_2017.csv` is the **Power Consumption of Tetouan City** dataset.

- Source: UCI Machine Learning Repository, dataset 849 — https://archive.ics.uci.edu/dataset/849
- DOI: 10.24432/C5B034
- Citation: Salam, A. & El Hibaoui, A. (2018). *Comparison of Machine Learning Algorithms for the Power Consumption Prediction: Case Study of Tetouan city.* IRSEC 2018.
- Origin: SCADA system of Amendis, the distribution operator for Tétouan, Morocco
- License: CC BY 4.0 (UCI)
- SHA-256 of the file shipped here: `ca8b2f30cf07dd872cb7f2aa45dac110ee38dbe5fc5910a6ff4bcc530dae3a40`

| Column | Unit | Used as |
|---|---|---|
| DateTime | 10-min stamps, 2017-01-01 00:00 → 2017-12-30 23:50 | index |
| Temperature | °C | weather feature, PV cell temperature |
| Humidity | % | weather feature |
| Wind Speed | m/s | weather feature |
| general diffuse flows | W/m² (treated as global horizontal irradiance) | weather feature, PV output |
| diffuse flows | W/m² | weather feature |
| Zone 1 Power Consumption | kW | **forecast target** (Quads feeder) |
| Zone 2 / Zone 3 Power Consumption | kW | robustness check only |

52,416 rows with no gaps and no missing values. Resampled to 8,736 hourly rows, of which 8,400 remain after dropping the first 14 days needed for lag features.

`processed/hourly.csv` is written by the pipeline.

## Why this dataset for an "off-grid" project

Public off-grid microgrid datasets that pair load with on-site weather and irradiance at hourly resolution for a full year are rare. Tetouan gives real metered load, temperature, humidity, wind and irradiance from the same place and time. The feeder's load *shape* is rescaled to a 100 kW-average village microgrid for the simulation. The simulation is linear in scale, so the percentages don't depend on that number. The forecasting metrics are computed on the original kW values.
