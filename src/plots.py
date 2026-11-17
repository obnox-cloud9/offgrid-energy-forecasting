"""All figures. One style, one axis per chart, colors from the validated palette."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import config

C = config.C
# color follows the strategy across every simulation chart
STRAT_COLOR = {"Random Forest": C[0], "Seasonal naive": C[1], "Static weekly profile": C[3],
               "Perfect forecast (oracle)": C[2]}
SEQ = LinearSegmentedColormap.from_list("seq_blue", config.SEQ_BLUE)

plt.rcParams.update({
    "figure.facecolor": config.SURFACE, "axes.facecolor": config.SURFACE,
    "savefig.facecolor": config.SURFACE, "axes.edgecolor": config.GRID,
    "axes.labelcolor": config.INK_2, "xtick.color": config.INK_2, "ytick.color": config.INK_2,
    "text.color": config.INK, "axes.titlecolor": config.INK, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True,
    "grid.color": config.GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9.5, "legend.frameon": False,
    "lines.linewidth": 2.0,
})


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(config.FIGURES / name, dpi=150)
    plt.close(fig)


def eda(hourly: pd.DataFrame, y: str):
    s = hourly[y] / 1000
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(s.index, s.values, color=C[0], linewidth=0.5, alpha=0.35, label="Hourly")
    ax.plot(s.resample("D").mean().index, s.resample("D").mean().values, color=C[0], label="Daily mean")
    ax.set_title("Zone 1 hourly demand, 2017")
    ax.set_ylabel("MW")
    ax.legend(loc="upper left")
    _save(fig, "01_demand_timeseries.png")

    pivot = s.groupby([s.index.month, s.index.hour]).mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap=SEQ)
    ax.set_yticks(range(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day")
    ax.grid(False)
    ax.set_title("Average demand by month and hour (MW)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    _save(fig, "02_month_hour_heatmap.png")

    prof = s.groupby([s.index.dayofweek >= 5, s.index.hour]).mean().unstack(0)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(prof.index, prof[False], color=C[0], label="Weekday")
    ax.plot(prof.index, prof[True], color=C[1], label="Weekend")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("MW")
    ax.set_title("Daily load shape: evening peak dominates")
    ax.legend()
    _save(fig, "03_daily_profile.png")

    d = hourly.resample("D").mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(d["temp_c"], d[y] / 1000, s=14, color=C[0], alpha=0.7, edgecolor=config.SURFACE, linewidth=0.5)
    ax.set_xlabel("Daily mean temperature (°C)")
    ax.set_ylabel("Daily mean demand (MW)")
    ax.set_title("Hotter days draw more power")
    _save(fig, "04_temp_vs_demand.png")

    cols = ["temp_c", "humidity_pct", "wind_ms", "ghi_wm2", "dhi_wm2", y]
    corr = hourly[cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    div = LinearSegmentedColormap.from_list("div", [C[0], "#e6e5e0", C[1]])
    im = ax.imshow(corr.values, cmap=div, vmin=-1, vmax=1)
    labels = ["Temp", "Humidity", "Wind", "GHI", "DHI", "Demand"]
    ax.set_xticks(range(6), labels, rotation=45)
    ax.set_yticks(range(6), labels)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8, color=config.INK)
    ax.grid(False)
    ax.set_title("Pearson correlation, hourly")
    fig.colorbar(im, ax=ax, fraction=0.04)
    _save(fig, "05_correlation.png")


def forecast(y_te, preds: dict, cut):
    week = y_te.index[: 24 * 7]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(week, y_te.loc[week] / 1000, color=config.INK, label="Actual")
    for (name, p), col in zip(preds.items(), [C[0], C[1]]):
        ax.plot(week, pd.Series(p, index=y_te.index).loc[week] / 1000, color=col, label=name, linewidth=1.6)
    ax.set_ylabel("MW")
    ax.set_title("First test week: day-ahead forecasts vs actual")
    ax.legend(ncol=3, loc="upper left")
    _save(fig, "06_forecast_week.png")

    rf = preds["Random Forest"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_te / 1000, rf / 1000, s=6, alpha=0.4, color=C[0])
    lo, hi = y_te.min() / 1000, y_te.max() / 1000
    ax.plot([lo, hi], [lo, hi], color=config.INK_2, linewidth=1, linestyle="--")
    ax.set_xlabel("Actual (MW)")
    ax.set_ylabel("Predicted (MW)")
    ax.set_title("Random Forest, test set")
    _save(fig, "07_pred_vs_actual.png")

    err = (pd.Series(rf, index=y_te.index) - y_te) / 1000
    fig, ax = plt.subplots(figsize=(7, 3.4))
    e = err.groupby(err.index.hour)
    ax.bar(range(24), e.apply(lambda x: x.abs().mean()), color=C[0], width=0.8)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("MAE (MW)")
    ax.set_title("Where the forecast misses: error by hour")
    _save(fig, "08_error_by_hour.png")


def model_comparison(table: pd.DataFrame):
    t = table.sort_values("R2")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    colors = [C[0] if n == "Random Forest" else "#b7b6b0" for n in t.index]
    ax.barh(t.index, t["R2"], color=colors, height=0.6)
    for i, v in enumerate(t["R2"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8.5, color=config.INK)
    ax.set_xlim(min(0, t["R2"].min()) , 1.0)
    ax.set_xlabel("R² on held-out test period")
    ax.set_title("Model comparison")
    ax.grid(axis="y", visible=False)
    _save(fig, "09_model_comparison.png")


def shap_plots(values, sample, feat_imp, grp_imp):
    import shap
    shap.summary_plot(values, sample, show=False, max_display=15, plot_size=(8, 6))
    fig = plt.gcf()
    fig.axes[0].set_title("SHAP: top 15 features (test sample)")
    _save(fig, "10_shap_beeswarm.png")

    g = grp_imp.sort_values("share_pct", ascending=False)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(g.index[::-1], g["share_pct"][::-1], color=C[0], height=0.55)
    for i, v in enumerate(g["share_pct"][::-1]):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8.5)
    ax.set_xlabel("Share of total mean |SHAP|")
    ax.set_title("Driver groups")
    ax.grid(axis="y", visible=False)
    _save(fig, "11_shap_groups.png")

    for feat, name in [("temp_c", "12_shap_dependence_temp.png"), ("hour", "13_shap_dependence_hour.png")]:
        if feat in sample:
            i = list(sample.columns).index(feat)
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.scatter(sample[feat], values[:, i] / 1000, s=10, color=C[0], alpha=0.6)
            ax.axhline(0, color=config.INK_2, linewidth=0.8)
            ax.set_xlabel(feat)
            ax.set_ylabel("SHAP value (MW)")
            ax.set_title(f"Effect of {feat} on predicted demand")
            _save(fig, name)


def simulation(sims: dict, pareto: pd.DataFrame):
    key_b, key_f = "Seasonal naive", "Random Forest"
    sb, sf = sims[key_b], sims[key_f]
    days = sb.index.normalize().unique()
    # the 7-day window where the baseline sheds the most load
    worst = sb["unmet"].groupby(sb.index.normalize()).sum().rolling(7).sum().idxmax()
    win = (sb.index > worst - pd.Timedelta(days=7)) & (sb.index <= worst + pd.Timedelta(hours=23))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(sb.index[win], sb["demand"][win], color=config.INK, label="Demand", linewidth=1.6)
    ax.plot(sb.index[win], sb["committed"][win], color=STRAT_COLOR[key_b], label=f"Genset committed ({key_b})", linewidth=1.4)
    ax.plot(sf.index[win], sf["committed"][win], color=STRAT_COLOR[key_f], label=f"Genset committed ({key_f})", linewidth=1.4)
    ax.set_ylabel("kW")
    ax.set_title("Week with the most baseline load shedding")
    ax.legend(ncol=3, loc="upper left")
    _save(fig, "14_sim_week.png")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, g in pareto.groupby("strategy", sort=False):
        col = STRAT_COLOR[name]
        ax.plot(g["genset_committed_mwh"], g["unmet_mwh"], marker="o", markersize=4, color=col, label=name)
    ax.set_xlabel("Genset energy committed (MWh)  — fuel/ops budget")
    ax.set_ylabel("Unmet demand (MWh)")
    ax.set_title("Reliability vs genset budget (reserve swept 0–60%)")
    ax.legend()
    _save(fig, "15_pareto_unmet_vs_commitment.png")

    names = list(sims)
    unmet = [sims[n]["unmet"].sum() / 1000 for n in names]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    cols = [STRAT_COLOR[n] for n in names]
    ax.barh(names[::-1], unmet[::-1], color=cols[::-1], height=0.55)
    for i, v in enumerate(unmet[::-1]):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8.5)
    ax.set_xlabel("Unmet demand over test period (MWh)")
    ax.set_title("Unmet demand at 10% spinning reserve")
    ax.grid(axis="y", visible=False)
    _save(fig, "16_unmet_by_strategy.png")
