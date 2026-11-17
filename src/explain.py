"""SHAP explanations for the level-normalized Random Forest.

The forest predicts r = demand / base, with base = 7-day same-hour mean.
TreeExplainer gives an exact additive split of r:  r = E + sum(phi).
Multiplying by the row's base gives an exact additive split in kW:
    prediction_kW = base * E + sum(base * phi)
so the SHAP values reported here are in kW and add up to the prediction.
"""
import numpy as np
import pandas as pd
import shap

import config


def shap_values(model, X_test: pd.DataFrame, n_samples=600):
    sample = X_test.sample(min(n_samples, len(X_test)), random_state=config.SEED).sort_index()
    explainer = shap.TreeExplainer(model.rf_)
    phi = explainer.shap_values(sample, check_additivity=False)
    base = sample[model.base_col].values[:, None]
    values_kw = phi * base
    expected = float(np.atleast_1d(explainer.expected_value)[0])
    recon = expected * base[:, 0] + values_kw.sum(axis=1)
    max_err = float(np.max(np.abs(recon - model.predict(sample))))
    return sample, values_kw, expected, max_err


def importance_tables(values, sample: pd.DataFrame, groups: dict):
    feat = pd.DataFrame({
        "feature": sample.columns,
        "group": [groups[c] for c in sample.columns],
        "mean_abs_shap_kW": np.abs(values).mean(axis=0),
    }).sort_values("mean_abs_shap_kW", ascending=False)
    feat["share_pct"] = feat["mean_abs_shap_kW"] / feat["mean_abs_shap_kW"].sum() * 100
    grp = feat.groupby("group")[["mean_abs_shap_kW", "share_pct"]].sum().sort_values("share_pct", ascending=False)
    return feat.reset_index(drop=True), grp
