"""Chronological split, baselines, Random Forest with time-series CV, metrics."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config


def chrono_split(X, y, test_fraction=config.TEST_FRACTION):
    """Split on whole days so no test day leaks partially into training."""
    days = X.index.normalize().unique()
    cut = days[int(len(days) * (1 - test_fraction))]
    tr, te = X.index < cut, X.index >= cut
    return X[tr], X[te], y[tr], y[te], cut


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    peak = y_true >= np.quantile(y_true, 0.9)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE_kW": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_kW": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_pct": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "peak_hours_MAPE_pct": float(np.mean(np.abs((y_true[peak] - y_pred[peak]) / y_true[peak])) * 100),
    }


def block_bootstrap_r2(y_true, y_pred, n=1000, seed=config.SEED):
    """95% CI for R2 resampling whole days (keeps within-day correlation)."""
    df = pd.DataFrame({"t": y_true, "p": y_pred})
    days = [g for _, g in df.groupby(df.index.normalize())]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        s = pd.concat([days[i] for i in rng.integers(0, len(days), len(days))])
        stats.append(r2_score(s["t"], s["p"]))
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


class LevelNormalizedRF(BaseEstimator, RegressorMixin):
    """Random Forest that predicts demand / (7-day same-hour mean).

    Trees can't extrapolate: a forest trained on Jan-Oct never outputs a value
    below its training range, but Nov-Dec load sits lower. Learning the
    *ratio* to the recent level lets the forest model shape (hour, weekday,
    weather, holidays) while the level comes from the last week of data.
    The level is itself a >=24 h lagged feature, so nothing leaks.
    """

    def __init__(self, base_col="same_hour_mean_7d", n_estimators=400, max_depth=None,
                 min_samples_leaf=1, max_features=1.0, random_state=config.SEED):
        self.base_col = base_col
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        self.rf_ = RandomForestRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
            random_state=self.random_state, n_jobs=-1,
        ).fit(X, np.asarray(y) / X[self.base_col].values)
        self.feature_names_in_ = np.array(X.columns)
        return self

    def predict(self, X):
        return self.rf_.predict(X) * X[self.base_col].values


def fit_rf(X_tr, y_tr):
    search = GridSearchCV(
        LevelNormalizedRF(),
        config.RF_PARAM_GRID,
        cv=TimeSeriesSplit(n_splits=config.CV_SPLITS),
        scoring="r2",
        n_jobs=1,
    )
    search.fit(X_tr, y_tr)
    cv = pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score"]]
    return search.best_estimator_, search.best_params_, float(search.best_score_), cv


def fit_ridge(X_tr, y_tr):
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
    return m.fit(X_tr, y_tr)


def baseline_predictions(X_te):
    return {
        "Seasonal naive (same hour yesterday)": X_te["lag_24h"].values,
        "Weekly naive (same hour last week)": X_te["lag_168h"].values,
        "7-day same-hour mean": X_te["same_hour_mean_7d"].values,
    }
