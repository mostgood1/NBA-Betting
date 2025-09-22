from __future__ import annotations

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, mean_squared_error
import numpy as np
from pathlib import Path
import joblib

from .config import paths


def _time_series_cv(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for tr, te in tscv.split(X):
        yield tr, te


def train_models(df: pd.DataFrame):
    # split features/targets
    base_feats = [
        "elo_diff",
        "home_rest_days", "visitor_rest_days", "home_b2b", "visitor_b2b",
        # Rolling form
        "home_form_off_5", "home_form_def_5", "visitor_form_off_5", "visitor_form_def_5",
        # Schedule intensity
        "home_games_last3", "visitor_games_last3", "home_games_last5", "visitor_games_last5",
        "home_3in4", "visitor_3in4", "home_4in6", "visitor_4in6",
    ]
    df = df.dropna(subset=["target_home_win", "target_margin", "target_total"])  # keep fully known games
    # Keep only columns that actually exist (period columns may not). This prevents KeyError if older features are missing.
    use_feats = [c for c in base_feats if c in df.columns]
    X = df[use_feats].fillna(0)

    # Win probability with scaling + small C grid over saga solver
    y_win = df["target_home_win"].astype(int)
    Cs = [0.25, 0.5, 1.0, 2.0]
    c_losses: dict[float, list[float]] = {c: [] for c in Cs}
    for tr, te in _time_series_cv(X, y_win):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y_win.iloc[tr], y_win.iloc[te]
        for c in Cs:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(solver="saga", max_iter=5000, C=c))
            ])
            pipe.fit(Xtr, ytr)
            p = pipe.predict_proba(Xte)[:, 1]
            c_losses[c].append(log_loss(yte, p, labels=[0, 1]))
    mean_c = {c: float(np.mean(v)) for c, v in c_losses.items()}
    best_c = min(mean_c, key=mean_c.get)
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(solver="saga", max_iter=5000, C=best_c))
    ])
    clf.fit(X, y_win)
    cv_losses = list(c_losses[best_c])

    # Spread (margin) regression with scaling + small alpha grid
    y_margin = df["target_margin"].astype(float)
    ridge_alphas = [1.0, 2.0, 5.0, 10.0]
    alpha_rmse_m: dict[float, list[float]] = {a: [] for a in ridge_alphas}
    for tr, te in _time_series_cv(X, y_margin):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y_margin.iloc[tr], y_margin.iloc[te]
        for a in ridge_alphas:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=a))
            ])
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            rmse = float(np.sqrt(mean_squared_error(yte, pred)))
            alpha_rmse_m[a].append(rmse)
    mean_rmse_m = {a: float(np.mean(v)) for a, v in alpha_rmse_m.items()}
    best_alpha_m = min(mean_rmse_m, key=mean_rmse_m.get)
    reg_margin = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=best_alpha_m))
    ])
    reg_margin.fit(X, y_margin)
    cv_rmse_m = list(alpha_rmse_m[best_alpha_m])

    # Totals regression with scaling + small alpha grid
    y_total = df["target_total"].astype(float)
    alpha_rmse_t: dict[float, list[float]] = {a: [] for a in ridge_alphas}
    for tr, te in _time_series_cv(X, y_total):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y_total.iloc[tr], y_total.iloc[te]
        for a in ridge_alphas:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=a))
            ])
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            rmse = float(np.sqrt(mean_squared_error(yte, pred)))
            alpha_rmse_t[a].append(rmse)
    mean_rmse_t = {a: float(np.mean(v)) for a, v in alpha_rmse_t.items()}
    best_alpha_t = min(mean_rmse_t, key=mean_rmse_t.get)
    reg_total = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=best_alpha_t))
    ])
    reg_total.fit(X, y_total)
    cv_rmse_t = list(alpha_rmse_t[best_alpha_t])

    # Halves models (same features baseline)
    models_halves = {}
    for half in ("h1", "h2"):
        cols = [f"target_{half}_home_win", f"target_{half}_margin", f"target_{half}_total"]
        if not all(c in df.columns for c in cols):
            continue
        mask = df[cols].notna().all(axis=1)
        if not mask.any():
            continue
        Xh = X.loc[mask]
        y_hw = df.loc[mask, f"target_{half}_home_win"].astype(int)
        y_hm = df.loc[mask, f"target_{half}_margin"].astype(float)
        y_ht = df.loc[mask, f"target_{half}_total"].astype(float)
        clf_h = Pipeline([("scaler", StandardScaler()), ("logreg", LogisticRegression(solver="saga", max_iter=5000, C=best_c))]).fit(Xh, y_hw)
        reg_hm = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=5.0))]).fit(Xh, y_hm)
        reg_ht = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=5.0))]).fit(Xh, y_ht)
        models_halves[half] = {"win": clf_h, "margin": reg_hm, "total": reg_ht}

    # Quarter models
    models_quarters = {}
    for q in ("q1", "q2", "q3", "q4"):
        cols = [f"target_{q}_home_win", f"target_{q}_margin", f"target_{q}_total"]
        if not all(c in df.columns for c in cols):
            continue
        mask = df[cols].notna().all(axis=1)
        if not mask.any():
            continue
        Xq = X.loc[mask]
        y_qw = df.loc[mask, f"target_{q}_home_win"].astype(int)
        y_qm = df.loc[mask, f"target_{q}_margin"].astype(float)
        y_qt = df.loc[mask, f"target_{q}_total"].astype(float)
        clf_q = Pipeline([("scaler", StandardScaler()), ("logreg", LogisticRegression(solver="saga", max_iter=5000, C=best_c))]).fit(Xq, y_qw)
        reg_qm = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=5.0))]).fit(Xq, y_qm)
        reg_qt = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=5.0))]).fit(Xq, y_qt)
        models_quarters[q] = {"win": clf_q, "margin": reg_qm, "total": reg_qt}

    # Save models and metrics
    paths.models.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, paths.models / "win_prob.joblib")
    joblib.dump(reg_margin, paths.models / "spread_margin.joblib")
    joblib.dump(reg_total, paths.models / "totals.joblib")
    # Persist the exact feature list used for training
    joblib.dump(use_feats, paths.models / "feature_columns.joblib")
    joblib.dump(models_halves, paths.models / "halves_models.joblib")
    joblib.dump(models_quarters, paths.models / "quarters_models.joblib")

    metrics = {
        "win_logloss_cv_mean": float(pd.Series(cv_losses).mean()) if cv_losses else None,
        "margin_rmse_cv_mean": float(pd.Series(cv_rmse_m).mean()) if cv_rmse_m else None,
        "total_rmse_cv_mean": float(pd.Series(cv_rmse_t).mean()) if cv_rmse_t else None,
    }
    return metrics
