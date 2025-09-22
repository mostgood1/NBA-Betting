from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import joblib
from typing import Dict, List

from .config import paths

TARGETS = ["t_pts","t_reb","t_ast","t_threes","t_pra"]


def _load_features() -> pd.DataFrame:
    p = paths.data_processed / "props_features.parquet"
    c = paths.data_processed / "props_features.csv"
    if p.exists():
        return pd.read_parquet(p)
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError("props_features not found; run build-props-features")


def train_props_models(alpha: float = 1.0) -> Dict[str, object]:
    df = _load_features()
    # Drop rows where targets are missing
    df = df.dropna(subset=TARGETS)
    # Feature columns: all roll/lag/min/b2b numeric features
    feat_cols = [c for c in df.columns if c.startswith("roll") or c.startswith("lag1_") or c in ("b2b",)]
    X = df[feat_cols].fillna(0.0).values
    models: Dict[str, Ridge] = {}
    for tgt in TARGETS:
        y = df[tgt].astype(float).values
        m = Ridge(alpha=alpha, random_state=42)
        m.fit(X, y)
        models[tgt] = m
    # Save models and feature columns
    paths.models.mkdir(parents=True, exist_ok=True)
    joblib.dump(feat_cols, paths.models / "props_feature_columns.joblib")
    joblib.dump(models, paths.models / "props_models.joblib")
    return models


def predict_props(features: pd.DataFrame) -> pd.DataFrame:
    feat_cols = joblib.load(paths.models / "props_feature_columns.joblib")
    models: Dict[str, Ridge] = joblib.load(paths.models / "props_models.joblib")
    X = features[feat_cols].fillna(0.0).values
    out = features.copy()
    for tgt, m in models.items():
        pred_col = tgt.replace("t_", "pred_")
        out[pred_col] = m.predict(X)
    return out
