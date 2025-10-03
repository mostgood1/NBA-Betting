from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

from .config import paths
from .props_train import predict_props
from .props_features import build_features_for_date
from .props_train import _load_features as _load_props_features  # reuse saved features for sigma calibration
from .odds_api import OddsApiConfig, backfill_player_props, fetch_player_props_current
from .odds_bovada import fetch_bovada_player_props_current


# Map OddsAPI player markets to our prediction columns
MARKET_TO_STAT = {
    "player_points": "pts",
    "player_rebounds": "reb",
    "player_assists": "ast",
    "player_three_pointers": "threes",
    "player_pr_points_rebounds_assists": "pra",
}


def _norm_name(s: str) -> str:
    if s is None:
        return ""
    t = str(s)
    # strip team suffixes in parentheses if present
    if "(" in t:
        t = t.split("(", 1)[0]
    t = t.replace(".", "").replace("'", "").strip()
    # remove common suffix tokens
    for suf in [" JR", " SR", " II", " III", " IV"]:
        if t.upper().endswith(suf):
            t = t[: -len(suf)]
    try:
        t = t.encode("ascii", "ignore").decode("ascii")
    except Exception:
        pass
    return t.upper().strip()


def _short_key(s: str) -> str:
    """Lightweight key: LASTNAME + FIRST_INITIAL, uppercase, ascii-only."""
    if not s:
        return ""
    s2 = _norm_name(s)
    parts = [p for p in s2.replace("-", " ").split() if p]
    if not parts:
        return s2
    last = parts[-1]
    first_initial = parts[0][0] if parts and parts[0] else ""
    return f"{last}{first_initial}"


def _american_implied_prob(price: float) -> float:
    try:
        a = float(price)
    except Exception:
        return np.nan
    if a > 0:
        return 100.0 / (a + 100.0)
    else:
        return (-a) / ((-a) + 100.0)


def _ev_per_unit(price: float, win_prob: float) -> float:
    # Expected value per 1 unit stake using American odds
    try:
        a = float(price)
        p = float(win_prob)
    except Exception:
        return np.nan
    if a > 0:
        return p * (a / 100.0) - (1 - p) * 1.0
    else:
        return p * (100.0 / (-a)) - (1 - p) * 1.0


@dataclass
class SigmaConfig:
    pts: float = 7.5
    reb: float = 3.0
    ast: float = 2.5
    threes: float = 1.3
    pra: float = 9.0


def _odds_for_date_from_saved(date: datetime) -> pd.DataFrame:
    # Load saved odds and filter to commence_time date
    raw_pq = paths.data_raw / "odds_nba_player_props.parquet"
    raw_csv = paths.data_raw / "odds_nba_player_props.csv"
    df = None
    if raw_pq.exists():
        try:
            df = pd.read_parquet(raw_pq)
        except Exception:
            df = None
    if df is None and raw_csv.exists():
        try:
            df = pd.read_csv(raw_csv)
        except Exception:
            df = None
    if df is None or df.empty:
        return pd.DataFrame()
    # Filter by commence_time date
    if "commence_time" in df.columns:
        dt = pd.to_datetime(df["commence_time"], errors="coerce")
        df = df.loc[dt.dt.date == pd.to_datetime(date).date()].copy()
    return df


def _fetch_odds_for_date(date: datetime, mode: str, api_key: Optional[str]) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame()
    cfg = OddsApiConfig(api_key=api_key)
    # Try historical first with a late timestamp on that date (UTC)
    if mode in ("auto", "historical"):
        ts = pd.Timestamp(pd.to_datetime(date).date()).to_pydatetime().replace(hour=23, minute=0, second=0)
        try:
            df = backfill_player_props(cfg, ts, verbose=False)
            # Filter to date in commence_time in case file had other days
            if df is not None and not df.empty:
                dt = pd.to_datetime(df["commence_time"], errors="coerce")
                df = df.loc[dt.dt.date == pd.to_datetime(date).date()].copy()
                if not df.empty:
                    return df
        except Exception:
            pass
        if mode == "historical":
            return pd.DataFrame()
    # Fallback to current event odds for the calendar date
    if mode in ("auto", "current"):
        try:
            df = fetch_player_props_current(cfg, pd.to_datetime(date))
            return df if df is not None else pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def compute_props_edges(date: str, sigma: SigmaConfig, use_saved: bool = True, mode: str = "auto", api_key: Optional[str] = None, source: str = "auto") -> pd.DataFrame:
    """Compute model edges/EV against OddsAPI player props for a given date.

    Returns a DataFrame with: date, player_name, stat, side, line, price, implied_prob, model_prob, edge, ev, bookmaker.
    """
    target_date = pd.to_datetime(date)
    # Predictions for slate (no odds filter yet)
    feats = build_features_for_date(target_date)
    preds = predict_props(feats)
    # Prepare prediction columns
    pred_map = {
        "pts": "pred_pts",
        "reb": "pred_reb",
        "ast": "pred_ast",
        "threes": "pred_threes",
        "pra": "pred_pra",
    }
    for need in ["player_id", "player_name"] + list(pred_map.values()):
        if need not in preds.columns:
            raise ValueError(f"Predictions missing column: {need}")

    preds = preds.copy()
    preds["name_key"] = preds["player_name"].astype(str).map(_norm_name)
    preds["short_key"] = preds["player_name"].astype(str).map(_short_key)

    # Load odds (OddsAPI or Bovada based on source)
    odds = pd.DataFrame()
    src = (source or "auto").lower()
    if src == "oddsapi":
        if use_saved:
            odds = _odds_for_date_from_saved(target_date)
        if odds is None or odds.empty:
            fetched = _fetch_odds_for_date(target_date, mode=mode, api_key=api_key)
            odds = fetched if fetched is not None else pd.DataFrame()
    elif src == "bovada":
        odds = fetch_bovada_player_props_current(target_date)
    else:
        # auto: try saved/current OddsAPI first, then fall back to Bovada
        if use_saved:
            odds = _odds_for_date_from_saved(target_date)
        if odds is None or odds.empty:
            fetched = _fetch_odds_for_date(target_date, mode=mode, api_key=api_key)
            odds = fetched if fetched is not None else pd.DataFrame()
        if odds is None or odds.empty:
            odds = fetch_bovada_player_props_current(target_date)
    if odds is None or odds.empty:
        return pd.DataFrame()

    # Normalize odds
    keep_cols = [
        "bookmaker", "bookmaker_title", "market", "outcome_name", "player_name", "point", "price", "commence_time"
    ]
    odds = odds[[c for c in keep_cols if c in odds.columns]].copy()
    # outcome_name is Over/Under, player_name may be in description
    if "player_name" not in odds.columns and "outcome_name" in odds.columns:
        # Some payloads put the player in outcome_name; try to salvage
        odds["player_name"] = odds["outcome_name"].astype(str)
    odds["name_key"] = odds["player_name"].astype(str).map(_norm_name)
    odds["short_key"] = odds["player_name"].astype(str).map(_short_key)
    odds["side"] = odds["outcome_name"].astype(str).str.upper().map(lambda x: "OVER" if "OVER" in x else ("UNDER" if "UNDER" in x else None))
    # Map markets to stat
    odds["stat"] = odds["market"].map(MARKET_TO_STAT)
    odds = odds.dropna(subset=["name_key", "stat", "side", "point", "price"]).copy()

    # Merge odds with predictions on name_key
    merged = odds.merge(
        preds[["name_key", "player_id", "player_name", "team", pred_map["pts"], pred_map["reb"], pred_map["ast"], pred_map["threes"], pred_map["pra"]]],
        on="name_key", how="left", suffixes=("", "_pred")
    )
    # Second-pass resolve using short key for any unmatched players
    unmatched = merged[merged["player_id"].isna()].copy()
    if not unmatched.empty:
        alt = odds.merge(
            preds[["short_key", "player_id", "player_name", "team", pred_map["pts"], pred_map["reb"], pred_map["ast"], pred_map["threes"], pred_map["pra"]]].rename(columns={"short_key": "short_key_pred"}),
            left_on="short_key", right_on="short_key_pred", how="left"
        )
        alt = alt[["short_key", "player_id", "player_name", "team", pred_map["pts"], pred_map["reb"], pred_map["ast"], pred_map["threes"], pred_map["pra"]]].copy()
        merged = merged.merge(alt.add_suffix("_alt"), left_on="short_key", right_on="short_key_alt", how="left")
        # Fill missing fields from alt
        for col in ["player_id", "player_name", "team", "model_mean"]:
            if col == "model_mean":
                # will be computed after selecting stat
                continue
            base_col = col
            alt_col = f"{col}_alt"
            if base_col in merged.columns and alt_col in merged.columns:
                merged[base_col] = merged[base_col].fillna(merged[alt_col])
    # Choose model mean based on stat
    def _select_pred(row) -> float:
        stat = row["stat"]
        col = pred_map.get(stat)
        val = row.get(col, np.nan)
        if pd.isna(val):
            # try alt merged columns
            alt_col = f"{col}_alt"
            return row.get(alt_col, np.nan)
        return val

    merged["model_mean"] = merged.apply(_select_pred, axis=1)
    # Sigma by stat
    sig_map = {"pts": sigma.pts, "reb": sigma.reb, "ast": sigma.ast, "threes": sigma.threes, "pra": sigma.pra}
    merged["sigma"] = merged["stat"].map(sig_map)

    # Model probability for Over: P(X > line) under Normal(mean, sigma)
    from math import erf, sqrt

    def _norm_cdf(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    def _prob_over(mean, sigma, line):
        if pd.isna(mean) or pd.isna(sigma) or pd.isna(line) or sigma <= 0:
            return np.nan
        z = (line - mean) / sigma
        # P(X > line) = 1 - CDF(line)
        return 1.0 - _norm_cdf(z)

    merged["line"] = pd.to_numeric(merged["point"], errors="coerce")
    merged["price"] = pd.to_numeric(merged["price"], errors="coerce")
    merged["p_over"] = merged.apply(lambda r: _prob_over(r["model_mean"], r["sigma"], r["line"]), axis=1)
    merged["model_prob"] = merged.apply(lambda r: (1.0 - r["p_over"]) if r["side"] == "UNDER" else r["p_over"], axis=1)
    merged["implied_prob"] = merged["price"].map(_american_implied_prob)
    merged["edge"] = merged["model_prob"] - merged["implied_prob"]
    merged["ev"] = merged.apply(lambda r: _ev_per_unit(r["price"], r["model_prob"]), axis=1)

    out_cols = [
        "player_id", "player_name", "team", "stat", "side", "line", "price", "implied_prob", "model_prob", "edge", "ev", "bookmaker", "bookmaker_title", "commence_time"
    ]
    out = merged[out_cols].copy()
    out.insert(0, "date", pd.to_datetime(target_date).date())
    out.sort_values(["stat", "edge"], ascending=[True, False], inplace=True)
    return out


def calibrate_sigma_for_date(date: str, window_days: int = 30, min_rows: int = 200, defaults: Optional[SigmaConfig] = None) -> SigmaConfig:
    """Estimate sigma per stat from recent residuals in props_features.

    Uses saved features with actual targets (t_*) over [date - window_days, date - 1],
    predicts with current models, and computes stddev of residuals per stat.
    Falls back to defaults if not enough rows.
    """
    if defaults is None:
        defaults = SigmaConfig()
    try:
        df = _load_props_features().copy()
    except Exception:
        return defaults
    if "date" not in df.columns:
        return defaults
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    end = pd.to_datetime(date)
    start = end - pd.Timedelta(days=window_days)
    hist = df[(df["date"] >= start) & (df["date"] < end)].copy()
    if hist.empty:
        return defaults
    # Predict on this window
    try:
        preds = predict_props(hist)
    except Exception:
        return defaults
    stats = {
        "pts": ("t_pts", "pred_pts"),
        "reb": ("t_reb", "pred_reb"),
        "ast": ("t_ast", "pred_ast"),
        "threes": ("t_threes", "pred_threes"),
        "pra": ("t_pra", "pred_pra"),
    }
    sig = {}
    for k, (tgt, pr) in stats.items():
        if tgt in preds.columns and pr in preds.columns:
            y = pd.to_numeric(preds[tgt], errors="coerce")
            p = pd.to_numeric(preds[pr], errors="coerce")
            m = y.notna() & p.notna()
            if m.sum() >= min_rows:
                sig[k] = float((y[m] - p[m]).std(ddof=1))
    return SigmaConfig(
        pts=sig.get("pts", defaults.pts),
        reb=sig.get("reb", defaults.reb),
        ast=sig.get("ast", defaults.ast),
        threes=sig.get("threes", defaults.threes),
        pra=sig.get("pra", defaults.pra),
    )
