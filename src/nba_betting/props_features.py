from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Iterable, Dict, List

from .config import paths


NUM_COL_MAP = {
    "PTS": ["PTS", "pts"],
    "REB": ["REB", "reb", "TREB", "treb"],
    "AST": ["AST", "ast"],
    "FG3M": ["FG3M", "fg3m", "FG3M_A"],
    "MIN": ["MIN", "min"],
}

DATE_COLS = ["GAME_DATE", "GAME_DATE_EST", "dateGame", "GAME_DATE_PT"]
PLAYER_ID_COLS = ["PLAYER_ID", "player_id", "idPlayer"]
PLAYER_NAME_COLS = ["PLAYER_NAME", "player_name", "namePlayer"]
GAME_ID_COLS = ["GAME_ID", "game_id", "idGame"]
TEAM_COLS = ["TEAM_ABBREVIATION", "team", "slugTeam"]
MATCHUP_COL = "MATCHUP"


def _find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k.lower() in cols:
            return cols[k.lower()]
    return None


def _to_minutes(v) -> float | None:
    if pd.isna(v):
        return np.nan
    s = str(v)
    if s.isdigit():
        return float(s)
    if ":" in s:
        try:
            mm, ss = s.split(":", 1)
            return float(int(mm) + int(ss) / 60.0)
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def load_player_logs() -> pd.DataFrame:
    # Prefer parquet then CSV
    p = paths.data_processed / "player_logs.parquet"
    c = paths.data_processed / "player_logs.csv"
    if p.exists():
        return pd.read_parquet(p)
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError("player_logs not found; run fetch-player-logs")


def build_props_features(windows: List[int] = [3, 5, 10]) -> pd.DataFrame:
    """Compute per-player rolling features and targets from player logs.

    Outputs a row per player's game with lagged rolling stats up to previous game.
    Targets: pts, reb, ast, threes (FG3M), pra for that game.
    """
    logs = load_player_logs().copy()
    # Identify columns
    pid = _find_col(logs, PLAYER_ID_COLS)
    pname = _find_col(logs, PLAYER_NAME_COLS)
    gid = _find_col(logs, GAME_ID_COLS)
    tcol = _find_col(logs, TEAM_COLS)
    dcol = _find_col(logs, DATE_COLS)
    pts = _find_col(logs, NUM_COL_MAP["PTS"]) ; reb = _find_col(logs, NUM_COL_MAP["REB"]) ; ast = _find_col(logs, NUM_COL_MAP["AST"]) ; fg3m = _find_col(logs, NUM_COL_MAP["FG3M"]) ; minc = _find_col(logs, NUM_COL_MAP["MIN"]) 
    for col in [pid, pname, gid, tcol, dcol, pts, reb, ast, fg3m]:
        if col is None:
            raise ValueError("Missing required columns in player_logs")
    logs[dcol] = pd.to_datetime(logs[dcol])
    logs.sort_values([pid, dcol], inplace=True)
    # Numeric conversions
    for c in [pts, reb, ast, fg3m]:
        logs[c] = pd.to_numeric(logs[c], errors="coerce")
    if minc is not None and minc in logs.columns:
        logs[minc] = logs[minc].apply(_to_minutes)
    else:
        logs[minc] = np.nan

    # Create PRA
    logs["_PRA"] = logs[[pts, reb, ast]].sum(axis=1, skipna=True)

    feats = []
    # Group by player and compute rolling windows; then shift by 1 to avoid leakage
    grp = logs.groupby(pid, sort=False)
    base_cols = {"player_id": pid, "player_name": pname, "game_id": gid, "team": tcol, "date": dcol}
    for p, g in grp:
        g = g.copy()
        g["minutes"] = g[minc]
        # rolling for numeric stats
        for w in windows:
            g[f"roll{w}_pts"] = g[pts].rolling(w, min_periods=1).mean().shift(1)
            g[f"roll{w}_reb"] = g[reb].rolling(w, min_periods=1).mean().shift(1)
            g[f"roll{w}_ast"] = g[ast].rolling(w, min_periods=1).mean().shift(1)
            g[f"roll{w}_threes"] = g[fg3m].rolling(w, min_periods=1).mean().shift(1)
            g[f"roll{w}_min"] = g["minutes"].rolling(w, min_periods=1).mean().shift(1)
        # simple last-game stats
        g["lag1_pts"] = g[pts].shift(1)
        g["lag1_reb"] = g[reb].shift(1)
        g["lag1_ast"] = g[ast].shift(1)
        g["lag1_threes"] = g[fg3m].shift(1)
        g["lag1_min"] = g["minutes"].shift(1)
        # b2b indicator: played previous day
        g["b2b"] = (g[dcol].diff().dt.days == 1).shift(0).astype(float)
        # targets for this game
        g["t_pts"] = g[pts]
        g["t_reb"] = g[reb]
        g["t_ast"] = g[ast]
        g["t_threes"] = g[fg3m]
        g["t_pra"] = g[[pts, reb, ast]].sum(axis=1, skipna=True)
        # append
        keep = list(base_cols.values()) + [
            "b2b",
            "lag1_pts","lag1_reb","lag1_ast","lag1_threes","lag1_min",
        ] + [f"roll{w}_{x}" for w in windows for x in ("pts","reb","ast","threes","min")] + [
            "t_pts","t_reb","t_ast","t_threes","t_pra"
        ]
        feats.append(g[keep].rename(columns={
            base_cols["player_id"]: "player_id",
            base_cols["player_name"]: "player_name",
            base_cols["game_id"]: "game_id",
            base_cols["team"]: "team",
            base_cols["date"]: "date",
        }))
    out = pd.concat(feats, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.date
    # Save
    out_path = paths.data_processed / "props_features.parquet"
    out.to_parquet(out_path, index=False)
    out_csv = paths.data_processed / "props_features.csv"
    out.to_csv(out_csv, index=False)
    return out


def build_features_for_date(date: str | pd.Timestamp, windows: List[int] = [3,5,10], players: List[int] | None = None) -> pd.DataFrame:
    """Build per-player features up to the day before the given date (no leakage).

    If players provided, filter to those player_ids; otherwise return all seen players.
    """
    logs = load_player_logs().copy()
    dcol = _find_col(logs, DATE_COLS)
    pid = _find_col(logs, PLAYER_ID_COLS)
    pname = _find_col(logs, PLAYER_NAME_COLS)
    tcol = _find_col(logs, TEAM_COLS)
    pts = _find_col(logs, NUM_COL_MAP["PTS"]) ; reb = _find_col(logs, NUM_COL_MAP["REB"]) ; ast = _find_col(logs, NUM_COL_MAP["AST"]) ; fg3m = _find_col(logs, NUM_COL_MAP["FG3M"]) ; minc = _find_col(logs, NUM_COL_MAP["MIN"]) 
    for col in [pid, pname, tcol, dcol, pts, reb, ast, fg3m]:
        if col is None:
            raise ValueError("Missing required columns in player_logs")
    logs[dcol] = pd.to_datetime(logs[dcol])
    # Only prior to date
    target_date = pd.to_datetime(date)
    hist = logs[logs[dcol] < target_date].copy()
    if players is not None:
        hist = hist[hist[pid].isin(players)].copy()
    # Convert numerics
    for c in [pts, reb, ast, fg3m]:
        hist[c] = pd.to_numeric(hist[c], errors="coerce")
    if minc is not None and minc in hist.columns:
        hist[minc] = hist[minc].apply(_to_minutes)
    else:
        hist[minc] = np.nan
    hist.sort_values([pid, dcol], inplace=True)
    rows = []
    grp = hist.groupby(pid, sort=False)
    for p, g in grp:
        g = g.copy()
        g["minutes"] = g[minc]
        rec = {
            "player_id": p,
            "player_name": g.iloc[-1][_find_col(hist, PLAYER_NAME_COLS)] if _find_col(hist, PLAYER_NAME_COLS) else None,
            "team": g.iloc[-1][_find_col(hist, TEAM_COLS)] if _find_col(hist, TEAM_COLS) else None,
            "asof_date": target_date.date(),
        }
        rec["lag1_pts"] = g[pts].iloc[-1] if len(g) > 0 else np.nan
        rec["lag1_reb"] = g[reb].iloc[-1] if len(g) > 0 else np.nan
        rec["lag1_ast"] = g[ast].iloc[-1] if len(g) > 0 else np.nan
        rec["lag1_threes"] = g[fg3m].iloc[-1] if len(g) > 0 else np.nan
        rec["lag1_min"] = g["minutes"].iloc[-1] if len(g) > 0 else np.nan
        # b2b based on last two games
        if len(g) >= 2:
            d1 = g[dcol].iloc[-1]; d0 = g[dcol].iloc[-2]
            rec["b2b"] = float((d1 - d0).days == 1)
        else:
            rec["b2b"] = 0.0
        for w in windows:
            rec[f"roll{w}_pts"] = g[pts].rolling(w, min_periods=1).mean().iloc[-1]
            rec[f"roll{w}_reb"] = g[reb].rolling(w, min_periods=1).mean().iloc[-1]
            rec[f"roll{w}_ast"] = g[ast].rolling(w, min_periods=1).mean().iloc[-1]
            rec[f"roll{w}_threes"] = g[fg3m].rolling(w, min_periods=1).mean().iloc[-1]
            rec[f"roll{w}_min"] = g["minutes"].rolling(w, min_periods=1).mean().iloc[-1]
        rows.append(rec)
    return pd.DataFrame(rows)
