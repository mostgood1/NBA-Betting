from __future__ import annotations

import click
import os
import itertools

# Reduce noisy ONNXRuntime logging on some platforms (e.g., Windows ARM).
# Safe to set even if ignored by the runtime.
os.environ.setdefault("ONNXRUNTIME_LOG_SEVERITY_LEVEL", "3")
os.environ.setdefault("ORT_DISABLE_CPUINFO", "1")

import pandas as pd
import numpy as np
import subprocess
from rich.console import Console
import re
from rich.progress import track

from .config import paths, reconcile_repo_data_to_active
# from .scrape_bref import scrape_games  # deprecated
from .features import build_features
# from .train import train_models  # MOVED TO CONDITIONAL IMPORT - requires sklearn
import joblib
from .elo import Elo
from .schedule import compute_rest_for_matchups, fetch_schedule_2025_26
from .rosters import fetch_rosters
from .league_status import build_league_status
from .availability import build_and_check_dressed_players
from .roster_audit import audit_roster_for_date
from .cards_settlement_audit import (
    audit_playable_prop_aliases,
    audit_playable_prop_coverage_gaps,
    audit_playable_prop_provider_anomalies,
    audit_playable_prop_settlement,
    build_playable_prop_alias_rollups,
    build_playable_prop_coverage_rollups,
    build_playable_prop_provider_rollups,
    build_playable_prop_settlement_rollups,
    default_playable_prop_alias_audit_path,
    default_playable_prop_alias_rollup_path,
    default_playable_prop_coverage_audit_path,
    default_playable_prop_coverage_rollup_path,
    default_playable_prop_provider_audit_path,
    default_playable_prop_provider_rollup_path,
    default_playable_prop_settlement_audit_path,
    default_playable_prop_settlement_rollup_path,
)
from .roster_checks import roster_sanity_check
from .roster_files import pick_rosters_file
from .player_logs import fetch_player_logs
from .player_names import normalize_player_name_key
from .teams import normalize_team, to_tricode
from .scrape_nba_api import fetch_games_nba_api, enrich_periods_existing, backfill_scoreboard
from .odds_api import backfill_historical_odds, OddsApiConfig, consensus_lines_at_close, backfill_player_props, fetch_player_props_current
from .odds_api import filter_player_prop_bookmakers_df, resolve_player_prop_bookmakers, player_prop_bookmakers_csv
from .pbp_markets import train_all_pbp_markets, predict_tip_for_date, predict_first_basket_for_date, predict_early_threes_for_date
from .odds_api import fetch_game_odds_current
from .odds_bovada import fetch_bovada_odds_current
from .props_actuals import fetch_prop_actuals_via_nbastatr, upsert_props_actuals
from .props_actuals import fetch_prop_actuals_via_nba_cdn, fetch_prop_actuals_via_nbaapi
from .props_features import build_props_features, build_features_for_date
from .finals import fetch_finals, write_finals_csv
from .pbp import fetch_pbp_for_date, backfill_pbp
from .boxscores import fetch_boxscores_for_date, backfill_boxscores, update_boxscores_history_for_date
from .pbp_espn import fetch_pbp_espn_for_date, update_pbp_espn_history_for_date, backfill_pbp_espn_history
from .rotation_priors import write_rotation_priors
from .rotations_espn import update_rotations_history_for_date
from .lineup_context_features import build_lineup_teammate_effects
# from .props_train import train_props_models, predict_props  # MOVED TO CONDITIONAL - requires sklearn
from .props_edges import compute_props_edges, SigmaConfig, calibrate_sigma_for_date
from .props_linear import train_linear_props_models, export_linear_to_onnx
from .props_backtest import backtest_linear_props
from .props_edges_backtest import BacktestConfig as PropsEdgesBacktestConfig, backtest_props_edges
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.endpoints import boxscoretraditionalv3
from nba_api.stats.library import http as nba_http
from nba_api.stats.static import teams as static_teams
import subprocess
from pathlib import Path
import sys
import time
from typing import Optional, Any
from datetime import date as _date

from .pbp_markets import _game_ids_for_date as _pbp_game_ids_for_date  # reuse for backtest
from .pbp_markets import _first_fg_event as _pbp_first_fg_event
from .pbp_markets import _jump_ball_event as _pbp_jump_ball_event
from .pbp_markets import _desc_cols as _pbp_desc_cols
from .pbp_markets import build_early_threes_dataset as _build_early_threes_dataset

console = Console()

# When running on Render with NBA_BETTING_DATA_ROOT pointing at a persistent disk,
# reconcile repo-committed artifacts into the active data root so deploys don't
# leave the disk stale.
try:
    reconcile_repo_data_to_active()
except Exception:
    pass


def _truthy_mask(values: pd.Series | Any) -> pd.Series:
    try:
        ser = values if isinstance(values, pd.Series) else pd.Series(values)
    except Exception:
        return pd.Series(dtype=bool)
    if len(ser) == 0:
        return pd.Series(dtype=bool)
    txt = ser.astype(str).str.strip().str.lower()
    out = txt.isin({"1", "true", "t", "yes", "y"})
    try:
        num = pd.to_numeric(ser, errors="coerce")
        out = out | ((num > 0.5) & num.notna())
    except Exception:
        pass
    return out.astype(bool)


def _falsey_mask(values: pd.Series | Any) -> pd.Series:
    try:
        ser = values if isinstance(values, pd.Series) else pd.Series(values)
    except Exception:
        return pd.Series(dtype=bool)
    if len(ser) == 0:
        return pd.Series(dtype=bool)
    txt = ser.astype(str).str.strip().str.lower()
    out = txt.isin({"0", "false", "f", "no", "n"})
    try:
        num = pd.to_numeric(ser, errors="coerce")
        out = out | ((num <= 0.5) & num.notna())
    except Exception:
        pass
    return out.astype(bool)


def _league_status_player_maps_for_date(
    date_str: str,
    *,
    allow_build: bool = True,
) -> tuple[dict[str, str], dict[str, set[str]], dict[str, set[str]]]:
    roster_name_to_tri: dict[str, str] = {}
    excluded_by_team: dict[str, set[str]] = {}
    allowed_by_team: dict[str, set[str]] = {}

    try:
        ls_path = paths.data_processed / f"league_status_{str(date_str).strip()}.csv"
        if ls_path.exists():
            lsdf = pd.read_csv(ls_path)
        elif allow_build:
            lsdf = build_league_status(str(date_str))
        else:
            lsdf = pd.DataFrame()
    except Exception:
        lsdf = pd.DataFrame()

    if lsdf is None or lsdf.empty:
        return roster_name_to_tri, excluded_by_team, allowed_by_team

    def _backfill_missing_roster_names() -> None:
        try:
            dts = pd.to_datetime(date_str, errors="coerce")
            season_start = None
            if not pd.isna(dts):
                season_start = int(dts.year) if int(dts.month) >= 7 else int(dts.year) - 1
            season_label = None
            if season_start is not None:
                season_label = f"{int(season_start)}-{str(int(season_start) + 1)[-2:]}"

            fp = pick_rosters_file(paths.data_processed, season=season_label)
            if fp is None or not fp.exists():
                return

            rdf = pd.read_csv(fp)
            if rdf is None or rdf.empty:
                return

            cols = {c.upper(): c for c in rdf.columns}
            tcol = cols.get("TEAM_ABBREVIATION") or cols.get("TEAM") or cols.get("TEAM_TRI")
            ncol = cols.get("PLAYER") or cols.get("PLAYER_NAME")
            if not (tcol and ncol):
                return

            tmp = rdf[[tcol, ncol]].dropna().copy()
            tmp[tcol] = tmp[tcol].astype(str).str.strip().str.upper()
            tmp[ncol] = tmp[ncol].astype(str)
            for _, rr in tmp.iterrows():
                pkey = str(normalize_player_name_key(rr.get(ncol), case="upper") or "").strip().upper()
                if not pkey or pkey in roster_name_to_tri:
                    continue
                tri = str(rr.get(tcol) or "").strip().upper()
                if len(tri) != 3:
                    tri = str(to_tricode(tri) or "").strip().upper()
                if tri:
                    roster_name_to_tri[pkey] = tri
        except Exception:
            return

    try:
        cols_upper = {c.upper(): c for c in lsdf.columns}
        cols_lower = {c.lower(): c for c in lsdf.columns}
        team_col = (
            cols_upper.get("TEAM_ABBREVIATION")
            or cols_upper.get("TEAM")
            or cols_upper.get("TEAM_TRI")
            or cols_lower.get("team_abbreviation")
            or cols_lower.get("team")
            or cols_lower.get("team_tri")
        )
        name_col = (
            cols_upper.get("PLAYER_NAME")
            or cols_upper.get("PLAYER")
            or cols_lower.get("player_name")
            or cols_lower.get("player")
        )
        playing_col = cols_lower.get("playing_today") or cols_upper.get("PLAYING_TODAY")
        on_slate_col = cols_lower.get("team_on_slate") or cols_upper.get("TEAM_ON_SLATE")

        if not (team_col and name_col):
            return roster_name_to_tri, excluded_by_team, allowed_by_team

        keep_cols = [team_col, name_col]
        if playing_col:
            keep_cols.append(playing_col)
        if on_slate_col:
            keep_cols.append(on_slate_col)
        tmp = lsdf[keep_cols].copy()
        tmp[team_col] = tmp[team_col].astype(str).map(lambda x: (to_tricode(str(x)) or str(x).strip().upper()))
        tmp[name_col] = tmp[name_col].astype(str)
        tmp["_pkey"] = tmp[name_col].map(lambda x: normalize_player_name_key(x, case="upper"))
        tmp = tmp[tmp[team_col].astype(str).str.len() > 0].copy()
        tmp = tmp[tmp["_pkey"].astype(str).str.len() > 0].copy()
        if tmp.empty:
            return roster_name_to_tri, excluded_by_team, allowed_by_team

        for _, rr in tmp[[team_col, "_pkey"]].dropna().iterrows():
            tri = str(rr.get(team_col) or "").strip().upper()
            pkey = str(rr.get("_pkey") or "").strip().upper()
            if tri and pkey:
                roster_name_to_tri.setdefault(pkey, tri)

        _backfill_missing_roster_names()

        on_mask = pd.Series(True, index=tmp.index)
        if on_slate_col and on_slate_col in tmp.columns:
            on_mask = _truthy_mask(tmp[on_slate_col]).reindex(tmp.index, fill_value=False)

        if playing_col and playing_col in tmp.columns:
            playing_true = _truthy_mask(tmp[playing_col]).reindex(tmp.index, fill_value=False)
            playing_false = _falsey_mask(tmp[playing_col]).reindex(tmp.index, fill_value=False)
        else:
            playing_true = pd.Series(False, index=tmp.index)
            playing_false = pd.Series(False, index=tmp.index)

        allow_df = tmp[on_mask & playing_true].copy()
        exclude_df = tmp[on_mask & playing_false].copy()

        for tri, grp in allow_df.groupby(team_col):
            key = str(tri or "").strip().upper()
            vals = set(grp["_pkey"].astype(str).str.upper().str.strip().tolist())
            if key and vals:
                allowed_by_team[key] = vals

        for tri, grp in exclude_df.groupby(team_col):
            key = str(tri or "").strip().upper()
            vals = set(grp["_pkey"].astype(str).str.upper().str.strip().tolist())
            if key and vals:
                excluded_by_team[key] = vals
    except Exception:
        return {}, {}, {}

    return roster_name_to_tri, excluded_by_team, allowed_by_team


def _smart_sim_injuries_excluded_map_for_date(
    ds: str,
    *,
    props_df: pd.DataFrame | None = None,
) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}

    def _norm_player_key(x: Any) -> str:
        return str(normalize_player_name_key(x, case="upper") or "").strip().upper()

    def _add(tri: str, name: str) -> None:
        t = str(tri or "").strip().upper()
        if not t:
            return
        k = _norm_player_key(name)
        if not k:
            return
        out.setdefault(t, set()).add(k)

    ds_s = str(ds).strip()

    try:
        cutoff_dt = pd.to_datetime(ds_s, errors="coerce")
        cutoff = cutoff_dt if pd.notna(cutoff_dt) else None
    except Exception:
        cutoff = None

    recency_days = 30
    try:
        fresh_cutoff = (cutoff - pd.Timedelta(days=int(recency_days))) if cutoff is not None else None
    except Exception:
        fresh_cutoff = None

    roster_name_to_tri: dict[str, str] = {}
    ls_allowed: dict[str, set[str]] = {}
    try:
        roster_name_to_tri, ls_excluded, ls_allowed = _league_status_player_maps_for_date(ds_s)
        for tri, names in ls_excluded.items():
            for name_key in names:
                if tri and name_key:
                    out.setdefault(str(tri).strip().upper(), set()).add(str(name_key).strip().upper())
    except Exception:
        roster_name_to_tri = {}
        ls_allowed = {}

    if not roster_name_to_tri:
        try:
            dts = pd.to_datetime(ds_s, errors="coerce")
            season_start = None
            if not pd.isna(dts):
                season_start = int(dts.year) if int(dts.month) >= 7 else int(dts.year) - 1
            candidates: list[Path] = []
            if season_start is not None:
                label = f"{int(season_start)}-{str(int(season_start) + 1)[-2:]}"
                candidates.append(paths.data_processed / f"rosters_{label}.csv")
                candidates.append(paths.data_processed / f"rosters_{int(season_start)}.csv")
            candidates.extend(sorted(paths.data_processed.glob("rosters_*.csv")))

            seen: set[str] = set()
            for fp in candidates:
                sp = str(fp)
                if sp in seen:
                    continue
                seen.add(sp)
                if not fp.exists():
                    continue
                rdf = pd.read_csv(fp)
                if rdf is None or rdf.empty:
                    continue
                cols = {c.upper(): c for c in rdf.columns}
                tcol = cols.get("TEAM_ABBREVIATION") or cols.get("TEAM") or cols.get("TEAM_TRI")
                ncol = cols.get("PLAYER") or cols.get("PLAYER_NAME")
                if not (tcol and ncol):
                    continue
                tmp = rdf[[tcol, ncol]].dropna().copy()
                tmp[tcol] = tmp[tcol].astype(str).str.strip().str.upper()
                tmp[ncol] = tmp[ncol].astype(str)
                for _, rr in tmp.iterrows():
                    nk = _norm_player_key(rr.get(ncol))
                    if not nk:
                        continue
                    tri = str(rr.get(tcol) or "").strip().upper()
                    if len(tri) != 3:
                        tri = str(to_tricode(tri) or "").strip().upper()
                    if tri:
                        roster_name_to_tri.setdefault(nk, tri)
                if roster_name_to_tri:
                    break
        except Exception:
            pass

    try:
        p = paths.data_processed / f"injuries_excluded_{ds_s}.csv"
        if p.exists():
            df = pd.read_csv(p)
            if df is not None and not df.empty:
                try:
                    if "date" in df.columns:
                        df = df.copy()
                        df["date"] = pd.to_datetime(df["date"], errors="coerce")
                        df = df[df["date"].notna()].copy()
                        if cutoff is not None:
                            df = df[df["date"] <= cutoff].copy()
                        if fresh_cutoff is not None:
                            st = df.get("status", "").astype(str).str.upper().str.strip() if "status" in df.columns else pd.Series([""] * len(df))
                            inj = df.get("injury", "").astype(str).str.upper().str.strip() if "injury" in df.columns else pd.Series([""] * len(df))
                            is_season = st.str.contains("SEASON", na=False) | st.str.contains("INDEFINITE", na=False) | st.str.contains("SEASON-ENDING", na=False)
                            is_season = is_season | inj.str.contains("OUT FOR SEASON", na=False) | inj.str.contains("SEASON-ENDING", na=False) | inj.str.contains("INDEFINITE", na=False)
                            df = df[(df["date"] >= fresh_cutoff) | is_season].copy()
                except Exception:
                    pass

                try:
                    if "status" in df.columns:
                        excl = {"OUT", "DOUBTFUL", "SUSPENDED", "INACTIVE", "REST"}
                        st = df["status"].astype(str).str.upper().str.strip()
                        season_out = (st.str.contains("SEASON", na=False) & st.str.contains("OUT", na=False)) | st.str.contains("INDEFINITE", na=False) | st.str.contains("SEASON-ENDING", na=False)
                        df = df[st.isin(excl) | season_out].copy()
                except Exception:
                    pass

                tcol = "team_tri" if "team_tri" in df.columns else ("team" if "team" in df.columns else None)
                ncol = "player" if "player" in df.columns else ("player_name" if "player_name" in df.columns else None)
                if tcol and ncol:
                    for _, r in df[[tcol, ncol]].dropna().iterrows():
                        _add(str(r.get(tcol) or ""), str(r.get(ncol) or ""))
    except Exception:
        pass

    try:
        raw = paths.data_raw / "injuries.csv"
        if raw.exists():
            df = pd.read_csv(raw)
            if df is not None and not df.empty:
                if "date" in df.columns:
                    df = df.copy()
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    df = df[df["date"].notna()].copy()
                    if cutoff is not None:
                        df = df[df["date"] <= cutoff].copy()
                    try:
                        if fresh_cutoff is not None:
                            st0 = df.get("status", "").astype(str).str.upper().str.strip() if "status" in df.columns else pd.Series([""] * len(df))
                            season_out0 = (st0.str.contains("SEASON", na=False) & st0.str.contains("OUT", na=False)) | st0.str.contains("INDEFINITE", na=False) | st0.str.contains("SEASON-ENDING", na=False)
                            df = df[(df["date"] >= fresh_cutoff) | season_out0].copy()
                    except Exception:
                        pass
                    try:
                        df = df.sort_values(["date"]).copy()
                        grp_cols = [c for c in ["player", "team"] if c in df.columns]
                        if grp_cols:
                            df = df.groupby(grp_cols, as_index=False).tail(1).copy()
                    except Exception:
                        pass
                status_col = "status" if "status" in df.columns else ("injury_status" if "injury_status" in df.columns else None)
                name_col = "player" if "player" in df.columns else ("player_name" if "player_name" in df.columns else None)
                team_col = "team" if "team" in df.columns else ("team_tri" if "team_tri" in df.columns else None)
                if status_col and name_col and team_col:
                    excl = {"OUT", "DOUBTFUL", "SUSPENDED", "INACTIVE", "REST"}
                    st = df[status_col].astype(str).str.upper().str.strip()
                    season_out = (st.str.contains("SEASON", na=False) & st.str.contains("OUT", na=False)) | st.str.contains("INDEFINITE", na=False) | st.str.contains("SEASON-ENDING", na=False)
                    df = df[st.isin(excl) | season_out].copy()
                    for _, r in df[[team_col, name_col]].dropna().iterrows():
                        nm = str(r.get(name_col) or "")
                        nk = _norm_player_key(nm)
                        tri = str(r.get(team_col) or "")
                        if nk and (nk in roster_name_to_tri):
                            tri = roster_name_to_tri.get(nk) or tri
                        _add(str(tri or ""), nm)
    except Exception:
        pass

    try:
        if isinstance(props_df, pd.DataFrame) and (not props_df.empty):
            if "team" in props_df.columns and "player_name" in props_df.columns:
                tmp = props_df[["team", "player_name"] + (["playing_today"] if "playing_today" in props_df.columns else [])].copy()
                tmp["team"] = tmp["team"].astype(str).str.strip().str.upper()
                tmp["player_name"] = tmp["player_name"].astype(str).str.strip()
                if "playing_today" in tmp.columns:
                    tmp = tmp[_truthy_mask(tmp["playing_today"]).reindex(tmp.index, fill_value=False)].copy()
                tmp = tmp[tmp["player_name"].ne("")].copy()
                for _, rr in tmp.iterrows():
                    tri = str(to_tricode(rr.get("team")) or rr.get("team") or "").strip().upper()
                    if not tri:
                        continue
                    k = _norm_player_key(rr.get("player_name"))
                    if not k:
                        continue
                    if tri in out and k in out[tri]:
                        out[tri].discard(k)
    except Exception:
        pass

    try:
        for tri, names in (ls_allowed or {}).items():
            t = str(tri or "").strip().upper()
            if not t or t not in out:
                continue
            out[t].difference_update(set(str(x or "").strip().upper() for x in names if str(x or "").strip()))
            if not out[t]:
                out.pop(t, None)
    except Exception:
        pass

    return out

# --- Calibration config helpers ---
def _load_player_calib_overrides():
    """Load per-stat player calibration overrides from processed config JSON.

    File format (example):
    {
      "updated_at": "2025-11-14",
      "window_days": 14,
      "criterion": "mae",
      "per_stat": {
        "pts": {"K": 8, "min_pairs": 6},
        "reb": {"K": 12, "min_pairs": 8}
      }
    }
    Returns: (k_map: dict[str,float], n_map: dict[str,int]) or (None, None)
    """
    try:
        import json as _json
        p = paths.data_processed / "props_player_calibration_config.json"
        if not p.exists():
            return None, None
        with open(p, "r", encoding="utf-8") as f:
            cfg = _json.load(f)
        ps = cfg.get("per_stat") or {}
        k_map: dict[str, float] = {}
        n_map: dict[str, int] = {}
        for k, v in ps.items():
            try:
                k_map[str(k).lower()] = float(v.get("K"))
            except Exception:
                pass
            try:
                n_map[str(k).lower()] = int(v.get("min_pairs"))
            except Exception:
                pass
        if not k_map and not n_map:
            return None, None
        return (k_map if k_map else None), (n_map if n_map else None)
    except Exception:
        return None, None


 


@click.group()
def cli():
    """NBA Betting command-line interface."""
    pass


def _load_dotenv_key(name: str) -> str | None:
    """Lightweight .env reader: looks for KEY=VALUE in a .env at repo root."""
    try:
        env_path = paths.root / ".env"
        if not env_path.exists():
            return None
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == name:
                return v.strip()
        return None
    except Exception:
        return None


def _us_slate_date() -> _date:
    """Return the current NBA slate date using US local time, not UTC."""
    import datetime as _dt

    tz_name = str(os.environ.get("APP_TZ") or "America/New_York").strip() or "America/New_York"
    cutoff_hour = 6
    try:
        from zoneinfo import ZoneInfo

        now_local = _dt.datetime.now(ZoneInfo(tz_name))
    except Exception:
        tz_map = {
            "America/New_York": -5,
            "US/Eastern": -5,
            "America/Chicago": -6,
            "US/Central": -6,
            "America/Denver": -7,
            "US/Mountain": -7,
            "America/Los_Angeles": -8,
            "US/Pacific": -8,
        }
        off = tz_map.get(tz_name)
        if off is None:
            try:
                off = int((os.environ.get("APP_TZ_OFFSET_HOURS") or "-5").strip())
            except Exception:
                off = -5
        now_local = _dt.datetime.utcnow() + _dt.timedelta(hours=off)
    if now_local.hour < cutoff_hour:
        now_local = now_local - _dt.timedelta(days=1)
    return now_local.date()


def _resolve_target_date(date_str: str | None) -> _date:
    import datetime as _dt

    if not date_str:
        return _us_slate_date()
    return _dt.datetime.strptime(date_str, "%Y-%m-%d").date()


def _load_schedule_day(date_str: str) -> pd.DataFrame:
    target = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(target):
        return pd.DataFrame()
    target_date = target.date()

    def _filter_day(frame: pd.DataFrame | None) -> pd.DataFrame:
        if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
            return pd.DataFrame()
        day_col = next((col for col in ("date_est", "date_utc", "date") if col in frame.columns), None)
        if not day_col:
            return pd.DataFrame()
        day_frame = frame.copy()
        day_frame[day_col] = pd.to_datetime(day_frame[day_col], errors="coerce").dt.date
        day_frame = day_frame[day_frame[day_col] == target_date].copy()
        return day_frame.reset_index(drop=True) if not day_frame.empty else pd.DataFrame()

    try:
        live_day = _filter_day(fetch_schedule_2025_26())
        if not live_day.empty:
            return live_day
    except Exception:
        pass

    season_start = target_date.year if target_date.month >= 7 else target_date.year - 1
    season_token = f"{season_start}_{str(season_start + 1)[-2:]}"
    schedule_candidates = [
        paths.data_processed / f"schedule_{season_token}.csv",
        paths.data_processed / f"schedule_{season_token}.json",
        paths.data_processed / "schedule_2025_26.csv",
        paths.data_processed / "schedule_2025_26.json",
    ]
    for sched_path in schedule_candidates:
        if not sched_path.exists():
            continue
        try:
            if sched_path.suffix.lower() == ".json":
                day = _filter_day(pd.read_json(sched_path))
            else:
                day = _filter_day(pd.read_csv(sched_path))
            if not day.empty:
                return day
        except Exception:
            continue

    return pd.DataFrame()


@cli.command("simulate-games")
@click.option("--date", "date_str", required=True, help="YYYY-MM-DD date to simulate")
@click.option("--sd-margin", type=float, default=12.0, help="Std dev for final margin (points)")
@click.option("--sd-total", type=float, default=22.0, help="Std dev for final total (points)")
def simulate_games_cmd(date_str: str, sd_margin: float, sd_total: float):
    """Analytical simulation for ML/ATS/TOTAL probabilities using rich factor adjustments.

    Reads odds (consensus-first), injuries impact, and opponent splits for the given date,
    computes adjusted spread/total means, and outputs probabilities plus EV to
    data/processed/games_sim_<date>.csv.
    """
    try:
        from .sim_games import SimConfig, simulate_games_for_date
    except Exception as e:
        console.print(f"[red]Import error: {e}")
        raise SystemExit(1)
    cfg = SimConfig(sd_margin=sd_margin, sd_total=sd_total)
    df = simulate_games_for_date(date_str, cfg)
    if df is None or df.empty:
        console.print(f"[yellow]No odds/factors available for {date_str}; wrote empty output if any.")
    else:
        console.print(f"[green]Wrote simulations for {date_str}: {len(df)} games")


@cli.command("smart-sim")
@click.option("--date", "date_str", required=True, help="YYYY-MM-DD date")
@click.option("--home", "home_tri", required=True, help="Home team tricode (e.g., LAL)")
@click.option("--away", "away_tri", required=True, help="Away team tricode (e.g., BOS)")
@click.option("--n-sims", type=int, default=2000, show_default=True, help="Number of event-level sims")
@click.option("--seed", type=int, default=None, help="RNG seed")
@click.option("--pbp/--no-pbp", default=True, show_default=True, help="Use unified possession-level sim (no forced quarter totals)")
@click.option("--market-total", type=float, default=None, help="Optional game total line")
@click.option("--home-spread", type=float, default=None, help="Optional home spread line (e.g., -3.5)")
def smart_sim_cmd(
    date_str: str,
    home_tri: str,
    away_tri: str,
    n_sims: int,
    seed: Optional[int],
    pbp: bool,
    market_total: Optional[float],
    home_spread: Optional[float],
):
    """Run SmartSim (event-level full-game simulation) for one matchup.

    Loads props predictions for the date and produces:
      - score distribution + ML/ATS/total probabilities
      - player stat distributions (pts/reb/ast/threes/stl/blk/tov + PRA)

    Output is written to data/processed/smart_sim_<date>_<HOME>_<AWAY>.json
    """
    try:
        import json

        from .sim.smart_sim import SmartSimConfig, simulate_smart_game
        from .sim.quarters import GameInputs, TeamContext, simulate_quarters
    except Exception as e:
        console.print(f"[red]Import error: {e}")
        raise SystemExit(1)

    home_tri = str(home_tri or "").strip().upper()
    away_tri = str(away_tri or "").strip().upper()
    if len(home_tri) != 3 or len(away_tri) != 3:
        console.print("[red]--home/--away must be NBA tricodes (3 letters)")
        raise SystemExit(2)

    props_path = paths.data_processed / f"props_predictions_{date_str}.csv"
    if not props_path.exists():
        console.print(f"[red]Missing props predictions: {props_path}")
        raise SystemExit(2)
    props_df = pd.read_csv(props_path)

    # When full-game odds are missing, attempt to approximate a market total from period lines
    # so quarter priors are not completely unanchored.
    market_total_for_quarters = market_total
    if market_total_for_quarters is None:
        try:
            from .sim.smart_sim import _period_lines_from_processed

            pl = _period_lines_from_processed(date_str=date_str, home_tri=home_tri, away_tri=away_tri) or {}
            h1 = pl.get("h1_total") if isinstance(pl, dict) else None
            h1f = float(h1) if h1 is not None else float("nan")
            if np.isfinite(h1f) and h1f > 0:
                market_total_for_quarters = float(2.0 * h1f)
        except Exception:
            pass

    excluded_map = _smart_sim_injuries_excluded_map_for_date(date_str, props_df=props_df)
    home_outs = int(max(0, min(5, len(excluded_map.get(home_tri, set())))))
    away_outs = int(max(0, min(5, len(excluded_map.get(away_tri, set())))))
    excluded_game = {
        str(home_tri): set(excluded_map.get(home_tri, set()) or set()),
        str(away_tri): set(excluded_map.get(away_tri, set()) or set()),
    }

    # Minimal quarter model: pace defaults to 98, ratings inferred from predicted total/margin
    # if predictions for this date exist; otherwise uses league-average.
    pred_path = paths.data_processed / f"predictions_{date_str}.csv"
    pred_total = None
    pred_margin = None
    pred_market_total = None
    pred_home_spread = None
    if pred_path.exists():
        try:
            pdf = pd.read_csv(pred_path)
            pdf["home_tri"] = pdf.get("home_team", "").astype(str).map(to_tricode)
            pdf["away_tri"] = pdf.get("visitor_team", "").astype(str).map(to_tricode)
            row = pdf[(pdf["home_tri"] == home_tri) & (pdf["away_tri"] == away_tri)].head(1)
            if not row.empty:
                r = row.iloc[0]
                pred_total = float(r.get("totals") or np.nan)
                pred_margin = float(r.get("spread_margin") or np.nan)
                try:
                    pred_market_total = float(r.get("total") or np.nan)
                except Exception:
                    pred_market_total = None
                try:
                    pred_home_spread = float(r.get("home_spread") or np.nan)
                except Exception:
                    pred_home_spread = None
        except Exception:
            pred_total = None
            pred_margin = None

    if market_total is None and (pred_market_total is not None) and np.isfinite(pred_market_total):
        market_total = float(pred_market_total)
    if home_spread is None and (pred_home_spread is not None) and np.isfinite(pred_home_spread):
        home_spread = float(pred_home_spread)

    if market_total is None or home_spread is None:
        odds_path = paths.data_processed / f"game_odds_{date_str}.csv"
        if odds_path.exists():
            try:
                odf = pd.read_csv(odds_path)
                if odf is not None and not odf.empty:
                    odf = odf.copy()
                    odf["home_tri"] = odf.get("home_team", "").astype(str).map(to_tricode)
                    odf["away_tri"] = odf.get("visitor_team", "").astype(str).map(to_tricode)
                    row2 = odf[(odf["home_tri"] == home_tri) & (odf["away_tri"] == away_tri)].head(1)
                    if not row2.empty:
                        rr = row2.iloc[0]
                        if market_total is None:
                            try:
                                v = float(pd.to_numeric(rr.get("total"), errors="coerce"))
                                if np.isfinite(v):
                                    market_total = v
                            except Exception:
                                pass
                        if home_spread is None:
                            try:
                                v = float(pd.to_numeric(rr.get("home_spread"), errors="coerce"))
                                if np.isfinite(v):
                                    home_spread = v
                            except Exception:
                                pass
            except Exception:
                pass

    # Best-effort: team-varying pace/defense priors from Basketball Reference season stats.
    adv_map: dict[str, dict[str, float]] = {}
    try:
        dts = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(dts):
            season_year = None
        else:
            season_year = int(dts.year + 1) if int(dts.month) >= 7 else int(dts.year)
        if season_year is not None:
            from .scrapers import BasketballReferenceScraper

            def _pick_asof_file(season_y: int, game_dt: pd.Timestamp) -> Optional[Path]:
                try:
                    pat = f"team_advanced_stats_{int(season_y)}_asof_*.csv"
                    cands = list(paths.data_processed.glob(pat))
                    if not cands:
                        return None
                    best = None
                    best_dt = None
                    for p in cands:
                        try:
                            s = p.stem
                            # team_advanced_stats_<season>_asof_YYYY-MM-DD
                            tag = s.split("_asof_", 1)[-1]
                            dt = pd.to_datetime(tag, errors="coerce")
                            if pd.isna(dt):
                                continue
                            dt = dt.normalize()
                            if dt <= game_dt.normalize() and (best_dt is None or dt > best_dt):
                                best_dt = dt
                                best = p
                        except Exception:
                            continue
                    return best
                except Exception:
                    return None

            fp = None
            try:
                if isinstance(dts, pd.Timestamp) and (not pd.isna(dts)):
                    fp = _pick_asof_file(int(season_year), dts)
            except Exception:
                fp = None
            if fp is None:
                fp = paths.data_processed / f"team_advanced_stats_{int(season_year)}.csv"

            sdf = None
            if fp.exists():
                try:
                    sdf = pd.read_csv(fp)
                except Exception:
                    sdf = None
            if sdf is None or sdf.empty:
                try:
                    scraper = BasketballReferenceScraper()
                    sdf = scraper.get_team_stats(int(season_year))
                    if sdf is not None and not sdf.empty:
                        try:
                            sdf.to_csv(fp, index=False)
                        except Exception:
                            pass
                except Exception:
                    sdf = None
            if sdf is not None and not sdf.empty:
                sdf = sdf.copy()
                sdf["team"] = sdf.get("team", "").astype(str).str.strip().str.upper()
                for _, rr in sdf.iterrows():
                    tri = str(rr.get("team") or "").strip().upper()
                    if not tri:
                        continue
                    adv_map[tri] = {
                        "pace": float(pd.to_numeric(rr.get("pace"), errors="coerce")),
                        "def_rtg": float(pd.to_numeric(rr.get("def_rtg"), errors="coerce")),
                        "off_rtg": float(pd.to_numeric(rr.get("off_rtg"), errors="coerce")),
                    }
    except Exception:
        adv_map = {}

    try:
        home_pace = float(adv_map.get(home_tri, {}).get("pace"))
    except Exception:
        home_pace = float("nan")
    try:
        away_pace = float(adv_map.get(away_tri, {}).get("pace"))
    except Exception:
        away_pace = float("nan")
    if not np.isfinite(home_pace):
        home_pace = 98.0
    if not np.isfinite(away_pace):
        away_pace = 98.0
    matchup_pace = float(np.mean([home_pace, away_pace])) if (np.isfinite(home_pace) and np.isfinite(away_pace)) else 98.0

    try:
        home_def_rtg = float(adv_map.get(home_tri, {}).get("def_rtg"))
    except Exception:
        home_def_rtg = float("nan")
    try:
        away_def_rtg = float(adv_map.get(away_tri, {}).get("def_rtg"))
    except Exception:
        away_def_rtg = float("nan")
    if not np.isfinite(home_def_rtg):
        home_def_rtg = 112.0
    if not np.isfinite(away_def_rtg):
        away_def_rtg = 112.0
    def _rating_from_mu(mu: Optional[float], pace_val: float) -> float:
        try:
            if mu is None or (not np.isfinite(mu)):
                return 112.0
            return float((float(mu) / max(1e-6, float(pace_val))) * 100.0)
        except Exception:
            return 112.0

    home_mu = None
    away_mu = None
    if pred_total is not None and pred_margin is not None and np.isfinite(pred_total) and np.isfinite(pred_margin):
        home_mu = 0.5 * (pred_total + pred_margin)
        away_mu = 0.5 * (pred_total - pred_margin)

    # Schedule-aware rest (best-effort) using features history.
    home_rest_days = None
    away_rest_days = None
    home_b2b = False
    away_b2b = False
    try:
        feats_csv = paths.data_processed / "features.csv"
        feats_parquet = paths.data_processed / "features.parquet"
        if feats_csv.exists():
            hist = pd.read_csv(feats_csv)
        elif feats_parquet.exists():
            hist = pd.read_parquet(feats_parquet)
        else:
            hist = None
        if hist is not None and not hist.empty:
            m = pd.DataFrame([
                {
                    "date": date_str,
                    "home_team": normalize_team(str(home_tri)),
                    "visitor_team": normalize_team(str(away_tri)),
                }
            ])
            h = hist[[c for c in ["date", "home_team", "visitor_team"] if c in hist.columns]].copy()
            if {"date", "home_team", "visitor_team"}.issubset(set(h.columns)):
                h["home_team"] = h["home_team"].astype(str).map(normalize_team)
                h["visitor_team"] = h["visitor_team"].astype(str).map(normalize_team)
                rest_df = compute_rest_for_matchups(m, h)
                if rest_df is not None and not rest_df.empty:
                    rr = rest_df.iloc[0]
                    try:
                        home_rest_days = int(pd.to_numeric(rr.get("home_rest_days"), errors="coerce")) if pd.notna(rr.get("home_rest_days")) else None
                    except Exception:
                        home_rest_days = None
                    try:
                        away_rest_days = int(pd.to_numeric(rr.get("visitor_rest_days"), errors="coerce")) if pd.notna(rr.get("visitor_rest_days")) else None
                    except Exception:
                        away_rest_days = None
                    home_b2b = bool(pd.to_numeric(rr.get("home_b2b"), errors="coerce") == 1) if pd.notna(rr.get("home_b2b")) else False
                    away_b2b = bool(pd.to_numeric(rr.get("visitor_b2b"), errors="coerce") == 1) if pd.notna(rr.get("visitor_b2b")) else False

    except Exception:
        pass

    home_ctx = TeamContext(
        team=home_tri,
        pace=float(home_pace),
        off_rating=_rating_from_mu(home_mu, matchup_pace),
        def_rating=float(home_def_rtg),
        injuries_out=int(home_outs),
        back_to_back=bool(home_b2b),
        rest_days=home_rest_days,
    )
    away_ctx = TeamContext(
        team=away_tri,
        pace=float(away_pace),
        off_rating=_rating_from_mu(away_mu, matchup_pace),
        def_rating=float(away_def_rtg),
        injuries_out=int(away_outs),
        back_to_back=bool(away_b2b),
        rest_days=away_rest_days,
    )
    qsum = simulate_quarters(
        GameInputs(date=date_str, home=home_ctx, away=away_ctx, market_total=market_total_for_quarters, market_home_spread=home_spread),
        n_samples=3000,
    )

    sim_cfg = SmartSimConfig(n_sims=int(n_sims), seed=seed, use_pbp=bool(pbp))
    pre_ctx = {
        "home_injuries_out": int(home_outs),
        "away_injuries_out": int(away_outs),
        "home_pace": float(home_pace) if np.isfinite(float(home_pace)) else None,
        "away_pace": float(away_pace) if np.isfinite(float(away_pace)) else None,
        "home_b2b": bool(home_b2b),
        "away_b2b": bool(away_b2b),
    }

    out = simulate_smart_game(
        date_str=date_str,
        home_tri=home_tri,
        away_tri=away_tri,
        props_df=props_df,
        quarters=qsum.quarters,
        market_total=market_total,
        market_home_spread=home_spread,
        cfg=sim_cfg,
        excluded_player_keys_by_team=excluded_game,
        pregame_context=pre_ctx,
    )

    out_path = paths.data_processed / f"smart_sim_{date_str}_{home_tri}_{away_tri}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    console.print({"output": str(out_path), "n_sims": int(n_sims), "home": home_tri, "away": away_tri})


_SMARTSIM_WORKER_STATE: dict[str, Any] = {}


def _smart_sim_worker_init(
    date_str: str,
    n_sims: int,
    seed: Optional[int],
    pbp: bool,
    props_path: str,
    roster_mode: str,
    excluded_map: dict[str, set[str]],
    adv_map: dict[str, dict[str, float]],
    game_id_map: dict[tuple[str, str], int],
    name_to_id: dict[str, int],
    team_name_to_id: dict[tuple[str, str], int],
) -> None:
    """Initializer for SmartSim worker processes (Windows-safe)."""
    import pandas as pd
    from pathlib import Path

    global _SMARTSIM_WORKER_STATE
    try:
        props_df = pd.read_csv(props_path) if props_path and Path(props_path).exists() else pd.DataFrame()
    except Exception:
        props_df = pd.DataFrame()

    _SMARTSIM_WORKER_STATE = {
        "date_str": str(date_str),
        "n_sims": int(n_sims),
        "seed": seed,
        "pbp": bool(pbp),
        "roster_mode": str(roster_mode or "historical"),
        "props_df": props_df,
        "excluded_map": excluded_map or {},
        "adv_map": adv_map or {},
        "game_id_map": game_id_map or {},
        "name_to_id": name_to_id or {},
        "team_name_to_id": team_name_to_id or {},
    }


def _smart_sim_worker_run(job: dict) -> dict:
    """Run SmartSim for a single game job.

    Returns: {status: wrote|failed, home, away, out_path, error?}
    """
    import json
    import re
    import unicodedata
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from .sim.quarters import GameInputs, TeamContext, simulate_quarters
    from .sim.smart_sim import SmartSimConfig, simulate_smart_game

    global _SMARTSIM_WORKER_STATE
    st = _SMARTSIM_WORKER_STATE or {}

    date_s = str(st.get("date_str") or job.get("date_str") or "")
    home_tri = str(job.get("home_tri") or "").strip().upper()
    away_tri = str(job.get("away_tri") or "").strip().upper()
    out_path_s = str(job.get("out_path") or "")
    out_path = Path(out_path_s)

    def _norm_name_key(s: str) -> str:
        s = (s or "").strip().upper()
        try:
            s = unicodedata.normalize("NFKD", s)
            s = s.encode("ascii", "ignore").decode("ascii")
        except Exception:
            pass
        s = s.replace("-", " ")
        if "(" in s:
            s = s.split("(", 1)[0]
        s = re.sub(r"[^A-Z0-9\s]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        for suf in (" JR", " SR", " II", " III", " IV", " V"):
            if s.endswith(suf):
                s = s[: -len(suf)].strip()
        return s

    try:
        market_total = job.get("market_total")
        home_spread = job.get("home_spread")

        market_total_for_quarters = market_total
        if market_total_for_quarters is None:
            try:
                from .sim.smart_sim import _period_lines_from_processed

                pl = _period_lines_from_processed(date_str=date_s, home_tri=home_tri, away_tri=away_tri) or {}
                h1 = pl.get("h1_total") if isinstance(pl, dict) else None
                h1f = float(h1) if h1 is not None else float("nan")
                if np.isfinite(h1f) and h1f > 0:
                    market_total_for_quarters = float(2.0 * h1f)
            except Exception:
                pass
        home_pace = float(job.get("home_pace") or 98.0)
        away_pace = float(job.get("away_pace") or 98.0)
        matchup_pace = float(job.get("matchup_pace") or np.mean([home_pace, away_pace]))
        home_def_rtg = float(job.get("home_def_rtg") or 112.0)
        away_def_rtg = float(job.get("away_def_rtg") or 112.0)
        home_off_rtg = float(job.get("home_off_rtg") or 112.0)
        away_off_rtg = float(job.get("away_off_rtg") or 112.0)
        home_outs = int(job.get("home_outs") or 0)
        away_outs = int(job.get("away_outs") or 0)
        home_b2b = bool(job.get("home_b2b") or False)
        away_b2b = bool(job.get("away_b2b") or False)
        home_rest_days = job.get("home_rest_days")
        away_rest_days = job.get("away_rest_days")

        home_ctx = TeamContext(
            team=home_tri,
            pace=float(home_pace),
            off_rating=float(home_off_rtg),
            def_rating=float(home_def_rtg),
            injuries_out=int(home_outs),
            back_to_back=bool(home_b2b),
            rest_days=(int(home_rest_days) if home_rest_days is not None else None),
        )
        away_ctx = TeamContext(
            team=away_tri,
            pace=float(away_pace),
            off_rating=float(away_off_rtg),
            def_rating=float(away_def_rtg),
            injuries_out=int(away_outs),
            back_to_back=bool(away_b2b),
            rest_days=(int(away_rest_days) if away_rest_days is not None else None),
        )

        qsum = simulate_quarters(
            GameInputs(date=date_s, home=home_ctx, away=away_ctx, market_total=market_total_for_quarters, market_home_spread=home_spread),
            n_samples=3000,
        )

        roster_mode = str(st.get("roster_mode") or job.get("roster_mode") or "historical")
        cfg = SmartSimConfig(
            n_sims=int(st.get("n_sims") or job.get("n_sims") or 0),
            seed=st.get("seed"),
            use_pbp=bool(st.get("pbp")),
            roster_mode=roster_mode,
        )
        pre_ctx = {
            "home_injuries_out": int(home_outs),
            "away_injuries_out": int(away_outs),
            "home_pace": float(home_pace) if np.isfinite(float(home_pace)) else None,
            "away_pace": float(away_pace) if np.isfinite(float(away_pace)) else None,
            "home_b2b": bool(home_b2b),
            "away_b2b": bool(away_b2b),
        }

        excluded_map_local = st.get("excluded_map") or {}
        excluded_game = {
            str(home_tri): set((excluded_map_local.get(home_tri) or set())),
            str(away_tri): set((excluded_map_local.get(away_tri) or set())),
        }

        out = simulate_smart_game(
            date_str=date_s,
            home_tri=home_tri,
            away_tri=away_tri,
            props_df=st.get("props_df"),
            quarters=qsum.quarters,
            market_total=market_total,
            market_home_spread=home_spread,
            cfg=cfg,
            excluded_player_keys_by_team=excluded_game,
            pregame_context=pre_ctx,
        )

        # Repair missing player_id fields via roster mapping.
        try:
            name_to_id_local = st.get("name_to_id") or {}
            team_name_to_id_local = st.get("team_name_to_id") or {}
            players = out.get("players") if isinstance(out, dict) else None
            if isinstance(players, dict):
                for side, team_tri in (("home", home_tri), ("away", away_tri)):
                    arr = players.get(side)
                    if not isinstance(arr, list):
                        continue
                    for pr in arr:
                        if not isinstance(pr, dict):
                            continue
                        pid = pr.get("player_id")
                        if pid is None or (isinstance(pid, float) and (not np.isfinite(pid))):
                            nm = str(pr.get("player_name") or "")
                            pk = _norm_name_key(nm) if nm else ""
                            if pk:
                                fixed = team_name_to_id_local.get((team_tri, pk)) or name_to_id_local.get(pk)
                                if fixed is not None:
                                    pr["player_id"] = int(fixed)
                        else:
                            try:
                                pr["player_id"] = int(pd.to_numeric(pid, errors="coerce"))
                            except Exception:
                                pass
        except Exception:
            pass

        # Attach game_id if we can map it.
        try:
            gid = (st.get("game_id_map") or {}).get((home_tri, away_tri))
            if gid is not None and isinstance(out, dict):
                out["game_id"] = int(gid)
        except Exception:
            pass

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
        try:
            tmp.replace(out_path)
        except Exception:
            out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
            try:
                tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass

        return {"status": "wrote", "home": home_tri, "away": away_tri, "out_path": str(out_path)}
    except Exception as e:
        return {"status": "failed", "home": home_tri, "away": away_tri, "out_path": out_path_s, "error": str(e)}


def _prune_stale_smart_sim_outputs(
    date_str: str,
    expected_matchups: set[tuple[str, str]],
    out_prefix: str = "smart_sim",
    remove_all: bool = False,
) -> int:
    removed = 0
    try:
        out_prefix_s = str(out_prefix or "smart_sim").strip() or "smart_sim"
        expected = {
            f"{out_prefix_s}_{date_str}_{str(home_tri or '').strip().upper()}_{str(away_tri or '').strip().upper()}.json"
            for home_tri, away_tri in (expected_matchups or set())
            if str(home_tri or "").strip() and str(away_tri or "").strip()
        }
        for fp in paths.data_processed.glob(f"{out_prefix_s}_{date_str}_*.json"):
            if (not remove_all) and fp.name in expected:
                continue
            try:
                fp.unlink()
                removed += 1
            except Exception:
                continue
    except Exception:
        return removed
    return removed


def _smart_sim_run_date(
    date_str: str,
    n_sims: int,
    seed: Optional[int],
    max_games: Optional[int],
    overwrite: bool,
    pbp: bool = True,
    workers: Optional[int] = None,
    roster_mode: str = "historical",
    out_prefix: str = "smart_sim",
) -> dict:
    """Internal: run SmartSim for every game on a date.

    Returns a summary dict with wrote/skipped/failures.
    """
    try:
        import json

        import os
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from pathlib import Path

        from .sim.smart_sim import SmartSimConfig, simulate_smart_game
        from .sim.quarters import GameInputs, TeamContext, simulate_quarters
    except Exception as e:
        console.print(f"[red]Import error: {e}")
        raise SystemExit(1)

    pred_path = paths.data_processed / f"predictions_{date_str}.csv"
    if not pred_path.exists():
        return {"date": date_str, "wrote": 0, "skipped": 0, "failures": 0, "reason": f"missing_predictions:{pred_path}"}
    pdf = pd.read_csv(pred_path)
    if pdf is None or pdf.empty:
        return {"date": date_str, "wrote": 0, "skipped": 0, "failures": 0, "reason": f"empty_predictions:{pred_path}"}

    props_path = paths.data_processed / f"props_predictions_{date_str}.csv"
    if not props_path.exists():
        return {"date": date_str, "wrote": 0, "skipped": 0, "failures": 0, "reason": f"missing_props:{props_path}"}
    props_df = pd.read_csv(props_path)

    excluded_map = _smart_sim_injuries_excluded_map_for_date(date_str, props_df=props_df)

    odds_path = paths.data_processed / f"game_odds_{date_str}.csv"
    odds_df = None
    if odds_path.exists():
        try:
            odds_df = pd.read_csv(odds_path)
            if odds_df is not None and not odds_df.empty:
                odds_df = odds_df.copy()
                odds_df["home_tri"] = odds_df.get("home_team", "").astype(str).map(to_tricode)
                odds_df["away_tri"] = odds_df.get("visitor_team", "").astype(str).map(to_tricode)
        except Exception:
            odds_df = None

    # Map (home_tri, away_tri) -> game_id when available (needed for reconciliation).
    # Prefer pbp_reconcile_<date>.csv for completed games (it is more complete/reliable than game_cards).
    game_id_map: dict[tuple[str, str], int] = {}
    try:
        pbp_map_p = paths.data_processed / f"pbp_reconcile_{date_str}.csv"
        if pbp_map_p.exists():
            mdf = pd.read_csv(pbp_map_p)
            if mdf is not None and not mdf.empty:
                mdf = mdf.copy()
                mdf["home_tri"] = mdf.get("home_team", "").astype(str).map(to_tricode)
                mdf["away_tri"] = mdf.get("visitor_team", "").astype(str).map(to_tricode)
                mdf["game_id"] = pd.to_numeric(mdf.get("game_id"), errors="coerce")
                mdf = mdf.dropna(subset=["home_tri", "away_tri", "game_id"])
                mdf = mdf.drop_duplicates(subset=["home_tri", "away_tri", "game_id"]).copy()
                for _, rr in mdf.iterrows():
                    ht = str(rr.get("home_tri") or "").strip().upper()
                    at = str(rr.get("away_tri") or "").strip().upper()
                    try:
                        gid = int(rr.get("game_id"))
                    except Exception:
                        continue
                    if ht and at:
                        game_id_map[(ht, at)] = gid
                        # Some upstream sources occasionally swap home/away; map both directions for lookup.
                        if (at, ht) not in game_id_map:
                            game_id_map[(at, ht)] = gid
    except Exception:
        game_id_map = {}

    # Supplement with game_cards_<date>.csv when present (useful pre-game / same-day).
    try:
        cards_p = paths.data_processed / f"game_cards_{date_str}.csv"
        if cards_p.exists():
            cdf = pd.read_csv(cards_p)
            if cdf is not None and not cdf.empty:
                cdf = cdf.copy()
                cdf["home_tri"] = cdf.get("home_team", "").astype(str).map(to_tricode)
                cdf["away_tri"] = cdf.get("visitor_team", "").astype(str).map(to_tricode)
                cdf["game_id"] = pd.to_numeric(cdf.get("game_id"), errors="coerce")
                cdf = cdf.dropna(subset=["home_tri", "away_tri", "game_id"])
                for _, rr in cdf.iterrows():
                    ht = str(rr.get("home_tri") or "").strip().upper()
                    at = str(rr.get("away_tri") or "").strip().upper()
                    try:
                        gid = int(rr.get("game_id"))
                    except Exception:
                        continue
                    if ht and at and (ht, at) not in game_id_map:
                        game_id_map[(ht, at)] = gid
    except Exception:
        pass

    # Load team advanced stats (pace/def_rtg) for this season, best-effort.
    # Basketball Reference season uses the end-year, e.g. 2026 for 2025-26.
    adv_map: dict[str, dict[str, float]] = {}
    try:
        dts = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(dts):
            season_year = None
        else:
            season_year = int(dts.year + 1) if int(dts.month) >= 7 else int(dts.year)
    except Exception:
        season_year = None

    def _load_adv_map(season_y: int) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        try:
            from .scrapers import BasketballReferenceScraper

            fp = paths.data_processed / f"team_advanced_stats_{int(season_y)}.csv"
            sdf = None
            if fp.exists():
                try:
                    sdf = pd.read_csv(fp)
                except Exception:
                    sdf = None
            if sdf is None or sdf.empty:
                try:
                    scraper = BasketballReferenceScraper()
                    sdf = scraper.get_team_stats(int(season_y))
                    if sdf is not None and not sdf.empty:
                        try:
                            sdf.to_csv(fp, index=False)
                        except Exception:
                            pass
                except Exception:
                    sdf = None
            if sdf is None or sdf.empty:
                return {}

            sdf = sdf.copy()
            sdf["team"] = sdf.get("team", "").astype(str).str.strip().str.upper()
            for _, rr in sdf.iterrows():
                tri = str(rr.get("team") or "").strip().upper()
                if not tri:
                    continue
                try:
                    pace_v = float(pd.to_numeric(rr.get("pace"), errors="coerce"))
                except Exception:
                    pace_v = float("nan")
                try:
                    def_v = float(pd.to_numeric(rr.get("def_rtg"), errors="coerce"))
                except Exception:
                    def_v = float("nan")
                try:
                    off_v = float(pd.to_numeric(rr.get("off_rtg"), errors="coerce"))
                except Exception:
                    off_v = float("nan")
                out[tri] = {
                    "pace": pace_v,
                    "def_rtg": def_v,
                    "off_rtg": off_v,
                }
        except Exception:
            return {}
        return out

    if season_year is not None:
        adv_map = _load_adv_map(int(season_year))

    def _num(x):
        try:
            v = float(pd.to_numeric(x, errors="coerce"))
            return v if np.isfinite(v) else None
        except Exception:
            return None

    # Compute schedule rest_days/b2b for today's matchups (best effort).
    # This ensures SmartSim quarter priors incorporate pregame schedule context.
    try:
        feats_csv = paths.data_processed / "features.csv"
        feats_parquet = paths.data_processed / "features.parquet"
        if feats_csv.exists():
            hist = pd.read_csv(feats_csv)
        elif feats_parquet.exists():
            hist = pd.read_parquet(feats_parquet)
        else:
            hist = None
        if hist is not None and not hist.empty:
            h = hist[[c for c in ["date", "home_team", "visitor_team"] if c in hist.columns]].copy()
            if {"date", "home_team", "visitor_team"}.issubset(set(h.columns)):
                h["home_team"] = h["home_team"].astype(str).map(normalize_team)
                h["visitor_team"] = h["visitor_team"].astype(str).map(normalize_team)
                m = pdf[[c for c in ["date", "home_team", "visitor_team"] if c in pdf.columns]].copy()
                if "date" not in m.columns:
                    m["date"] = date_str
                m["home_team"] = m.get("home_team", "").astype(str).map(normalize_team)
                m["visitor_team"] = m.get("visitor_team", "").astype(str).map(normalize_team)
                rest_df = compute_rest_for_matchups(m, h)
                if rest_df is not None and not rest_df.empty:
                    # Merge onto pdf by date+teams.
                    rest_cols = [c for c in ["home_rest_days", "visitor_rest_days", "home_b2b", "visitor_b2b"] if c in rest_df.columns]
                    key_cols = [c for c in ["date", "home_team", "visitor_team"] if c in rest_df.columns]
                    if rest_cols and key_cols:
                        pdf = pdf.copy()
                        pdf["date"] = pdf.get("date", date_str)
                        pdf["home_team"] = pdf.get("home_team", "").astype(str).map(normalize_team)
                        pdf["visitor_team"] = pdf.get("visitor_team", "").astype(str).map(normalize_team)
                        rest_df = rest_df.copy()
                        rest_df["date"] = rest_df.get("date", date_str)
                        rest_df["home_team"] = rest_df.get("home_team", "").astype(str).map(normalize_team)
                        rest_df["visitor_team"] = rest_df.get("visitor_team", "").astype(str).map(normalize_team)
                        pdf = pdf.merge(rest_df[key_cols + rest_cols], on=key_cols, how="left")

    except Exception:
        pass

    # Standardize tricodes
    pdf = pdf.copy()
    pdf["home_tri"] = pdf.get("home_team", "").astype(str).map(to_tricode)
    pdf["away_tri"] = pdf.get("visitor_team", "").astype(str).map(to_tricode)
    pdf = pdf[(pdf["home_tri"].astype(str).str.len() == 3) & (pdf["away_tri"].astype(str).str.len() == 3)].copy()
    if pdf.empty:
        return {"date": date_str, "wrote": 0, "skipped": 0, "failures": 0, "reason": "no_valid_games"}

    out_prefix_s = str(out_prefix or "smart_sim").strip()
    if not out_prefix_s:
        out_prefix_s = "smart_sim"

    expected_matchups = {
        (str(r.get("home_tri") or "").strip().upper(), str(r.get("away_tri") or "").strip().upper())
        for _, r in pdf.iterrows()
        if str(r.get("home_tri") or "").strip() and str(r.get("away_tri") or "").strip()
    }

    # Always remove stale extra matchup files so phantom games cannot survive a slate correction.
    # When using --overwrite with --max-games, clear all same-date outputs to avoid mixing partial reruns.
    try:
        _prune_stale_smart_sim_outputs(
            date_str,
            expected_matchups,
            out_prefix=out_prefix_s,
            remove_all=bool(overwrite and max_games is not None),
        )
    except Exception:
        pass

    if max_games is not None:
        try:
            pdf = pdf.head(int(max_games))
        except Exception:
            pass

    wrote = 0
    skipped = 0
    failures: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []

    # Roster-based player_id repair for SmartSim outputs.
    # Some older SmartSim paths can emit players without ids; fix via processed rosters.
    name_to_id: dict[str, int] = {}
    team_name_to_id: dict[tuple[str, str], int] = {}
    try:
        import re as _re
        import unicodedata as _ud

        def _norm_name_key(s: str) -> str:
            s = (s or "").strip().upper()
            try:
                s = _ud.normalize("NFKD", s)
                s = s.encode("ascii", "ignore").decode("ascii")
            except Exception:
                pass
            s = s.replace("-", " ")
            if "(" in s:
                s = s.split("(", 1)[0]
            s = _re.sub(r"[^A-Z0-9\s]", "", s)
            s = _re.sub(r"\s+", " ", s).strip()
            for suf in (" JR", " SR", " II", " III", " IV", " V"):
                if s.endswith(suf):
                    s = s[: -len(suf)].strip()
            return s

        # First preference for historical dates: boxscores mapping (complete for teams that played).
        try:
            bs_p = paths.data_processed / f"boxscores_{date_str}.csv"
            if bs_p.exists():
                bdf = pd.read_csv(bs_p)
                if bdf is not None and not bdf.empty and {"PLAYER_NAME", "PLAYER_ID"}.issubset(set(bdf.columns)):
                    bdf = bdf.copy()
                    bdf["PLAYER_ID"] = pd.to_numeric(bdf["PLAYER_ID"], errors="coerce")
                    bdf = bdf.dropna(subset=["PLAYER_ID"])
                    bdf["_pkey"] = bdf["PLAYER_NAME"].astype(str).map(_norm_name_key)
                    if "TEAM_ABBREVIATION" in bdf.columns:
                        bdf["_tri"] = bdf["TEAM_ABBREVIATION"].astype(str).map(lambda x: to_tricode(str(x)) or str(x))
                        bdf["_tri"] = bdf["_tri"].astype(str).str.upper().str.strip()
                    for _, rr in bdf.iterrows():
                        try:
                            pk = str(rr.get("_pkey") or "").strip().upper()
                            pid = int(rr.get("PLAYER_ID"))
                            if pk:
                                name_to_id.setdefault(pk, pid)
                            if "_tri" in rr and pk:
                                tri = str(rr.get("_tri") or "").strip().upper()
                                if tri:
                                    team_name_to_id.setdefault((tri, pk), pid)
                        except Exception:
                            continue
        except Exception:
            pass

        try:
            d = pd.to_datetime(date_str, errors="coerce")
            start_year = int(d.year) if (not pd.isna(d)) and int(d.month) >= 7 else (int(d.year) - 1 if not pd.isna(d) else None)
            season = f"{start_year}-{str(start_year+1)[-2:]}" if start_year is not None else None
        except Exception:
            season = None

        roster_file = None
        if season:
            cand = paths.data_processed / f"rosters_{season}.csv"
            if cand.exists():
                roster_file = cand
        if roster_file is None:
            files = list(paths.data_processed.glob("rosters_*.csv"))
            if files:
                files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
                roster_file = files[0]

        if roster_file is not None and roster_file.exists():
            rdf = pd.read_csv(roster_file)
            if rdf is not None and not rdf.empty:
                cols = {c.upper(): c for c in rdf.columns}
                name_col = cols.get("PLAYER")
                id_col = cols.get("PLAYER_ID")
                tri_col = cols.get("TEAM_ABBREVIATION")
                if name_col and id_col:
                    tmp = rdf[[name_col, id_col] + ([tri_col] if tri_col else [])].copy()
                    tmp[id_col] = pd.to_numeric(tmp[id_col], errors="coerce")
                    tmp = tmp.dropna(subset=[id_col])
                    tmp["_pkey"] = tmp[name_col].astype(str).map(_norm_name_key)
                    if tri_col:
                        tmp["_tri"] = tmp[tri_col].astype(str).map(lambda x: to_tricode(str(x)) or str(x))
                        tmp["_tri"] = tmp["_tri"].astype(str).str.upper().str.strip()
                    for _, rr in tmp.iterrows():
                        try:
                            pk = str(rr.get("_pkey") or "").strip().upper()
                            pid = int(rr.get(id_col))
                            if pk:
                                name_to_id.setdefault(pk, pid)
                            if tri_col:
                                tri = str(rr.get("_tri") or "").strip().upper()
                                if pk and tri:
                                    team_name_to_id.setdefault((tri, pk), pid)
                        except Exception:
                            continue
    except Exception:
        name_to_id = {}
        team_name_to_id = {}

    for _, r in pdf.iterrows():
        home_tri = str(r.get("home_tri") or "").strip().upper()
        away_tri = str(r.get("away_tri") or "").strip().upper()
        if not home_tri or not away_tri:
            continue

        out_path = paths.data_processed / f"{out_prefix_s}_{date_str}_{home_tri}_{away_tri}.json"
        if out_path.exists() and (not overwrite):
            skipped += 1
            continue

        market_total = _num(r.get("total"))
        home_spread = _num(r.get("home_spread"))
        if (market_total is None or home_spread is None) and (odds_df is not None and not odds_df.empty):
            m = odds_df[(odds_df["home_tri"] == home_tri) & (odds_df["away_tri"] == away_tri)]
            if not m.empty:
                rr = m.iloc[0]
                if market_total is None:
                    market_total = _num(rr.get("total"))
                if home_spread is None:
                    home_spread = _num(rr.get("home_spread"))

        pred_total = _num(r.get("totals"))
        pred_margin = _num(r.get("spread_margin"))
        home_mu = (0.5 * (pred_total + pred_margin)) if (pred_total is not None and pred_margin is not None) else None
        away_mu = (0.5 * (pred_total - pred_margin)) if (pred_total is not None and pred_margin is not None) else None

        # Use team-varying pace/defense priors when available.
        try:
            home_pace = float(adv_map.get(home_tri, {}).get("pace"))
        except Exception:
            home_pace = float("nan")
        try:
            away_pace = float(adv_map.get(away_tri, {}).get("pace"))
        except Exception:
            away_pace = float("nan")
        if not np.isfinite(home_pace):
            home_pace = 98.0
        if not np.isfinite(away_pace):
            away_pace = 98.0
        matchup_pace = float(np.mean([home_pace, away_pace])) if (np.isfinite(home_pace) and np.isfinite(away_pace)) else 98.0

        try:
            home_def_rtg = float(adv_map.get(home_tri, {}).get("def_rtg"))
        except Exception:
            home_def_rtg = float("nan")
        try:
            away_def_rtg = float(adv_map.get(away_tri, {}).get("def_rtg"))
        except Exception:
            away_def_rtg = float("nan")
        if not np.isfinite(home_def_rtg):
            home_def_rtg = 112.0
        if not np.isfinite(away_def_rtg):
            away_def_rtg = 112.0

        def _rating_from_mu(mu: Optional[float], pace_val: float) -> float:
            try:
                if mu is None or (not np.isfinite(mu)):
                    return 112.0
                return float((float(mu) / max(1e-6, float(pace_val))) * 100.0)
            except Exception:
                return 112.0

        home_off_rtg = _rating_from_mu(home_mu, matchup_pace)
        away_off_rtg = _rating_from_mu(away_mu, matchup_pace)

        try:
            home_rest_days = int(pd.to_numeric(r.get("home_rest_days"), errors="coerce")) if pd.notna(r.get("home_rest_days")) else None
        except Exception:
            home_rest_days = None
        try:
            away_rest_days = int(pd.to_numeric(r.get("visitor_rest_days"), errors="coerce")) if pd.notna(r.get("visitor_rest_days")) else None
        except Exception:
            away_rest_days = None
        try:
            home_b2b = bool(pd.to_numeric(r.get("home_b2b"), errors="coerce") == 1) if pd.notna(r.get("home_b2b")) else False
        except Exception:
            home_b2b = False
        try:
            away_b2b = bool(pd.to_numeric(r.get("visitor_b2b"), errors="coerce") == 1) if pd.notna(r.get("visitor_b2b")) else False
        except Exception:
            away_b2b = False

        home_outs = int(max(0, min(5, len(excluded_map.get(home_tri, set())))))
        away_outs = int(max(0, min(5, len(excluded_map.get(away_tri, set())))))

        jobs.append(
            {
                "date_str": date_str,
                "home_tri": home_tri,
                "away_tri": away_tri,
                "out_path": str(out_path),
                "roster_mode": str(roster_mode or "historical"),
                "market_total": market_total,
                "home_spread": home_spread,
                "home_pace": float(home_pace),
                "away_pace": float(away_pace),
                "matchup_pace": float(matchup_pace),
                "home_def_rtg": float(home_def_rtg),
                "away_def_rtg": float(away_def_rtg),
                "home_off_rtg": float(home_off_rtg),
                "away_off_rtg": float(away_off_rtg),
                "home_outs": int(home_outs),
                "away_outs": int(away_outs),
                "home_b2b": bool(home_b2b),
                "away_b2b": bool(away_b2b),
                "home_rest_days": home_rest_days,
                "away_rest_days": away_rest_days,
            }
        )

    # Resolve workers: explicit arg -> env -> default 1
    env_workers = None
    try:
        env_workers = int(os.environ.get("SMARTSIM_WORKERS") or os.environ.get("SMART_SIM_WORKERS") or 0)
    except Exception:
        env_workers = None
    if workers is None:
        workers = env_workers if (env_workers is not None and int(env_workers) > 0) else 1
    try:
        workers = int(workers)
    except Exception:
        workers = 1
    if workers < 1:
        workers = 1

    # Run jobs serially or in parallel
    if jobs:
        if workers == 1 or len(jobs) == 1:
            _smart_sim_worker_init(
                date_str=str(date_str),
                n_sims=int(n_sims),
                seed=seed,
                pbp=bool(pbp),
                props_path=str(props_path),
                roster_mode=str(roster_mode or "historical"),
                excluded_map=excluded_map,
                adv_map=adv_map,
                game_id_map=game_id_map,
                name_to_id=name_to_id,
                team_name_to_id=team_name_to_id,
            )
            for job in jobs:
                res = _smart_sim_worker_run(job)
                if res.get("status") == "wrote":
                    wrote += 1
                else:
                    failures.append({"home": res.get("home"), "away": res.get("away"), "error": res.get("error")})
        else:
            w = min(int(workers), int(len(jobs)))
            with ProcessPoolExecutor(
                max_workers=int(w),
                initializer=_smart_sim_worker_init,
                initargs=(
                    str(date_str),
                    int(n_sims),
                    seed,
                    bool(pbp),
                    str(props_path),
                    str(roster_mode or "historical"),
                    excluded_map,
                    adv_map,
                    game_id_map,
                    name_to_id,
                    team_name_to_id,
                ),
            ) as ex:
                futs = [ex.submit(_smart_sim_worker_run, job) for job in jobs]
                for fut in as_completed(futs):
                    try:
                        res = fut.result()
                    except Exception as e:
                        failures.append({"home": None, "away": None, "error": str(e)})
                        continue
                    if res.get("status") == "wrote":
                        wrote += 1
                    else:
                        failures.append({"home": res.get("home"), "away": res.get("away"), "error": res.get("error")})

    if failures:
        fp = paths.data_processed / f"{out_prefix_s}_failures_{date_str}.csv"
        try:
            pd.DataFrame(failures).to_csv(fp, index=False)
        except Exception:
            pass
        return {"date": date_str, "wrote": int(wrote), "skipped": int(skipped), "failures": int(len(failures)), "failures_file": str(fp)}

    return {"date": date_str, "wrote": int(wrote), "skipped": int(skipped), "failures": 0}


def _season_year_from_date_str(date_str: str) -> int:
    """Return NBA season year (e.g., 2026 for 2025-26) from a YYYY-MM-DD date."""
    d = pd.to_datetime(str(date_str).strip()).date()
    # Season label is the calendar year in which the season ends.
    # NBA season starts in fall (Oct) and ends in spring (Jun).
    return int(d.year + 1) if int(d.month) >= 7 else int(d.year)


def _ensure_team_advanced_stats_asof(season: int, as_of: str) -> Path | None:
    """Ensure an as-of team advanced stats file exists (built from cached boxscores).

    Returns the path if present/created, else None.
    """
    as_of_s = str(as_of).strip()
    safe = as_of_s.replace(":", "-")
    out_path = paths.data_processed / f"team_advanced_stats_{int(season)}_asof_{safe}.csv"
    if out_path.exists():
        return out_path
    try:
        from .advanced_stats_boxscores import compute_team_advanced_stats_from_boxscores

        try:
            console.print(
                f"Building team advanced stats as-of {as_of_s} (season={int(season)})",
                style="cyan",
            )
        except Exception:
            pass
        stats = compute_team_advanced_stats_from_boxscores(int(season), as_of=as_of_s)
        if stats is None or stats.empty:
            # Fallback: compute from cached player logs (no per-game boxscore cache needed).
            try:
                try:
                    console.print(
                        "Boxscore-cache advanced stats unavailable; falling back to player_logs",
                        style="yellow",
                    )
                except Exception:
                    pass
                from .advanced_stats_player_logs import compute_team_advanced_stats_from_player_logs

                stats = compute_team_advanced_stats_from_player_logs(int(season), as_of=as_of_s)
            except Exception:
                stats = None
        if stats is None or stats.empty:
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(out_path, index=False)
        return out_path
    except Exception:
        return None


@cli.command("smart-sim-date")
@click.option("--date", "date_str", required=True, help="YYYY-MM-DD date")
@click.option("--n-sims", type=int, default=2000, show_default=True, help="Number of event-level sims per game")
@click.option("--seed", type=int, default=None, help="Optional RNG seed")
@click.option("--pbp/--no-pbp", default=True, show_default=True, help="Use unified possession-level sim (no forced quarter totals)")
@click.option("--max-games", type=int, default=None, help="Optional cap for quick runs")
@click.option("--workers", type=int, default=1, show_default=True, help="Parallel workers (per-game). Use >1 to speed up full slates")
@click.option(
    "--roster-mode",
    type=str,
    default="historical",
    show_default=True,
    help="Roster sourcing mode: 'historical' (may use boxscore fallbacks) or 'pregame' (props+season rosters only)",
)
@click.option(
    "--out-prefix",
    type=str,
    default="smart_sim",
    show_default=True,
    help="Output filename prefix under data/processed (default: smart_sim). Example: smart_sim_pregame",
)
@click.option(
    "--refresh-asof-priors/--no-refresh-asof-priors",
    default=True,
    show_default=True,
    help="Build team_advanced_stats_<season>_asof_<date>.csv from cached boxscores before sim (no network); non-fatal if unavailable",
)
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing smart_sim_*.json outputs")
def smart_sim_date_cmd(
    date_str: str,
    n_sims: int,
    seed: Optional[int],
    pbp: bool,
    max_games: Optional[int],
    workers: int,
    roster_mode: str,
    out_prefix: str,
    refresh_asof_priors: bool,
    overwrite: bool,
):
    """Run SmartSim for every game on a date (from predictions_<date>.csv).

    Writes one JSON per game to data/processed/smart_sim_<date>_<HOME>_<AWAY>.json
    """
    if refresh_asof_priors:
        try:
            asof_for_priors = str(date_str)
            rm = str(roster_mode or "historical").strip().lower()
            if rm in {"pregame", "pregame_safe", "pregame-safe", "safe_pregame", "no_boxscore", "no-boxscore"}:
                ts = pd.to_datetime(asof_for_priors, errors="coerce")
                if ts is not None and (not pd.isna(ts)):
                    asof_for_priors = (ts.normalize() - pd.Timedelta(days=1)).date().isoformat()
            season = _season_year_from_date_str(asof_for_priors)
            _ensure_team_advanced_stats_asof(season=season, as_of=asof_for_priors)
        except Exception:
            pass

    summary = _smart_sim_run_date(
        date_str=date_str,
        n_sims=n_sims,
        seed=seed,
        max_games=max_games,
        overwrite=overwrite,
        pbp=bool(pbp),
        workers=int(workers),
        roster_mode=str(roster_mode or "historical"),
        out_prefix=str(out_prefix or "smart_sim"),
    )
    console.print(summary)


@cli.command("smart-sim-range")
@click.option("--start", "start_date", required=True, help="Start date YYYY-MM-DD")
@click.option("--end", "end_date", required=True, help="End date YYYY-MM-DD")
@click.option("--n-sims", type=int, default=2000, show_default=True, help="Number of event-level sims per game")
@click.option("--seed", type=int, default=None, help="Optional RNG seed")
@click.option("--pbp/--no-pbp", default=True, show_default=True, help="Use unified possession-level sim (no forced quarter totals)")
@click.option(
    "--roster-mode",
    type=str,
    default="historical",
    show_default=True,
    help="Roster sourcing mode: 'historical' (may use boxscore fallbacks) or 'pregame' (props+season rosters only)",
)
@click.option(
    "--out-prefix",
    type=str,
    default="smart_sim",
    show_default=True,
    help="Output filename prefix under data/processed (default: smart_sim). Example: smart_sim_pregame",
)
@click.option(
    "--refresh-asof-priors/--no-refresh-asof-priors",
    default=True,
    show_default=True,
    help="Build team_advanced_stats_<season>_asof_<date>.csv from cached boxscores before each date's sim (no network); non-fatal if unavailable",
)
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing smart_sim_*.json outputs")
@click.option("--max-games", type=int, default=None, help="Optional cap per date for quick runs")
@click.option("--workers", type=int, default=1, show_default=True, help="Parallel workers (per-game) for each date")
@click.option("--sleep", type=float, default=0.0, show_default=True, help="Sleep seconds between dates")
def smart_sim_range_cmd(
    start_date: str,
    end_date: str,
    n_sims: int,
    seed: Optional[int],
    pbp: bool,
    roster_mode: str,
    out_prefix: str,
    refresh_asof_priors: bool,
    overwrite: bool,
    max_games: Optional[int],
    workers: int,
    sleep: float,
):
    """Backfill SmartSim across a date range.

    Uses predictions_<date>.csv + props_predictions_<date>.csv per day.
    Skips existing smart_sim_*.json unless --overwrite is provided.
    """
    console.rule("SmartSim Range")
    try:
        s = pd.to_datetime(start_date).date()
        e = pd.to_datetime(end_date).date()
    except Exception:
        console.print("Invalid --start/--end (YYYY-MM-DD)", style="red")
        raise SystemExit(2)
    if e < s:
        console.print("--end must be >= --start", style="red")
        raise SystemExit(2)

    total_wrote = 0
    total_skipped = 0
    total_failures = 0
    days = 0
    for d in pd.date_range(s, e, freq="D"):
        ds = d.strftime("%Y-%m-%d")
        days += 1
        if refresh_asof_priors:
            try:
                asof_for_priors = str(ds)
                rm = str(roster_mode or "historical").strip().lower()
                if rm in {"pregame", "pregame_safe", "pregame-safe", "safe_pregame", "no_boxscore", "no-boxscore"}:
                    ts = pd.to_datetime(asof_for_priors, errors="coerce")
                    if ts is not None and (not pd.isna(ts)):
                        asof_for_priors = (ts.normalize() - pd.Timedelta(days=1)).date().isoformat()
                season = _season_year_from_date_str(asof_for_priors)
                _ensure_team_advanced_stats_asof(season=season, as_of=asof_for_priors)
            except Exception:
                pass
        summary = _smart_sim_run_date(
            date_str=ds,
            n_sims=n_sims,
            seed=seed,
            max_games=max_games,
            overwrite=overwrite,
            pbp=bool(pbp),
            workers=int(workers),
            roster_mode=str(roster_mode or "historical"),
            out_prefix=str(out_prefix or "smart_sim"),
        )
        console.print(summary)
        total_wrote += int(summary.get("wrote") or 0)
        total_skipped += int(summary.get("skipped") or 0)
        total_failures += int(summary.get("failures") or 0)
        if sleep and sleep > 0:
            try:
                time.sleep(float(sleep))
            except Exception:
                pass

    console.print({"range": f"{start_date}..{end_date}", "days": int(days), "wrote": int(total_wrote), "skipped": int(total_skipped), "failures": int(total_failures)})

@cli.command("backfill-injuries-season")
@click.option("--start", "start_date", type=str, required=True, help="Start date YYYY-MM-DD")
@click.option("--end", "end_date", type=str, required=True, help="End date YYYY-MM-DD")
def backfill_injuries_season_cmd(start_date: str, end_date: str):
    """Backfill injuries_overrides_<date>.csv for each date in a range using season-long/indefinite OUT flags from injuries.csv.

    Reads data/raw/injuries.csv and writes per-day overrides with columns [team, player, status] and status='OUT'.
    """
    console.rule("Backfill Injury Overrides (season-long/indefinite)")
    try:
        s = pd.to_datetime(start_date).date()
        e = pd.to_datetime(end_date).date()
    except Exception:
        console.print("Invalid --start/--end (YYYY-MM-DD)", style="red"); return
    if e < s:
        console.print("--end must be >= --start", style="red"); return
    inj_path = paths.data_raw / "injuries.csv"
    if not inj_path.exists():
        console.print(f"Injuries DB not found: {inj_path}", style="red"); return
    df = pd.read_csv(inj_path)
    if df is None or df.empty:
        console.print("Injuries DB is empty", style="red"); return
    # Normalize
    for c in ("team","player","status","injury"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    df["status_norm"] = df.get("status", "").astype(str).str.upper()
    df["injury_norm"] = df.get("injury", "").astype(str).str.upper()
    # Season-long or indefinite classifier
    def _season_long(u: str, j: str) -> bool:
        u = str(u or "").upper(); j = str(j or "").upper()
        if ("OUT" in u) and ("SEASON" in u or "INDEFINITE" in u):
            return True
        if ("SEASON-ENDING" in u):
            return True
        if ("OUT FOR SEASON" in j) or ("SEASON-ENDING" in j) or ("OUT INDEFINITELY" in j):
            return True
        return False
    df["season_block"] = df.apply(lambda r: _season_long(r.get("status_norm"), r.get("injury_norm")), axis=1)
    block = df[df["season_block"]].copy()
    if block.empty:
        console.print("No season-long/indefinite injuries found in DB; nothing to backfill", style="yellow"); return
    from .teams import to_tricode as _to_tri
    block["team_tri"] = block["team"].map(lambda x: _to_tri(str(x)))
    block = block[block["team_tri"].astype(str).str.len() == 3]
    uniq = block.drop_duplicates(subset=["player","team_tri"]).copy()
    dates = pd.date_range(s, e, freq="D").date
    total_rows = 0
    for d in dates:
        out = paths.data_raw / f"injuries_overrides_{d}.csv"
        rows = uniq[["team_tri","player"]].rename(columns={"team_tri":"team"}).copy()
        rows["status"] = "OUT"
        rows.to_csv(out, index=False)
        total_rows += len(rows)
        console.print({"date": str(d), "rows": int(len(rows)), "output": str(out)})
    console.print({"dates": f"{start_date}..{end_date}", "files": int(len(dates)), "rows": int(total_rows)})


@cli.command("backfill-injuries-from-excluded")
@click.option("--src-date", type=str, required=True, help="Source date of injuries_excluded_YYYY-MM-DD.csv")
@click.option("--start", "start_date", type=str, required=True, help="Start date YYYY-MM-DD")
@click.option("--end", "end_date", type=str, required=True, help="End date YYYY-MM-DD")
def backfill_injuries_from_excluded_cmd(src_date: str, start_date: str, end_date: str):
    """Generate per-day injuries_overrides_<date>.csv from an injuries_excluded_<src>.csv snapshot.

    This is a pragmatic backfill for short ranges: we take the excluded list (status OUT) from src-date diagnostics
    and apply it to each date in [start..end].
    """
    console.rule("Backfill Injury Overrides (from diagnostics)")
    try:
        s = pd.to_datetime(start_date).date(); e = pd.to_datetime(end_date).date(); sd = pd.to_datetime(src_date).date()
    except Exception:
        console.print("Invalid --src-date/--start/--end (YYYY-MM-DD)", style="red"); return
    if e < s:
        console.print("--end must be >= --start", style="red"); return
    src = paths.data_processed / f"injuries_excluded_{sd}.csv"
    if not src.exists():
        console.print(f"Diagnostics not found: {src}", style="red"); return
    df = pd.read_csv(src)
    if df is None or df.empty:
        console.print("Diagnostics file is empty", style="red"); return
    cols = set(c.lower() for c in df.columns)
    # Heuristics for column names
    team_col = next((c for c in df.columns if c.lower() in ("team","team_tri","team_abbr","teamabbr")), None)
    player_col = next((c for c in df.columns if c.lower() in ("player","player_name")), None)
    status_col = next((c for c in df.columns if c.lower() == "status"), None)
    if not (team_col and player_col and status_col):
        console.print("Diagnostics missing team/player/status columns", style="red"); return
    from .teams import to_tricode as _to_tri
    tmp = df.copy()
    tmp["team"] = tmp[team_col].astype(str).map(lambda x: _to_tri(str(x)))
    tmp["player"] = tmp[player_col].astype(str)
    tmp["status"] = tmp[status_col].astype(str).str.upper()
    tmp = tmp[tmp["status"] == "OUT"]
    tmp = tmp[tmp["team"].astype(str).str.len() == 3]
    tmp = tmp.drop_duplicates(subset=["team","player"])
    if tmp.empty:
        console.print("No OUT rows in diagnostics", style="yellow"); return
    dates = pd.date_range(s, e, freq="D").date
    total_rows = 0
    for d in dates:
        out = paths.data_raw / f"injuries_overrides_{d}.csv"
        rows = tmp[["team","player","status"]].copy()
        rows.to_csv(out, index=False)
        total_rows += len(rows)
        console.print({"date": str(d), "rows": int(len(rows)), "output": str(out)})
    console.print({"dates": f"{start_date}..{end_date}", "files": int(len(dates)), "rows": int(total_rows)})


@cli.command("evaluate-models")
@click.option("--start", "start", type=str, required=False, help="Start date YYYY-MM-DD")
@click.option("--end", "end", type=str, required=False, help="End date YYYY-MM-DD")
@click.option("--days", "days", type=int, default=30, show_default=True, help="If start/end not provided, evaluate last N days")
def evaluate_models_cmd(start: str | None, end: str | None, days: int):
    """Run the evaluation harness over a date range.

    Writes a rollup CSV to data/processed/metrics_eval_rollup.csv and prints a compact dict summary.
    """
    console.rule("Evaluate Models")
    try:
        script = paths.root / "tools" / "evaluate_models.py"
        if not script.exists():
            console.print(f"Missing evaluation script: {script}", style="red"); return
        args = [sys.executable, str(script)]
        if start and end:
            args += ["--start", str(start), "--end", str(end)]
        else:
            args += ["--days", str(int(days))]
        console.print({"run": " ".join(args)})
        cp = subprocess.run(args, capture_output=False, check=False)
        if cp.returncode != 0:
            console.print(f"Evaluation exited with code {cp.returncode}", style="red")
    except Exception as e:
        console.print(f"Evaluation failed: {e}", style="red")


@cli.command("evaluate-reliability")
@click.option("--start", "start", type=str, required=False, help="Start date YYYY-MM-DD")
@click.option("--end", "end", type=str, required=False, help="End date YYYY-MM-DD")
@click.option("--days", "days", type=int, default=60, show_default=True, help="If start/end not provided, evaluate last N days")
@click.option("--bins", "bins", type=int, default=10, show_default=True, help="Number of probability bins for reliability curve")
def evaluate_reliability_cmd(start: str | None, end: str | None, days: int, bins: int):
    """Compute reliability curves and write to processed metrics CSVs."""
    console.rule("Evaluate Reliability")
    try:
        script = paths.root / "tools" / "reliability.py"
        if not script.exists():
            console.print(f"Missing reliability script: {script}", style="red"); return
        args = [sys.executable, str(script)]
        if start and end:
            args += ["--start", str(start), "--end", str(end)]
        else:
            args += ["--days", str(int(days))]
        args += ["--bins", str(int(bins))]
        console.print({"run": " ".join(args)})
        cp = subprocess.run(args, capture_output=False, check=False)
        if cp.returncode != 0:
            console.print(f"Reliability exited with code {cp.returncode}", style="red")
    except Exception as e:
        console.print(f"Reliability failed: {e}", style="red")


@cli.command("evaluate-quarters")
@click.option("--start", "start", type=str, required=False, help="Start date YYYY-MM-DD")
@click.option("--end", "end", type=str, required=False, help="End date YYYY-MM-DD")
@click.option("--days", "days", type=int, default=30, show_default=True, help="If start/end not provided, evaluate last N days (ending at latest available)")
@click.option(
    "--source",
    "source",
    type=click.Choice(["both", "recon", "smart_sim"], case_sensitive=False),
    default="both",
    show_default=True,
    help="Which quarter dataset(s) to evaluate",
)
@click.option(
    "--smart-sim-path",
    "smart_sim_path",
    type=str,
    required=False,
    help="Optional explicit smart_sim_quarter_eval CSV path (defaults to latest)",
)
def evaluate_quarters_cmd(start: str | None, end: str | None, days: int, source: str, smart_sim_path: str | None):
    """Sanity-check quarter predictions vs actuals.

    - Quarter totals: reads data/processed/recon_quarters_*.csv
    - Team quarter scores: reads data/processed/smart_sim_quarter_eval_*.csv

    Writes CSV(s) + a JSON summary under data/processed/.
    """
    console.rule("Evaluate Quarters")
    try:
        script = paths.root / "tools" / "evaluate_quarters.py"
        if not script.exists():
            console.print(f"Missing quarter evaluation script: {script}", style="red");
            return
        args = [sys.executable, str(script), "--source", str(source).lower()]
        if start and end:
            args += ["--start", str(start), "--end", str(end)]
        else:
            args += ["--days", str(int(days))]
            if end:
                args += ["--end", str(end)]
        if smart_sim_path:
            args += ["--smart-sim-path", str(smart_sim_path)]
        console.print({"run": " ".join(args)})
        cp = subprocess.run(args, capture_output=False, check=False)
        if cp.returncode != 0:
            console.print(f"Quarter evaluation exited with code {cp.returncode}", style="red")
    except Exception as e:
        console.print(f"Quarter evaluation failed: {e}", style="red")


@cli.command("evaluate-sim-realism")
@click.option("--start", "start", type=str, required=False, help="Start date YYYY-MM-DD")
@click.option("--end", "end", type=str, required=False, help="End date YYYY-MM-DD (default: latest recon date)")
@click.option("--days", "days", type=int, default=30, show_default=True, help="If start/end not provided, evaluate last N days")
@click.option("--n-samples", "n_samples", type=int, default=2000, show_default=True, help="Sim samples per game")
def evaluate_sim_realism_cmd(start: str | None, end: str | None, days: int, n_samples: int):
    """Backtest sim realism (total/margin/win calibration) over historical games.

    Writes per-game CSV + JSON summary under data/processed/.
    """
    console.rule("Evaluate Sim Realism")
    try:
        script = paths.root / "tools" / "evaluate_sim_realism.py"
        if not script.exists():
            console.print(f"Missing sim realism script: {script}", style="red"); return
        args = [sys.executable, str(script), "--n-samples", str(int(n_samples))]
        if start and end:
            args += ["--start", str(start), "--end", str(end)]
        else:
            args += ["--days", str(int(days))]
            if end:
                args += ["--end", str(end)]
        console.print({"run": " ".join(args)})
        cp = subprocess.run(args, capture_output=False, check=False)
        if cp.returncode != 0:
            console.print(f"Sim realism exited with code {cp.returncode}", style="red")
    except Exception as e:
        console.print(f"Sim realism failed: {e}", style="red")


@cli.command("evaluate-connected-realism")
@click.option("--start", "start", type=str, required=False, help="Start date YYYY-MM-DD")
@click.option("--end", "end", type=str, required=False, help="End date YYYY-MM-DD (default: latest player log date)")
@click.option("--days", "days", type=int, default=14, show_default=True, help="If start/end not provided, evaluate last N days")
@click.option("--n-quarter-samples", "n_quarter_samples", type=int, default=3500, show_default=True, help="Quarter sim samples per game")
@click.option("--n-connected-samples", "n_connected_samples", type=int, default=1200, show_default=True, help="Connected sim samples per game")
@click.option("--minutes-lookback-days", "minutes_lookback_days", type=int, default=21, show_default=True, help="Lookback window for minutes priors")
@click.option("--top-k", "top_k", type=int, default=8, show_default=True, help="Score top-K players by actual minutes")
@click.option(
    "--hist-exp-blend-alpha",
    "hist_exp_blend_alpha",
    type=float,
    default=0.0,
    show_default=True,
    help="Optional: blend rotations-history expected minutes into bench minutes (0 disables)",
)
@click.option(
    "--hist-exp-blend-max-cov",
    "hist_exp_blend_max_cov",
    type=float,
    default=0.67,
    show_default=True,
    help="Only apply hist-exp blending when minutes_expected_coverage <= this threshold",
)
@click.option(
    "--coach-rotation-alpha",
    "coach_rotation_alpha",
    type=float,
    default=0.0,
    show_default=True,
    help="Optional: scale coach/rotation shaping from rotation priors (0 disables)",
)
@click.option(
    "--rotation-shock-alpha",
    "rotation_shock_alpha",
    type=float,
    default=0.0,
    show_default=True,
    help="Optional: detect rotation shock and blend minutes toward priors (0 disables)",
)
@click.option(
    "--garbage-time-alpha",
    "garbage_time_alpha",
    type=float,
    default=0.0,
    show_default=True,
    help="Optional: shift minutes from starters to bench when blowout likelihood is high (0 disables)",
)
@click.option(
    "--guardrail-alpha",
    "guardrail_alpha",
    type=float,
    default=0.0,
    show_default=True,
    help="Optional: softly anchor quarter samples to model priors (0 disables)",
)
@click.option(
    "--guardrail-max-scale",
    "guardrail_max_scale",
    type=float,
    default=0.10,
    show_default=True,
    help="Max |scale-1| per team/quarter when applying guardrails",
)
@click.option(
    "--event-level",
    "event_level",
    is_flag=True,
    help="Use event-level (possession) stat-mix for representative box score (keeps points allocation)",
)
@click.option("--skip-ot", "skip_ot", is_flag=True, help="Skip likely OT games (team minutes >245)")
@click.option("--seed", "seed", type=int, default=1, show_default=True, help="Seed base")
@click.option("--out-games-csv", "out_games_csv", type=str, required=False, help="Override output games CSV path")
@click.option("--out-players-csv", "out_players_csv", type=str, required=False, help="Override output players CSV path")
@click.option("--out-json", "out_json", type=str, required=False, help="Override output JSON summary path")
def evaluate_connected_realism_cmd(
    start: str | None,
    end: str | None,
    days: int,
    n_quarter_samples: int,
    n_connected_samples: int,
    minutes_lookback_days: int,
    top_k: int,
    hist_exp_blend_alpha: float,
    hist_exp_blend_max_cov: float,
    coach_rotation_alpha: float,
    rotation_shock_alpha: float,
    garbage_time_alpha: float,
    guardrail_alpha: float,
    guardrail_max_scale: float,
    event_level: bool,
    skip_ot: bool,
    seed: int,
    out_games_csv: str | None,
    out_players_csv: str | None,
    out_json: str | None,
):
    """Backtest connected (player boxscore) sim realism vs player_logs.csv.

    Writes per-game CSV, per-player CSV, and a JSON summary under data/processed/.
    """
    console.rule("Evaluate Connected Realism")
    try:
        script = paths.root / "tools" / "evaluate_connected_realism.py"
        if not script.exists():
            console.print(f"Missing connected realism script: {script}", style="red"); return
        args = [
            sys.executable,
            str(script),
            "--n-quarter-samples",
            str(int(n_quarter_samples)),
            "--n-connected-samples",
            str(int(n_connected_samples)),
            "--minutes-lookback-days",
            str(int(minutes_lookback_days)),
            "--top-k",
            str(int(top_k)),
            "--hist-exp-blend-alpha",
            str(float(hist_exp_blend_alpha or 0.0)),
            "--hist-exp-blend-max-cov",
            str(float(hist_exp_blend_max_cov if hist_exp_blend_max_cov is not None else 0.67)),
            "--coach-rotation-alpha",
            str(float(coach_rotation_alpha or 0.0)),
            "--rotation-shock-alpha",
            str(float(rotation_shock_alpha or 0.0)),
            "--garbage-time-alpha",
            str(float(garbage_time_alpha or 0.0)),
            "--guardrail-alpha",
            str(float(guardrail_alpha or 0.0)),
            "--guardrail-max-scale",
            str(float(guardrail_max_scale if guardrail_max_scale is not None else 0.10)),
            "--seed",
            str(int(seed)),
        ]
        if event_level:
            args += ["--event-level"]
        if skip_ot:
            args += ["--skip-ot"]
        if out_games_csv:
            args += ["--out-games-csv", str(out_games_csv)]
        if out_players_csv:
            args += ["--out-players-csv", str(out_players_csv)]
        if out_json:
            args += ["--out-json", str(out_json)]
        if start and end:
            args += ["--start", str(start), "--end", str(end)]
        else:
            args += ["--days", str(int(days))]
            if end:
                args += ["--end", str(end)]
        console.print({"run": " ".join(args)})
        cp = subprocess.run(args, capture_output=False, check=False)
        if cp.returncode != 0:
            console.print(f"Connected realism exited with code {cp.returncode}", style="red")
    except Exception as e:
        console.print(f"Connected realism failed: {e}", style="red")

@cli.command("evaluate-props-lite")
@click.option("--start", type=str, required=False, help="Start date YYYY-MM-DD")
@click.option("--end", type=str, required=False, help="End date YYYY-MM-DD")
@click.option("--days", type=int, default=14, show_default=True, help="If start/end not provided, evaluate last N days")
def evaluate_props_lite_cmd(start: str | None, end: str | None, days: int):
    """Lightweight props probability calibration (scaffold)."""
    console.rule("Evaluate Props (Lite)")
    try:
        script = paths.root / "tools" / "evaluate_props.py"
        if not script.exists():
            console.print(f"Missing props lite script: {script}", style="red"); return
        args = [sys.executable, str(script)]
        if start and end:
            args += ["--start", str(start), "--end", str(end)]
        else:
            args += ["--days", str(int(days))]
        console.print({"run": " ".join(args)})
        cp = subprocess.run(args, capture_output=False, check=False)
        if cp.returncode != 0:
            console.print(f"Props lite exited with code {cp.returncode}", style="red")
    except Exception as e:
        console.print(f"Props lite failed: {e}", style="red")


@cli.command("recommend-picks")
@click.option("--date", "date_str", type=str, required=True, help="Slate date YYYY-MM-DD")
@click.option("--topN", type=int, default=10, show_default=True, help="Max picks per market type")
@click.option("--minScore", type=float, default=0.15, show_default=True, help="Minimum confidence score threshold")
@click.option("--minAtsEdge", type=float, default=0.05, show_default=True, help="Minimum ATS probability edge vs market (no-vig)")
@click.option("--minAtsEV", type=float, default=0.00, show_default=True, help="Minimum ATS expected value (ROI per $1)")
@click.option("--atsBlend", type=float, default=0.25, show_default=True, help="ATS prob blend weight: w*model + (1-w)*market")
@click.option("--minTotalEdge", type=float, default=0.02, show_default=True, help="Minimum total probability edge vs market (no-vig)")
@click.option("--minTotalEV", type=float, default=0.00, show_default=True, help="Minimum total expected value (ROI per $1)")
@click.option("--totalsBlend", type=float, default=0.10, show_default=True, help="Totals prob blend weight: w*model + (1-w)*market")
def recommend_picks_cmd(
    date_str: str,
    topn: int,
    minscore: float,
    minatsedge: float,
    minatsev: float,
    atsblend: float,
    mintotaledge: float,
    mintotalev: float,
    totalsblend: float,
):
    """Generate daily high-confidence picks across moneyline, spread, and totals.

    Reads predictions and odds/closing lines for the date, applies reliability-adjusted confidence,
    and writes picks CSV to data/processed/picks_<date>.csv.
    """
    console.rule("Recommend Picks")
    try:
        _ = pd.to_datetime(date_str).date()
    except Exception:
        console.print("Invalid --date. Use YYYY-MM-DD.", style="red"); return
    try:
        script = paths.root / "tools" / "recommend_picks.py"
        if not script.exists():
            console.print(f"Missing recommend script: {script}", style="red"); return
        args = [
            sys.executable,
            str(script),
            "--date", str(date_str),
            "--topN", str(int(topn)),
            "--minScore", str(float(minscore)),
            "--minAtsEdge", str(float(minatsedge)),
            "--minAtsEV", str(float(minatsev)),
            "--atsBlend", str(float(atsblend)),
            "--minTotalEdge", str(float(mintotaledge)),
            "--minTotalEV", str(float(mintotalev)),
            "--totalsBlend", str(float(totalsblend)),
        ]
        console.print({"run": " ".join(args)})
        cp = subprocess.run(args, capture_output=False, check=False)
        if cp.returncode != 0:
            console.print(f"Recommend exited with code {cp.returncode}", style="red")
    except Exception as e:
        console.print(f"Recommend failed: {e}", style="red")


@cli.command()
@click.option("--season", type=str, default="2025-26", show_default=True, help="NBA season string, e.g., 2025-26")
def fetch_rosters_cmd(season: str):
    """Fetch all team rosters for a season and save under data/processed/rosters_*.{csv,parquet}."""
    console.rule("Fetch Rosters")
    try:
        df = fetch_rosters(season=season)
        console.print({"rows": 0 if df is None else int(len(df)), "season": season})
    except Exception as e:
        console.print(f"Failed to fetch rosters: {e}", style="red")

@cli.command()
@click.option("--years", default=10, help="Number of past seasons to fetch")
@click.option("--with-periods/--no-periods", default=True, help="Fetch quarter/OT line scores (slower)")
@click.option("--verbose", is_flag=True, default=False, help="Print progress while fetching")
@click.option("--rate-delay", type=float, default=0.6, help="Delay between requests in seconds")
@click.option("--max-workers", type=int, default=1, help="Concurrent workers for period fetch (1 = serial)")
def fetch(years: int, with_periods: bool, verbose: bool, rate_delay: float, max_workers: int):
    """Fetch last N seasons from NBA Stats API"""
    console.rule("Fetch data")
    df = fetch_games_nba_api(last_n=years, with_periods=with_periods, verbose=verbose, rate_delay=rate_delay, max_workers=max_workers)
    console.print(f"Saved raw to {paths.data_raw}")
    console.print(df.head(3))


@cli.command("build-features")
def build_features_cmd():
    """Build features and save processed dataset"""
    console.rule("Build features")
    # Prefer CSV (engine-free) then parquet; NBA API outputs
    candidates = [
        paths.data_raw / "games_nba_api.csv",
        paths.data_raw / "games_nba_api.parquet",
    ]
    raw = next((p for p in candidates if p.exists()), None)
    if raw is None:
        console.print("No raw games file found. Run fetch first.", style="red")
        return
    # Read raw with fallback if parquet engine is missing
    if raw.suffix == ".parquet":
        try:
            df = pd.read_parquet(raw)
        except Exception:
            alt = paths.data_raw / "games_nba_api.csv"
            if alt.exists():
                df = pd.read_csv(alt)
            else:
                raise
    else:
        df = pd.read_csv(raw)
    feats = build_features(df)
    out_dir = paths.data_processed
    out_dir.mkdir(parents=True, exist_ok=True)
    # Always write CSV for maximum compatibility (no extra deps)
    out_csv = out_dir / "features.csv"
    feats.to_csv(out_csv, index=False)
    wrote_parquet = False
    try:
        out_pq = out_dir / "features.parquet"
        feats.to_parquet(out_pq, index=False)
        wrote_parquet = True
        console.print(f"Saved features to {out_pq}")
    except Exception as e:
        console.print(f"Parquet write skipped (engine missing): {e}", style="yellow")
        console.print(f"Saved features to {out_csv}")


@cli.command()
def train():
    """Train baseline models"""
    from .train import train_models  # Import here to avoid sklearn dependency at module level
    console.rule("Train models")
    feats_path = paths.data_processed / "features.parquet"
    df = pd.read_parquet(feats_path)
    metrics = train_models(df)
    console.print(metrics)


@cli.command("fetch-schedule")
@click.option("--season", "season", type=str, default="2025-26", show_default=True, help="Season string to export (currently only 2025-26 supported)")
def fetch_schedule_cmd(season: str):
    """Fetch the NBA schedule for 2025-26 from the public CDN and export JSON/CSV for the frontend."""
    console.rule("Fetch Schedule")
    if season != "2025-26":
        console.print("Only 2025-26 is supported right now; ignoring provided season.", style="yellow")
    try:
        df = fetch_schedule_2025_26()
    except Exception as e:
        console.print(f"Failed to fetch schedule: {e}", style="red"); return
    out_dir = paths.data_processed
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "schedule_2025_26.json"
    csv_path = out_dir / "schedule_2025_26.csv"
    try:
        # Save compact JSON list
        df.to_json(json_path, orient="records", date_format="iso")
        # Save CSV
        df.to_csv(csv_path, index=False)
        console.print(f"Saved schedule to {json_path} and {csv_path}")
        # Quick stats for sanity
        total_games = len(df)
        dates = df['date_utc'].dropna().unique()
        console.print({"rows": int(total_games), "unique_dates": int(len(dates))})
    except Exception as e:
        console.print(f"Failed to write schedule files: {e}", style="red")


@cli.command("fetch-rosters")
@click.option("--season", type=str, default="2025-26", show_default=True, help="Season string like 2025-26")
def fetch_rosters_cmd(season: str):
    """Fetch team rosters for the given season and save processed CSV/Parquet."""
    console.rule("Fetch Rosters")
    try:
        df = fetch_rosters(season=season)
    except Exception as e:
        console.print(f"Failed to fetch rosters: {e}", style="red"); return
    if df.empty:
        console.print("No roster data returned.", style="yellow"); return
    console.print({"rows": int(len(df)), "teams": int(df['TEAM_ID'].nunique())})


@cli.command("fetch-player-logs")
@click.option("--seasons", type=str, required=True, help="Comma-separated seasons like 2023-24,2024-25,2025-26")
def fetch_player_logs_cmd(seasons: str):
    """Fetch player game logs for the given seasons and save to processed folder."""
    console.rule("Fetch Player Logs")
    season_list = [s.strip() for s in seasons.split(',') if s.strip()]
    try:
        df = fetch_player_logs(season_list)
    except Exception as e:
        raise click.ClickException(f"Failed to fetch player logs: {e}") from e
    if df.empty:
        raise click.ClickException("No player logs returned.")
    if df.attrs.get("source") == "boxscores_history":
        console.print("Using boxscores history fallback for player logs.", style="yellow")
    elif df.attrs.get("source") == "espn_recent_boxscores":
        console.print("Using recent ESPN boxscore fallback for player logs.", style="yellow")
    console.print({"rows": int(len(df)), "players": int(df['PLAYER_ID'].nunique()), "games": int(df['GAME_ID'].nunique())})


@cli.command("fetch-pbp")
@click.option("--date", "date_str", type=str, required=True, help="Date YYYY-MM-DD (US/Eastern slate)")
@click.option("--include-live/--finals-only", "include_live", default=False, show_default=True, help="Include in-progress games as well (may be partial logs)")
@click.option("--rate-delay", type=float, default=0.35, show_default=True, help="Delay between game fetches (seconds)")
def fetch_pbp_cmd(date_str: str, include_live: bool, rate_delay: float):
    """Fetch PlayByPlay logs for all games on a date and write CSVs under data/processed.

    - Per-game files under data/processed/pbp/pbp_<gameId>.csv
    - Combined file under data/processed/pbp_<date>.csv
    """
    console.rule("Fetch PBP Logs")
    try:
        df, gids = fetch_pbp_for_date(date_str, only_final=(not include_live), rate_delay=rate_delay)
        console.print({"date": date_str, "games": len(gids), "rows": 0 if df is None else int(len(df))})
    except Exception as e:
        console.print(f"Failed to fetch PBP: {e}", style="red")

@cli.command("fetch-pbp-r")
@click.option("--date", "date_str", type=str, required=False, help="Target date YYYY-MM-DD")
@click.option("--start", type=str, required=False, help="Start date YYYY-MM-DD")
@click.option("--end", type=str, required=False, help="End date YYYY-MM-DD")
def fetch_pbp_r_cmd(date_str: str | None, start: str | None, end: str | None):
    """Fetch PBP using an R helper (hoopR preferred, nbastatR fallback).

    Requires R installed locally and either hoopR or nbastatR available in R.
    Writes per-game CSVs to data/processed/pbp and combined per-date CSVs.
    """
    console.rule("Fetch PBP via R")
    import subprocess, sys, shutil
    from pathlib import Path
    script = str(paths.root / "scripts" / "pbp_fetch_nbastatr.R")
    # Prefer Rscript on PATH; otherwise probe common install locations
    rscript = shutil.which("Rscript")
    if not rscript:
        candidates = []
        for base in (Path("C:/Program Files/R"), Path("C:/Program Files (x86)/R")):
            try:
                for p in base.glob("R-*/bin/Rscript.exe"):
                    candidates.append(str(p))
            except Exception:
                pass
        rscript = candidates[0] if candidates else None
    if not rscript:
        console.print("Rscript not found. Please install R or add Rscript.exe to PATH.", style="red")
        return
    args = [rscript, script]
    if date_str:
        args += ["--date", date_str]
    if start and end and not date_str:
        args += ["--start", start, "--end", end]
    try:
        r = subprocess.run(args, cwd=str(paths.root), capture_output=True, text=True, check=False)
        console.print({"cmd": " ".join(args), "returncode": r.returncode})
        if r.stdout:
            console.print(r.stdout)
        if r.stderr:
            console.print(r.stderr, style="yellow")
    except FileNotFoundError:
        console.print("Rscript not found. Please install R and ensure Rscript is on PATH.", style="red")
    except Exception as e:
        console.print(f"Failed to run R script: {e}", style="red")


@cli.command("fetch-boxscores")
@click.option("--date", "date_str", type=str, required=True, help="Date YYYY-MM-DD (US/Eastern slate)")
@click.option("--include-live/--finals-only", "include_live", default=False, show_default=True, help="Include in-progress games as well (may be partial)")
@click.option("--rate-delay", type=float, default=0.35, show_default=True, help="Delay between game fetches (seconds)")
def fetch_boxscores_cmd(date_str: str, include_live: bool, rate_delay: float):
    """Fetch BoxScoreTraditionalV3 player stats for all games on a date.

    - Per-game files under data/processed/boxscores/boxscore_<gameId>.csv
    - Combined file under data/processed/boxscores_<date>.csv
    """
    console.rule("Fetch Boxscores")
    try:
        df, gids = fetch_boxscores_for_date(date_str, only_final=(not include_live), rate_delay=rate_delay)
        console.print({"date": date_str, "games": len(gids), "rows": 0 if df is None else int(len(df))})
    except Exception as e:
        console.print(f"Failed to fetch boxscores: {e}", style="red")


@cli.command("update-boxscores-history")
@click.option("--date", "date_str", type=str, required=True, help="Date YYYY-MM-DD (US/Eastern slate)")
@click.option("--include-live/--finals-only", "include_live", default=False, show_default=True, help="Include in-progress games as well (may be partial)")
@click.option("--rate-delay", type=float, default=0.35, show_default=True, help="Delay between game fetches (seconds)")
def update_boxscores_history_cmd(date_str: str, include_live: bool, rate_delay: float):
    """Fetch boxscores for a date and append into data/processed/boxscores_history.*."""
    console.rule("Update Boxscores History")
    try:
        info = update_boxscores_history_for_date(date_str, include_live=include_live, rate_delay=rate_delay)
        console.print(info)
    except Exception as e:
        console.print(f"Failed to update boxscores history: {e}", style="red")


@cli.command("fetch-pbp-espn")
@click.option("--date", "date_str", type=str, required=True, help="Date YYYY-MM-DD (US/Eastern slate)")
@click.option("--include-live/--finals-only", "include_live", default=False, show_default=True, help="Include in-progress games as well (may be partial)")
@click.option("--rate-delay", type=float, default=0.25, show_default=True, help="Delay between event fetches (seconds)")
def fetch_pbp_espn_cmd(date_str: str, include_live: bool, rate_delay: float):
    """Fetch play-by-play via ESPN summary endpoint.

    - Per-game files under data/processed/pbp_espn/pbp_espn_<gameId>.csv
    - Combined file under data/processed/pbp_espn_<date>.csv
    """
    console.rule("Fetch PBP (ESPN)")
    try:
        df, gids = fetch_pbp_espn_for_date(date_str, only_final=(not include_live), rate_delay=rate_delay)
        console.print({"date": date_str, "games": len(gids), "rows": 0 if df is None else int(len(df))})
    except Exception as e:
        console.print(f"Failed to fetch ESPN PBP: {e}", style="red")


@cli.command("update-pbp-espn-history")
@click.option("--date", "date_str", type=str, required=True, help="Date YYYY-MM-DD (US/Eastern slate)")
@click.option("--include-live/--finals-only", "include_live", default=False, show_default=True, help="Include in-progress games as well (may be partial)")
@click.option("--rate-delay", type=float, default=0.25, show_default=True, help="Delay between event fetches (seconds)")
def update_pbp_espn_history_cmd(date_str: str, include_live: bool, rate_delay: float):
    """Fetch ESPN PBP for a date and append into data/processed/pbp_espn_history.*."""
    console.rule("Update PBP History (ESPN)")
    try:
        info = update_pbp_espn_history_for_date(date_str, include_live=include_live, rate_delay=rate_delay)
        console.print(info)
    except Exception as e:
        console.print(f"Failed to update ESPN PBP history: {e}", style="red")


@cli.command("backfill-pbp-espn-history")
@click.option("--start", "start_date", type=str, required=True, help="Start date YYYY-MM-DD")
@click.option("--end", "end_date", type=str, required=True, help="End date YYYY-MM-DD")
@click.option("--finals-only/--include-live", "finals_only", default=True, show_default=True, help="Fetch only final games (recommended)")
@click.option("--rate-delay", type=float, default=0.25, show_default=True, help="Delay between event fetches (seconds)")
def backfill_pbp_espn_history_cmd(start_date: str, end_date: str, finals_only: bool, rate_delay: float):
    """Backfill ESPN PBP history over a date range."""
    console.rule("Backfill PBP History (ESPN)")
    try:
        df = backfill_pbp_espn_history(start_date, end_date, finals_only=finals_only, rate_delay=rate_delay)
        console.print({"start": start_date, "end": end_date, "days": int(len(df))})
    except Exception as e:
        console.print(f"Failed to backfill ESPN PBP history: {e}", style="red")


@cli.command("write-rotation-priors")
@click.option("--lookback-days", type=int, default=60, show_default=True, help="Lookback window ending at latest date in history")
@click.option("--min-games", type=int, default=10, show_default=True, help="Minimum games required per team")
def write_rotation_priors_cmd(lookback_days: int, min_games: int):
    """Compute and write rotation priors derived from ESPN substitution events."""
    console.rule("Write Rotation Priors")
    try:
        info = write_rotation_priors(lookback_days=lookback_days, min_games=min_games)
        console.print(info)
    except Exception as e:
        raise click.ClickException(f"Failed to write rotation priors: {e}")


@cli.command("update-rotations-espn-history")
@click.option("--date", "date_str", type=str, required=True, help="Date YYYY-MM-DD (US/Eastern slate)")
@click.option("--rate-delay", type=float, default=0.25, show_default=True, help="Delay between event fetches (seconds)")
def update_rotations_espn_history_cmd(date_str: str, rate_delay: float):
    """Build full-game rotation stints + pair-minutes from ESPN substitutions and append to history."""
    console.rule("Update Rotations History (ESPN)")
    try:
        info = update_rotations_history_for_date(date_str, rate_delay=rate_delay)
        console.print(info)
    except Exception as e:
        console.print(f"Failed to update rotations history: {e}", style="red")


@cli.command("backfill-rotations-espn-history")
@click.option("--start", "start_date", type=str, required=False, help="Start date YYYY-MM-DD; default = season start (Oct 1) of current season")
@click.option("--end", "end_date", type=str, required=False, help="End date YYYY-MM-DD; default = today (local)")
@click.option("--rate-delay", type=float, default=0.25, show_default=True, help="Delay between event fetches (seconds)")
@click.option(
    "--resume-file",
    type=click.Path(dir_okay=False, writable=True),
    default=str(paths.data_processed / "_rotations_espn_resume.json"),
    show_default=True,
    help="JSON resume file tracking last completed date",
)
@click.option("--ignore-resume", is_flag=True, default=False, show_default=True, help="Ignore the resume file and process the full requested range")
@click.option("--max-days", type=int, default=None, help="Process at most this many days (smoke test)")
def backfill_rotations_espn_history_cmd(start_date: str | None, end_date: str | None, rate_delay: float, resume_file: str, ignore_resume: bool, max_days: int | None):
    """Backfill ESPN rotations (stints, pairs, play_context) over a date range.

    Defaults to the full current season (Oct 1 .. today).
    """
    console.rule("Backfill Rotations History (ESPN)")
    import datetime as _dt

    today = _dt.date.today()
    if end_date is None:
        e = today - _dt.timedelta(days=1)
    else:
        try:
            e = pd.to_datetime(end_date).date()
        except Exception:
            console.print("Invalid --end (YYYY-MM-DD)", style="red"); return
    if start_date is None:
        yr = today.year
        if today.month < 7:
            yr -= 1
        s = _dt.date(yr, 10, 1)
    else:
        try:
            s = pd.to_datetime(start_date).date()
        except Exception:
            console.print("Invalid --start (YYYY-MM-DD)", style="red"); return
    if e < s:
        console.print("--end must be >= --start", style="red"); return

    dates = pd.date_range(s, e, freq="D").strftime("%Y-%m-%d").tolist()
    if max_days is not None:
        try:
            dates = dates[: int(max_days)]
        except Exception:
            pass

    # Resume support
    last_done = None
    if not bool(ignore_resume):
        try:
            import json as _json
            rp = Path(resume_file)
            if rp.exists():
                last_done = (_json.loads(rp.read_text(encoding="utf-8")) or {}).get("last_completed")
                last_done = str(last_done or "").strip() or None
        except Exception:
            last_done = None

        if last_done:
            dates = [d for d in dates if str(d) > str(last_done)]

    ok = 0
    fail: list[dict[str, Any]] = []
    for ds in track(dates, description="Rotations backfill"):
        try:
            info = update_rotations_history_for_date(ds, rate_delay=rate_delay)
            err = info.get("error")
            if err == "no_games":
                ok += 1
            elif err:
                fail.append({"date": ds, "error": str(err)})
            else:
                # Guard: some ESPN summaries return no plays even for finished games.
                if int(info.get("games") or 0) > 0 and int(info.get("rows_plays_ctx") or 0) == 0:
                    fail.append({"date": ds, "error": "no_plays_ctx", "games": int(info.get("games") or 0)})
                else:
                    ok += 1

            # update resume after each day (success or fail) so we can restart quickly
            try:
                import json as _json
                Path(resume_file).write_text(_json.dumps({"last_completed": ds}), encoding="utf-8")
            except Exception:
                pass
        except Exception:
            fail.append({"date": ds, "error": "exception"})

    console.print({"start": str(s), "end": str(e), "days": int(len(dates)), "ok": int(ok), "fail": int(len(fail))})
    if fail:
        try:
            pd.DataFrame(fail).to_csv(paths.data_processed / "rotations_espn_backfill_failures.csv", index=False)
        except Exception:
            pass
        console.print({"failures_file": str(paths.data_processed / "rotations_espn_backfill_failures.csv")})


@cli.command("build-lineup-teammate-effects")
@click.option("--start", "start_date", type=str, required=False, help="Start date YYYY-MM-DD; default = season start (Oct 1) of current season")
@click.option("--end", "end_date", type=str, required=False, help="End date YYYY-MM-DD; default = today (local)")
@click.option("--min-minutes-together", type=float, default=25.0, show_default=True, help="Only keep teammate pairs with at least this many minutes together")
def build_lineup_teammate_effects_cmd(start_date: str | None, end_date: str | None, min_minutes_together: float):
    """Build lineup-conditioned teammate effect tables for the requested range.

    Requires that `play_context_history.*`, `rotation_stints_history.*`, and `pair_minutes_history.*` are populated.
    """
    console.rule("Build Lineup Teammate Effects")
    try:
        info = build_lineup_teammate_effects(start_date=start_date, end_date=end_date, min_minutes_together=min_minutes_together)
        console.print(info)
    except Exception as e:
        console.print(f"Failed to build teammate effects: {e}", style="red")


@cli.command("backfill-pbp")
@click.option("--start", "start_date", type=str, required=True, help="Start date YYYY-MM-DD")
@click.option("--end", "end_date", type=str, required=True, help="End date YYYY-MM-DD")
@click.option("--finals-only/--include-live", "finals_only", default=True, show_default=True, help="Fetch only final games (recommended)")
@click.option("--rate-delay", type=float, default=0.35, show_default=True, help="Delay between requests")
def backfill_pbp_cmd(start_date: str, end_date: str, finals_only: bool, rate_delay: float):
    """Backfill PBP logs over a date range, skipping dates that already exist."""
    console.rule("Backfill PBP")
    try:
        df = backfill_pbp(start_date, end_date, only_final=finals_only, rate_delay=rate_delay)
        rows = 0 if df is None else int(len(df))
        console.print({"start": start_date, "end": end_date, "rows": rows})
    except Exception as e:
        console.print(f"Failed to backfill PBP: {e}", style="red")


@cli.command("backfill-boxscores")
@click.option("--start", "start_date", type=str, required=True, help="Start date YYYY-MM-DD")
@click.option("--end", "end_date", type=str, required=True, help="End date YYYY-MM-DD")
@click.option("--finals-only/--include-live", "finals_only", default=True, show_default=True, help="Fetch only final games (recommended)")
@click.option("--rate-delay", type=float, default=0.35, show_default=True, help="Delay between requests")
def backfill_boxscores_cmd(start_date: str, end_date: str, finals_only: bool, rate_delay: float):
    """Backfill boxscores over a date range, skipping dates that already exist."""
    console.rule("Backfill Boxscores")
    try:
        df = backfill_boxscores(start_date, end_date, only_final=finals_only, rate_delay=rate_delay)
        rows = 0 if df is None else int(len(df))
        console.print({"start": start_date, "end": end_date, "rows": rows})
    except Exception as e:
        console.print(f"Failed to backfill boxscores: {e}", style="red")


@cli.command("backfill-scoreboard")
@click.option("--seasons", type=str, required=True, help="Comma-separated season end years (e.g., 2018,2019,2020)")
@click.option("--rate-delay", type=float, default=0.8, help="Delay between day requests in seconds")
@click.option("--day-limit", type=int, default=None, help="Process at most this many days per season (for smoke tests)")
@click.option("--resume-file", type=click.Path(dir_okay=False, writable=True), default=str(paths.data_raw / "_scoreboard_resume.json"), help="Path to JSON resume file")
@click.option("--verbose", is_flag=True, default=False, help="Print progress while backfilling")
def backfill_scoreboard_cmd(seasons: str, rate_delay: float, day_limit: int | None, resume_file: str, verbose: bool):
    """Backfill games (with per-periods) via ScoreboardV2 day-by-day with resume support."""
    console.rule("Backfill (ScoreboardV2)")
    try:
        season_list = [int(s.strip()) for s in seasons.split(',') if s.strip()]
    except Exception:
        console.print("Invalid seasons list", style="red"); return
    df = backfill_scoreboard(seasons=season_list, rate_delay=rate_delay, verbose=verbose, day_limit=day_limit, resume_file=resume_file)
    console.print(f"Backfill complete. Raw rows={len(df)}")


@cli.command("finals-export")
@click.option("--date", "date_str", type=str, required=False, help="Single date YYYY-MM-DD")
@click.option("--since", "since_str", type=str, required=False, help="Start date YYYY-MM-DD (inclusive)")
@click.option("--until", "until_str", type=str, required=False, help="End date YYYY-MM-DD (inclusive; defaults to today)")
def finals_export_cmd(date_str: str | None, since_str: str | None, until_str: str | None):
    """Export final scores to data/processed/finals_<date>.csv for a date or date range.

    Source order: nba_api ScoreboardV2 -> NBA CDN -> ESPN, with +/- 1 day fallback.
    """
    console.rule("Export Finals")
    dates: list[str] = []
    if date_str:
        dates = [date_str]
    else:
        try:
            from datetime import datetime as _dt, timedelta as _td
            if not since_str:
                console.print("Provide --date or --since/--until", style="red"); return
            if not until_str:
                until_str = _dt.utcnow().date().isoformat()
            ds = _dt.strptime(since_str, "%Y-%m-%d").date()
            de = _dt.strptime(until_str, "%Y-%m-%d").date()
            if de < ds:
                ds, de = de, ds
            cur = ds
            while cur <= de:
                dates.append(cur.isoformat())
                cur += _td(days=1)
        except Exception as e:
            console.print(f"Invalid dates: {e}", style="red"); return
    out = []
    for d in dates:
        try:
            path, n = write_finals_csv(d)
            out.append({"date": d, "path": str(path), "rows": int(n)})
        except Exception as e:
            out.append({"date": d, "error": str(e)})
    console.print({"count": len(out), "items": out})


@cli.command("build-league-status")
@click.option("--date", "date_str", type=str, required=True, help="Date YYYY-MM-DD to build league_status_<date>.csv")
def build_league_status_cmd(date_str: str):
    """Build unified league roster+injury status for a date and write data/processed/league_status_<date>.csv."""
    console.rule("Build League Status")
    try:
        df = build_league_status(date_str)
        rows = 0 if df is None else int(len(df))
        out_path = (paths.data_processed / f"league_status_{date_str}.csv")
        exists = out_path.exists()
        console.print({"date": date_str, "rows": rows, "path": str(out_path), "wrote_file": bool(exists)})
        if not exists:
            raise SystemExit(2)
    except Exception as e:
        console.print(f"Failed to build league status: {e}", style="red")
        raise SystemExit(1)


@cli.command("roster-sanity")
@click.option("--date", "date_str", type=str, required=True, help="Date YYYY-MM-DD to validate league_status roster sanity")
@click.option("--min-total-roster-per-team", type=int, default=10, show_default=True, help="Fail if a slate team has fewer than this many total roster rows")
@click.option("--min-playing-today-per-team", type=int, default=8, show_default=True, help="Fail if a slate team has fewer than this many playing_today==True players")
@click.option("--max-team-mismatches-in-props", type=int, default=0, show_default=True, help="Fail if props_predictions team mismatches vs league_status exceeds this")
def roster_sanity_cmd(
    date_str: str,
    min_total_roster_per_team: int,
    min_playing_today_per_team: int,
    max_team_mismatches_in_props: int,
):
    """Fail-fast roster sanity check using league_status_<date>.csv.

    Intended to catch obvious roster/team mapping breakage (trades/waives, stale overrides,
    missing slate teams) before we run any expensive sims/predictions.
    """
    console.rule("Roster Sanity")
    res = roster_sanity_check(
        date_str,
        min_total_roster_per_team=int(min_total_roster_per_team),
        min_playing_today_per_team=int(min_playing_today_per_team),
        max_team_mismatches_in_props=int(max_team_mismatches_in_props),
    )
    console.print({
        "date": res.date,
        "ok": bool(res.ok),
        "issues": res.issues,
        "summary": res.summary,
    })
    if not res.ok:
        raise SystemExit(2)


@cli.command("check-dressed")
@click.option("--date", "date_str", type=str, required=True, help="Date YYYY-MM-DD to build/check dressed_players_<date>.csv")
@click.option("--min-dressed-per-team", type=int, default=8, show_default=True, help="Fail if a slate team has fewer than this many expected dressed players")
@click.option("--min-total-roster-per-team", type=int, default=10, show_default=True, help="Fail if a slate team has fewer than this many total roster rows")
def check_dressed_cmd(date_str: str, min_dressed_per_team: int, min_total_roster_per_team: int):
    """First-step gate: build an 'expected dressed to play' list for today's slate.

    Writes:
      - data/processed/dressed_players_<date>.csv
      - data/processed/dressed_summary_<date>.json

    Exits non-zero if the player pool looks obviously wrong (thin teams, duplicated player IDs, etc).
    """
    console.rule("Check Dressed Players")
    try:
        res = build_and_check_dressed_players(
            date_str,
            min_dressed_per_team=int(min_dressed_per_team),
            min_total_roster_per_team=int(min_total_roster_per_team),
            fail_on_error=True,
        )
        console.print({
            "date": date_str,
            "ok": bool(res.ok),
            "dressed_players": str(res.dressed_players_path),
            "summary": str(res.summary_path),
            "issues": res.summary.get("issues"),
        })
    except Exception as e:
        console.print(f"Dressed-to-play check failed: {e}", style="red")
        raise SystemExit(2)


@cli.command("audit-rosters")
@click.option("--date", "date_str", type=str, required=True, help="Date YYYY-MM-DD to audit league_status vs boxscores")
def audit_rosters_cmd(date_str: str):
    """Audit roster team assignment accuracy by comparing league_status_<date>.csv vs boxscores_<date>.csv.

    Writes data/processed/roster_audit_<date>.csv and prints summary metrics.
    """
    console.rule("Audit Rosters")
    try:
        audit, summary = audit_roster_for_date(date_str)
        if audit is None or audit.empty:
            console.print({"date": date_str, "rows": 0, "summary": summary}, style="yellow")
            return
        out_path = paths.data_processed / f"roster_audit_{date_str}.csv"
        console.print({"date": date_str, "rows": int(len(audit)), "output": str(out_path), **summary})
    except Exception as e:
        console.print(f"Failed to audit rosters: {e}", style="red")


@cli.command("audit-playable-prop-settlement")
@click.option("--season", type=int, default=2026, show_default=True, help="Season to audit")
@click.option("--date", "date_str", type=str, required=False, help="Stop audit at date YYYY-MM-DD")
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), required=False, help="Optional output CSV path")
def audit_playable_prop_settlement_cmd(season: int, date_str: str | None, out_path: Path | None):
    """Export unresolved playable prop settlement gaps for season cards.

    Writes a CSV classifying each unresolved playable prop by source coverage and lookup failure mode.
    """
    console.rule("Audit Playable Prop Settlement")
    try:
        audit, summary = audit_playable_prop_settlement(season, requested_date=date_str)
        if audit is None or audit.empty:
            console.print(summary, style="yellow")
            return
        target = out_path if out_path is not None else default_playable_prop_settlement_audit_path(season=season, requested_date=date_str)
        target.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(target, index=False)
        rollups = build_playable_prop_settlement_rollups(audit)
        player_target = default_playable_prop_settlement_rollup_path("players", season=season, requested_date=date_str)
        date_target = default_playable_prop_settlement_rollup_path("dates", season=season, requested_date=date_str)
        sleeve_target = default_playable_prop_settlement_rollup_path("sleeves", season=season, requested_date=date_str)
        rollups["players"].to_csv(player_target, index=False)
        rollups["dates"].to_csv(date_target, index=False)
        rollups["sleeves"].to_csv(sleeve_target, index=False)
        top_players = []
        if not rollups["players"].empty:
            top_players = rollups["players"].head(10).to_dict(orient="records")
        console.print(
            {
                **summary,
                "output": str(target),
                "player_rollup": str(player_target),
                "date_rollup": str(date_target),
                "sleeve_rollup": str(sleeve_target),
                "top_players": top_players,
            }
        )
    except Exception as e:
        console.print(f"Failed to audit playable prop settlement gaps: {e}", style="red")


@cli.command("audit-playable-prop-aliases")
@click.option("--season", type=int, default=2026, show_default=True, help="Season to audit")
@click.option("--date", "date_str", type=str, required=False, help="Stop audit at date YYYY-MM-DD")
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), required=False, help="Optional output CSV path")
def audit_playable_prop_aliases_cmd(season: int, date_str: str | None, out_path: Path | None):
    """Export likely alias candidates for unresolved playable prop settlement misses.

    Replays lookup misses against the actuals feed and surfaces same-team name variants that may be fixable via canonical aliases.
    """
    console.rule("Audit Playable Prop Aliases")
    try:
        audit, summary = audit_playable_prop_aliases(season, requested_date=date_str)
        if audit is None or audit.empty:
            console.print(summary, style="yellow")
            return
        target = out_path if out_path is not None else default_playable_prop_alias_audit_path(season=season, requested_date=date_str)
        target.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(target, index=False)
        rollups = build_playable_prop_alias_rollups(audit)
        pair_target = default_playable_prop_alias_rollup_path("pairs", season=season, requested_date=date_str)
        player_target = default_playable_prop_alias_rollup_path("players", season=season, requested_date=date_str)
        rollups["pairs"].to_csv(pair_target, index=False)
        rollups["players"].to_csv(player_target, index=False)
        console.print(
            {
                **summary,
                "output": str(target),
                "pair_rollup": str(pair_target),
                "player_rollup": str(player_target),
            }
        )
    except Exception as e:
        console.print(f"Failed to audit playable prop aliases: {e}", style="red")


@cli.command("audit-playable-prop-provider-anomalies")
@click.option("--season", type=int, default=2026, show_default=True, help="Season to audit")
@click.option("--date", "date_str", type=str, required=False, help="Stop audit at date YYYY-MM-DD")
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), required=False, help="Optional output CSV path")
def audit_playable_prop_provider_anomalies_cmd(season: int, date_str: str | None, out_path: Path | None):
    """Export provider-side anomaly candidates for unresolved playable prop lookup misses."""
    console.rule("Audit Playable Prop Provider Anomalies")
    try:
        audit, summary = audit_playable_prop_provider_anomalies(season, requested_date=date_str)
        if audit is None or audit.empty:
            console.print(summary, style="yellow")
            return
        target = out_path if out_path is not None else default_playable_prop_provider_audit_path(season=season, requested_date=date_str)
        target.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(target, index=False)
        rollups = build_playable_prop_provider_rollups(audit)
        player_target = default_playable_prop_provider_rollup_path("players", season=season, requested_date=date_str)
        date_target = default_playable_prop_provider_rollup_path("dates", season=season, requested_date=date_str)
        pair_target = default_playable_prop_provider_rollup_path("pairs", season=season, requested_date=date_str)
        rollups["players"].to_csv(player_target, index=False)
        rollups["dates"].to_csv(date_target, index=False)
        rollups["pairs"].to_csv(pair_target, index=False)
        console.print(
            {
                **summary,
                "output": str(target),
                "player_rollup": str(player_target),
                "date_rollup": str(date_target),
                "pair_rollup": str(pair_target),
            }
        )
    except Exception as e:
        console.print(f"Failed to audit playable prop provider anomalies: {e}", style="red")


@cli.command("audit-playable-prop-coverage-gaps")
@click.option("--season", type=int, default=2026, show_default=True, help="Season to audit")
@click.option("--date", "date_str", type=str, required=False, help="Stop audit at date YYYY-MM-DD")
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), required=False, help="Optional output CSV path")
def audit_playable_prop_coverage_gaps_cmd(season: int, date_str: str | None, out_path: Path | None):
    """Export coverage-gap rows for unresolved playable prop misses after alias and provider checks."""
    console.rule("Audit Playable Prop Coverage Gaps")
    try:
        audit, summary = audit_playable_prop_coverage_gaps(season, requested_date=date_str)
        if audit is None or audit.empty:
            console.print(summary, style="yellow")
            return
        target = out_path if out_path is not None else default_playable_prop_coverage_audit_path(season=season, requested_date=date_str)
        target.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(target, index=False)
        rollups = build_playable_prop_coverage_rollups(audit)
        player_target = default_playable_prop_coverage_rollup_path("players", season=season, requested_date=date_str)
        date_target = default_playable_prop_coverage_rollup_path("dates", season=season, requested_date=date_str)
        team_target = default_playable_prop_coverage_rollup_path("teams", season=season, requested_date=date_str)
        rollups["players"].to_csv(player_target, index=False)
        rollups["dates"].to_csv(date_target, index=False)
        rollups["teams"].to_csv(team_target, index=False)
        console.print(
            {
                **summary,
                "output": str(target),
                "player_rollup": str(player_target),
                "date_rollup": str(date_target),
                "team_rollup": str(team_target),
            }
        )
    except Exception as e:
        console.print(f"Failed to audit playable prop coverage gaps: {e}", style="red")


@cli.command("backfill-player-props")
@click.option("--date", type=str, required=True, help="Snapshot date YYYY-MM-DD (UTC) for player props")
@click.option("--markets", type=str, required=False, help="Comma-separated OddsAPI player markets (default common set)")
@click.option("--mode", type=click.Choice(["auto","historical","current"]), default="auto", show_default=True, help="Fetch mode: historical uses snapshots by timestamp; current pulls current event odds for the day.")
@click.option("--api-key", envvar="ODDS_API_KEY", type=str, required=False, help="OddsAPI key (or set env ODDS_API_KEY)")
def backfill_player_props_cmd(date: str, markets: str | None, mode: str, api_key: str | None):
    """Fetch OddsAPI player props snapshot for a date and save to data/raw.

    Requires env var ODDS_API_KEY or configure via code.
    """
    console.rule("Backfill Player Props (OddsAPI)")
    if not api_key:
        # Fallback to .env file
        api_key = _load_dotenv_key("ODDS_API_KEY")
    if not api_key:
        console.print("Provide --api-key, set ODDS_API_KEY env, or add to .env at repo root.", style="red"); return
    cfg = OddsApiConfig(api_key=api_key)
    import datetime as _dt
    try:
        d = _dt.datetime.strptime(date, "%Y-%m-%d")
    except Exception:
        console.print("Invalid date; expected YYYY-MM-DD", style="red"); return
    mkts = [m.strip() for m in markets.split(',')] if markets else None
    # Modes: historical, current, or auto (try historical then current)
    from pathlib import Path as _P
    out_parq = paths.data_raw / "odds_nba_player_props.parquet"
    out_csv = paths.data_raw / "odds_nba_player_props.csv"
    def save(df):
        if df is None or df.empty:
            return 0
        df.to_csv(out_csv, index=False)
        try:
            df.to_parquet(out_parq, index=False)
        except Exception:
            pass
        return len(df)
    rows = 0
    if mode in ("historical","auto"):
        dfh = backfill_player_props(cfg, date=d, markets=mkts, verbose=True)
        rows += save(dfh)
    if rows == 0 and mode in ("current","auto"):
        # Use current event odds for the same calendar date
        dfc = fetch_player_props_current(cfg, date=d, markets=mkts, verbose=True)
        rows += save(dfc)
    console.print({"rows": int(rows)})


@cli.command("fetch-prop-actuals")
@click.option("--date", "date_str", type=str, required=False, help="Single date YYYY-MM-DD")
@click.option("--start", "start_str", type=str, required=False, help="Start date YYYY-MM-DD")
@click.option("--end", "end_str", type=str, required=False, help="End date YYYY-MM-DD")
def fetch_prop_actuals_cmd(date_str: str | None, start_str: str | None, end_str: str | None):
    """Fetch player prop actuals (PTS, REB, AST, STL, BLK, TOV, 3PM, PRA) via nbastatR and upsert to processed files.

    Requires R and the nbastatR package installed. On Windows, ensure Rscript.exe is on PATH.
    """
    console.rule("Fetch Prop Actuals (nbastatR)")
    # First attempt nbastatR via Rscript; if R not available, fall back to nba_api
    try:
        if date_str:
            df = fetch_prop_actuals_via_nbastatr(date=date_str)
        else:
            if not start_str or not end_str:
                console.print("Provide --date or both --start and --end", style="red"); return
            df = fetch_prop_actuals_via_nbastatr(start=start_str, end=end_str)
    except (FileNotFoundError, RuntimeError):
        # Fallbacks (single-date only): try NBA CDN first, then nba_api
        if not date_str:
            console.print("R not available and fallback supports only --date.", style="red"); return
        console.print(f"nbastatR unavailable, trying NBA liveData CDN for {date_str}...", style="yellow")
        try:
            df = fetch_prop_actuals_via_nba_cdn(date_str)
        except Exception as e_cdn:
            console.print(f"NBA CDN fallback failed: {e_cdn}; trying nba_api...", style="yellow")
            try:
                df = fetch_prop_actuals_via_nbaapi(date_str)
            except Exception as ee:
                console.print(f"nba_api fallback failed: {ee}", style="red"); return
    if df is None or df.empty:
        console.print("No rows returned.", style="yellow"); return
    out_path = upsert_props_actuals(df)
    console.print({"rows": int(len(df)), "output": str(out_path)})
    # Also write simple recon_props_{date}.csv for the frontend if single date
    if date_str:
        try:
            dd = pd.to_datetime(date_str).date()
            small = df.copy()
            small["date"] = pd.to_datetime(small["date"]).dt.date
            small = small[small["date"] == dd]
            if "team_abbr" not in small.columns and "team" in small.columns:
                small["team_abbr"] = small["team"]
            keep = [c for c in ["date","game_id","player_id","player_name","team_abbr","pts","reb","ast","threes","stl","blk","tov","pra"] if c in small.columns]
            out_csv = paths.data_processed / f"recon_props_{date_str}.csv"
            small[keep].to_csv(out_csv, index=False)
            console.print({"recon_props": str(out_csv), "rows": int(len(small))})
        except Exception:
            pass


@cli.command("build-props-features")
def build_props_features_cmd():
    """Build per-player props features (rolling windows) from player logs and save to processed folder."""
    console.rule("Build Props Features")
    try:
        df = build_props_features()
    except Exception as e:
        console.print(f"Failed to build props features: {e}", style="red"); return
    console.print({"rows": int(len(df))})


@cli.command("train-props")
@click.option("--alpha", type=float, default=1.0, show_default=True, help="Ridge regularization strength")
def train_props_cmd(alpha: float):
    """Train props regression models for PTS/REB/AST/3PM/PRA and save to models folder."""
    from .props_train import train_props_models  # Import here to avoid sklearn dependency
    console.rule("Train Props Models")
    try:
        train_props_models(alpha=alpha)
        console.print("Saved props models and feature columns.")
    except FileNotFoundError as e:
        console.print(str(e), style="red")
    except Exception as e:
        console.print(f"Failed to train props models: {e}", style="red")


@cli.command("playoff-retune")
@click.option("--date", "date_str", type=str, required=False, help="Reference date YYYY-MM-DD; defaults to today")
@click.option("--season", type=str, required=False, help="Season string like 2025-26; defaults from --date")
@click.option("--refresh-player-logs/--no-refresh-player-logs", default=False, show_default=True, help="Refresh current-season player logs before rebuilding props features")
@click.option("--retune-live-lens/--no-retune-live-lens", default=True, show_default=True, help="Retune game and prop live lens thresholds from the full regular season")
def playoff_retune_cmd(date_str: str | None, season: str | None, refresh_player_logs: bool, retune_live_lens: bool):
    """Retrain and retune games, props, and live lens from the completed regular season for playoff transition."""
    console.rule("Playoff Retune")
    import datetime as _dt
    from .playoff_transition import run_playoff_transition

    try:
        target_date = _dt.datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else _dt.date.today()
    except Exception:
        console.print("Invalid --date (expected YYYY-MM-DD)", style="red")
        return

    try:
        summary = run_playoff_transition(
            target_date=target_date,
            season=season,
            refresh_player_logs=bool(refresh_player_logs),
            retune_live_lens=bool(retune_live_lens),
        )
    except Exception as e:
        console.print(f"Playoff retune failed: {e}", style="red")
        return

    console.print({
        "season": summary.get("season"),
        "regular_season_start": summary.get("regular_season_start"),
        "regular_season_end": summary.get("regular_season_end"),
        "games_trained": summary.get("games_trained"),
        "props_feature_rows": summary.get("props_feature_rows"),
        "summary_path": summary.get("summary_path"),
    })


@cli.command("train-props-pure")
@click.option("--targets", type=str, default="t_stl,t_blk,t_tov", show_default=True, help="Comma-separated targets to train with pure linear fallback")
@click.option("--alpha", type=float, default=1.0, show_default=True, help="Ridge regularization strength")
def train_props_pure_cmd(targets: str, alpha: float):
    """Train pure linear (numpy ridge) fallback models for selected targets (no sklearn)."""
    console.rule("Train Props (Pure Linear)")
    try:
        tgt_list = [t.strip() for t in targets.split(',') if t.strip()]
        out = train_linear_props_models(targets=tgt_list, alpha=alpha)
        console.print({"saved": str(out), "targets": tgt_list})
    except Exception as e:
        console.print(f"Failed to train pure linear models: {e}", style="red")


@cli.command("export-props-onnx")
@click.option("--targets", type=str, default="t_stl,t_blk,t_tov", show_default=True, help="Comma-separated targets to export from pure linear models")
def export_props_onnx_cmd(targets: str):
    """Export pure-linear models (MatMul + Add) to ONNX for NPU path."""
    console.rule("Export Props ONNX (from pure-linear)")
    try:
        tgt_list = [t.strip() for t in targets.split(',') if t.strip()]
        out = export_linear_to_onnx(tgt_list)
        console.print({"exported": {k: str(v) for k, v in out.items()}})
    except Exception as e:
        console.print(f"Failed to export ONNX: {e}", style="red")


@cli.command("props-backtest")
@click.option("--targets", type=str, default="t_stl,t_blk,t_tov", show_default=True, help="Comma-separated targets to backtest (use t_* names)")
@click.option("--start", type=str, required=False, help="Start date YYYY-MM-DD (if available in features)")
@click.option("--end", type=str, required=False, help="End date YYYY-MM-DD (if available in features)")
def props_backtest_cmd(targets: str, start: str | None, end: str | None):
    """Quick backtest of pure-linear props models on historical features with basic metrics."""
    console.rule("Props Backtest (pure-linear)")
    tgt_list = [t.strip() for t in targets.split(',') if t.strip()]
    try:
        df = backtest_linear_props(tgt_list, start=start, end=end)
        rows = 0 if df is None else int(len(df))
        console.print({"rows": rows})
        if rows:
            console.print(df)
    except Exception as e:
        console.print(f"Failed to run backtest: {e}", style="red")


@cli.command("props-edges-backtest")
@click.option(
    "--preset",
    type=click.Choice(["default", "max-profit", "oos-threes", "oos-threes-excl-betmgm"], case_sensitive=False),
    default="default",
    show_default=True,
    help="Convenience preset for pick rules",
)
@click.option("--start", type=str, required=False, help="Start date YYYY-MM-DD (optional if --days is used)")
@click.option("--end", type=str, required=False, help="End date YYYY-MM-DD (optional; defaults to yesterday if --days used)")
@click.option("--days", type=int, default=None, help="Rolling window length (sets start/end). Example: --days 30")
@click.option("--sort-by", type=click.Choice(["ev", "edge"], case_sensitive=False), default="ev", show_default=True)
@click.option("--top-n-per-day", type=int, default=12, show_default=True, help="Take top N bets per day")
@click.option("--top-n-per-game", type=int, default=None, help="If set, take top N bets per game (by home_team/away_team)")
@click.option("--min-ev", type=float, default=None, help="Optional minimum EV filter")
@click.option("--min-edge", type=float, default=None, help="Optional minimum edge filter")
@click.option("--min-price", type=float, default=None, help="Optional minimum American odds (price) filter, e.g. -200")
@click.option("--max-price", type=float, default=None, help="Optional maximum American odds (price) filter, e.g. 150")
@click.option("--include-stats", type=str, default=None, help="Comma-separated stat allowlist (e.g., pts,reb,threes)")
@click.option("--exclude-stats", type=str, default=None, help="Comma-separated stat blocklist (e.g., ast)")
@click.option("--bookmaker", type=str, default=None, help="Filter to a single bookmaker id (e.g., bovada)")
@click.option("--exclude-bookmakers", type=str, default=None, help="Comma-separated bookmaker ids to exclude (e.g., betmgm,bovada)")
@click.option("--include-dd-td/--exclude-dd-td", default=False, show_default=True, help="Include DD/TD YES/NO markets in top picks")
@click.option("--no-dedupe", is_flag=True, default=False, help="Disable dedupe across books for identical bets")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), default=None, help="Optional output CSV for per-bet ledger")
@click.option("--out-daily", "out_daily_path", type=click.Path(dir_okay=False), default=None, help="Optional output CSV for per-day summary")
def props_edges_backtest_cmd(
    preset: str,
    start: str | None,
    end: str | None,
    days: int | None,
    sort_by: str,
    top_n_per_day: int,
    top_n_per_game: int | None,
    min_ev: float | None,
    min_edge: float | None,
    min_price: float | None,
    max_price: float | None,
    include_stats: str | None,
    exclude_stats: str | None,
    bookmaker: str | None,
    exclude_bookmakers: str | None,
    include_dd_td: bool,
    no_dedupe: bool,
    out_path: str | None,
    out_daily_path: str | None,
):
    """Backtest top props picks from props_edges_<date>.csv vs recon_props_<date>.csv.

    Grades OVER/UNDER outcomes using recon_props actuals for markets:
      pts, reb, ast, threes, pra, plus combos ra/pr/pa computed from pts/reb/ast.

    Output:
      - prints summary metrics to console
      - optional per-bet ledger CSV via --out
    """
    console.rule("Props Edges Backtest")

    # Apply presets (override / fill defaults)
    if str(preset).lower() == "max-profit":
        sort_by = "edge"
        if not include_stats:
            include_stats = "pts,reb,threes"

    # Walk-forward ROI tuned (recent): lower volume, better ROI, capped longshots
    if str(preset).lower() == "oos-threes":
        sort_by = "edge"
        top_n_per_day = 3
        if min_ev is None:
            min_ev = 0.5
        if min_edge is None:
            min_edge = 0.02
        if min_price is None:
            min_price = -200
        if max_price is None:
            max_price = 300
        if not include_stats:
            include_stats = "threes"

    # Same as oos-threes, but always excludes BetMGM unless overridden.
    if str(preset).lower() == "oos-threes-excl-betmgm":
        sort_by = "edge"
        top_n_per_day = 3
        if min_ev is None:
            min_ev = 0.5
        if min_edge is None:
            min_edge = 0.02
        if min_price is None:
            min_price = -200
        if max_price is None:
            max_price = 300
        if not include_stats:
            include_stats = "threes"
        if not exclude_bookmakers:
            exclude_bookmakers = "betmgm"

    # Resolve date range
    try:
        if days is not None:
            if days <= 0:
                raise ValueError("--days must be > 0")
            # Default to last completed day unless an explicit --end is provided.
            end_d = (pd.Timestamp(_date.today()) - pd.Timedelta(days=1)).date().isoformat() if not end else str(end)
            start_d = (pd.to_datetime(end_d) - pd.Timedelta(days=int(days) - 1)).date().isoformat()
        else:
            if not start or not end:
                console.print("Provide --start and --end, or use --days.", style="red")
                raise SystemExit(2)
            start_d = str(start)
            end_d = str(end)
    except SystemExit:
        raise
    except Exception as e:
        console.print(f"Invalid date range args: {e}", style="red")
        raise SystemExit(2)

    cfg = PropsEdgesBacktestConfig(
        sort_by=str(sort_by).lower(),
        top_n_per_day=int(top_n_per_day),
        top_n_per_game=(int(top_n_per_game) if top_n_per_game is not None else None),
        min_ev=(float(min_ev) if min_ev is not None else None),
        min_edge=(float(min_edge) if min_edge is not None else None),
        min_price=(float(min_price) if min_price is not None else None),
        max_price=(float(max_price) if max_price is not None else None),
        include_stats=(tuple([p.strip().lower() for p in include_stats.split(",") if p.strip()]) if include_stats else None),
        exclude_stats=(tuple([p.strip().lower() for p in exclude_stats.split(",") if p.strip()]) if exclude_stats else None),
        bookmaker=(str(bookmaker) if bookmaker else None),
        exclude_bookmakers=(
            tuple([p.strip().lower() for p in exclude_bookmakers.split(",") if p.strip()])
            if exclude_bookmakers
            else None
        ),
        dedupe_best_book=(not bool(no_dedupe)),
        include_dd_td=bool(include_dd_td),
    )
    try:
        ledger, summary, daily = backtest_props_edges(start=start_d, end=end_d, cfg=cfg)
        if summary is None or summary.empty:
            # Distinguish between missing files vs. zero picks after filters.
            try:
                days = [d.date().isoformat() for d in pd.date_range(start=start_d, end=end_d, freq="D")]
                proc = paths.data_processed
                has_both = 0
                for d in days:
                    if (proc / f"props_edges_{d}.csv").exists() and (proc / f"recon_props_{d}.csv").exists():
                        has_both += 1
                note = "No matching days (missing props_edges or recon_props)."
                if has_both:
                    note = "No graded bets selected (files exist, but filters produced zero picks)."
                console.print({"rows": 0, "note": note}, style="yellow")
            except Exception:
                console.print({"rows": 0, "note": "No matching days (missing props_edges or recon_props)."}, style="yellow")
            return

        console.print(summary)

        if daily is not None and not daily.empty:
            console.rule("Daily")
            console.print(daily)

        # Extra breakdowns (quick sanity)
        try:
            if ledger is not None and not ledger.empty and "result" in ledger.columns:
                df = ledger.copy()
                df["graded"] = df["result"].notna()
                df["win"] = df["result"] == "W"
                df["loss"] = df["result"] == "L"
                df["push"] = df["result"] == "P"
                df["profit"] = pd.to_numeric(df.get("profit"), errors="coerce")

                def agg(keys: list[str]):
                    g = df.groupby(keys, dropna=False)
                    out = g.agg(
                        bets_total=("result", "size"),
                        bets_graded=("graded", "sum"),
                        wins=("win", "sum"),
                        losses=("loss", "sum"),
                        pushes=("push", "sum"),
                        profit=("profit", "sum"),
                    ).reset_index()
                    out["hit_rate"] = out.apply(
                        lambda r: (r.wins / (r.wins + r.losses)) if (r.wins + r.losses) else np.nan,
                        axis=1,
                    )
                    out["roi_per_bet"] = out.apply(
                        lambda r: (r.profit / r.bets_graded) if r.bets_graded else np.nan,
                        axis=1,
                    )
                    return out.sort_values(by=["roi_per_bet"], ascending=False, na_position="last")

                if "stat" in df.columns:
                    console.rule("By Stat")
                    console.print(agg(["stat"]))
                if "bookmaker" in df.columns:
                    console.rule("By Bookmaker")
                    console.print(agg(["bookmaker"]))
        except Exception:
            pass

        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(out_path, index=False)
            console.print(f"Wrote ledger: {out_path}", style="green")
        else:
            # Default: write into processed for convenience
            try:
                out = paths.data_processed / f"props_edges_backtest_{start_d}_to_{end_d}.csv"
                ledger.to_csv(out, index=False)
                console.print(f"Wrote ledger: {out}", style="green")
            except Exception:
                pass

        try:
            if daily is not None and not daily.empty:
                if out_daily_path:
                    Path(out_daily_path).parent.mkdir(parents=True, exist_ok=True)
                    daily.to_csv(out_daily_path, index=False)
                    console.print(f"Wrote daily: {out_daily_path}", style="green")
                else:
                    outd = paths.data_processed / f"props_edges_backtest_daily_{start_d}_to_{end_d}.csv"
                    daily.to_csv(outd, index=False)
                    console.print(f"Wrote daily: {outd}", style="green")
        except Exception:
            pass
    except Exception as e:
        console.print(f"Failed props-edges-backtest: {e}", style="red")


@cli.command("props-edges-optimize")
@click.option("--start", type=str, required=False, help="Start date YYYY-MM-DD (optional if --days is used)")
@click.option("--end", type=str, required=False, help="End date YYYY-MM-DD (optional; defaults to yesterday if --days used)")
@click.option("--days", type=int, default=30, show_default=True, help="If start/end not provided, optimize over last N completed days")
@click.option("--objective", type=click.Choice(["profit", "roi"], case_sensitive=False), default="profit", show_default=True)
@click.option("--min-bets", type=int, default=40, show_default=True, help="Minimum graded bets required to consider a config")
@click.option("--top-n", "top_n_list", type=str, default="3,5,8,12", show_default=True, help="Comma list for top_n_per_day")
@click.option("--min-ev", "min_ev_list", type=str, default="none,0.5,1.0", show_default=True, help="Comma list for min_ev; use 'none'")
@click.option("--min-edge", "min_edge_list", type=str, default="none,0.02,0.04", show_default=True, help="Comma list for min_edge; use 'none'")
@click.option("--sort-by", "sort_by_list", type=str, default="ev,edge", show_default=True, help="Comma list for sort key")
@click.option(
    "--include-stats",
    "include_stats_sets",
    multiple=True,
    help="Comma list allowlist of stats to include; may be repeated. Use 'all' for no filter.",
)
@click.option(
    "--exclude-stats",
    "exclude_stats_sets",
    multiple=True,
    help="Comma list blocklist of stats to exclude; may be repeated.",
)
@click.option("--try-dd-td/--no-try-dd-td", default=False, show_default=True, help="Include both include_dd_td=False and True in grid")
@click.option("--bookmakers", type=str, default=None, help="Comma list of bookmaker ids to test individually (plus all)")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), default=None, help="Optional output CSV for optimization results")
def props_edges_optimize_cmd(
    start: str | None,
    end: str | None,
    days: int,
    objective: str,
    min_bets: int,
    top_n_list: str,
    min_ev_list: str,
    min_edge_list: str,
    sort_by_list: str,
    include_stats_sets: tuple[str, ...],
    exclude_stats_sets: tuple[str, ...],
    try_dd_td: bool,
    bookmakers: str | None,
    out_path: str | None,
):
    """Grid-search pick-selection parameters to maximize profit/ROI on historical props_edges."""
    console.rule("Props Edges Optimize")

    # Resolve date range (through last completed day)
    try:
        if start and end:
            start_d = str(start)
            end_d = str(end)
        else:
            if days <= 0:
                raise ValueError("--days must be > 0")
            end_d = (pd.Timestamp(_date.today()) - pd.Timedelta(days=1)).date().isoformat() if not end else str(end)
            start_d = (pd.to_datetime(end_d) - pd.Timedelta(days=int(days) - 1)).date().isoformat() if not start else str(start)
    except Exception as e:
        console.print(f"Invalid date range args: {e}", style="red")
        raise SystemExit(2)

    def _parse_num_list(s: str) -> list[float | None]:
        out: list[float | None] = []
        for part in (s or "").split(","):
            t = part.strip().lower()
            if not t or t in {"none", "null", "na"}:
                out.append(None)
                continue
            out.append(float(t))
        return out

    def _parse_int_list(s: str) -> list[int]:
        out: list[int] = []
        for part in (s or "").split(","):
            t = part.strip()
            if not t:
                continue
            out.append(int(t))
        return out

    def _parse_sets(raw: tuple[str, ...], default_sets: list[str]) -> list[tuple[str, ...] | None]:
        items = list(raw) if raw else default_sets
        sets: list[tuple[str, ...] | None] = []
        for spec in items:
            spec = (spec or "").strip()
            if not spec or spec.lower() == "all":
                sets.append(None)
                continue
            parts = [p.strip().lower() for p in spec.split(",") if p.strip()]
            sets.append(tuple(parts) if parts else None)
        # Dedupe while preserving order
        seen: set[tuple[str, ...] | None] = set()
        uniq: list[tuple[str, ...] | None] = []
        for x in sets:
            if x in seen:
                continue
            seen.add(x)
            uniq.append(x)
        return uniq

    top_ns = _parse_int_list(top_n_list)
    min_evs = _parse_num_list(min_ev_list)
    min_edges = _parse_num_list(min_edge_list)
    sort_bys = [s.strip().lower() for s in (sort_by_list or "").split(",") if s.strip()]
    if not sort_bys:
        sort_bys = ["ev"]

    include_sets = _parse_sets(
        include_stats_sets,
        default_sets=["all", "threes", "pts,threes", "pts,reb,threes", "reb,threes"],
    )
    exclude_sets = _parse_sets(exclude_stats_sets, default_sets=["all"])

    bm_list: list[str | None] = [None]
    if bookmakers:
        bm_list = [None] + [b.strip().lower() for b in bookmakers.split(",") if b.strip()]

    dd_td_vals = [False, True] if try_dd_td else [False]

    rows: list[dict[str, object]] = []
    combos = list(itertools.product(top_ns, min_evs, min_edges, sort_bys, include_sets, exclude_sets, bm_list, dd_td_vals))
    console.print({"start": start_d, "end": end_d, "combos": int(len(combos)), "min_bets": int(min_bets), "objective": str(objective).lower()})

    for top_n, min_ev, min_edge, sort_by, include_stats, exclude_stats, bm, include_dd_td in combos:
        cfg = PropsEdgesBacktestConfig(
            sort_by=str(sort_by),
            top_n_per_day=int(top_n),
            top_n_per_game=None,
            min_ev=min_ev,
            min_edge=min_edge,
            bookmaker=bm,
            dedupe_best_book=True,
            include_dd_td=bool(include_dd_td),
            include_stats=include_stats,
            exclude_stats=exclude_stats,
        )
        try:
            _, summary, _ = backtest_props_edges(start=start_d, end=end_d, cfg=cfg)
        except Exception:
            continue
        if summary is None or summary.empty:
            continue
        r = summary.iloc[0].to_dict()
        if int(r.get("bets_graded") or 0) < int(min_bets):
            continue
        rows.append(r)

    res = pd.DataFrame(rows)
    if res.empty:
        console.print({"rows": 0, "note": "No configs met min-bets / missing files."}, style="yellow")
        return

    # De-dupe identical configs (can occur if different grid specs normalize to same config)
    cfg_cols = [
        c
        for c in [
            "sort_by",
            "top_n_per_day",
            "top_n_per_game",
            "min_ev",
            "min_edge",
            "bookmaker",
            "dedupe_best_book",
            "include_dd_td",
            "include_stats",
            "exclude_stats",
        ]
        if c in res.columns
    ]
    if cfg_cols:
        tmp = res[cfg_cols].copy()
        tmp = tmp.where(tmp.notna(), "<NA>")
        keep_idx = tmp.drop_duplicates(subset=cfg_cols, keep="first").index
        res = res.loc[keep_idx].reset_index(drop=True)

    obj = str(objective).lower()
    score_col = "profit" if obj == "profit" else "roi_per_bet"
    res = res.sort_values(by=[score_col], ascending=False, na_position="last").reset_index(drop=True)
    console.rule(f"Top Configs by {score_col}")
    show_cols = [
        c
        for c in [
            "profit",
            "roi_per_bet",
            "hit_rate",
            "bets_graded",
            "sort_by",
            "top_n_per_day",
            "min_ev",
            "min_edge",
            "include_stats",
            "exclude_stats",
            "bookmaker",
            "include_dd_td",
        ]
        if c in res.columns
    ]
    console.print(res[show_cols].head(25) if show_cols else res.head(25))

    # Print a runnable command for the best config
    try:
        best = res.iloc[0].to_dict()
        cmd = [
            ".\\.venv\\Scripts\\python.exe",
            "-m",
            "nba_betting.cli",
            "props-edges-backtest",
            "--start",
            str(start_d),
            "--end",
            str(end_d),
            "--sort-by",
            str(best.get("sort_by") or "ev"),
            "--top-n-per-day",
            str(int(best.get("top_n_per_day") or 12)),
        ]
        if best.get("min_ev") is not None and str(best.get("min_ev")) != "nan":
            cmd += ["--min-ev", str(best.get("min_ev"))]
        if best.get("min_edge") is not None and str(best.get("min_edge")) != "nan":
            cmd += ["--min-edge", str(best.get("min_edge"))]
        if best.get("include_stats"):
            cmd += ["--include-stats", str(best.get("include_stats"))]
        if best.get("exclude_stats"):
            cmd += ["--exclude-stats", str(best.get("exclude_stats"))]
        if best.get("bookmaker"):
            cmd += ["--bookmaker", str(best.get("bookmaker"))]
        if bool(best.get("include_dd_td")):
            cmd += ["--include-dd-td"]
        console.rule("Best Config Command")
        console.print(" ".join(cmd))
    except Exception:
        pass

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(out_path, index=False)
        console.print(f"Wrote optimize results: {out_path}", style="green")
    else:
        try:
            out = paths.data_processed / f"props_edges_optimize_{start_d}_to_{end_d}.csv"
            res.to_csv(out, index=False)
            console.print(f"Wrote optimize results: {out}", style="green")
        except Exception:
            pass


@cli.command("props-edges-walkforward")
@click.option(
    "--preset",
    type=click.Choice(["default", "oos-threes", "oos-threes-excl-betmgm", "oos-threes-excl-betmgm-tune", "oos-threes-excl-betmgm-tune-profit"], case_sensitive=False),
    default="default",
    show_default=True,
    help="Convenience preset for walk-forward config search",
)
@click.option("--start", type=str, required=False, help="Test start date YYYY-MM-DD (optional if --days is used)")
@click.option("--end", type=str, required=False, help="Test end date YYYY-MM-DD (optional; defaults to yesterday if --days used)")
@click.option("--days", type=int, default=14, show_default=True, help="If start/end not provided, test over last N completed days")
@click.option("--train-days", type=int, default=30, show_default=True, help="Training window length (days before each test day)")
@click.option("--objective", type=click.Choice(["profit", "roi"], case_sensitive=False), default="profit", show_default=True)
@click.option("--min-bets-train", type=int, default=80, show_default=True, help="Minimum graded bets required in training window")
@click.option("--recent-days", type=int, default=0, show_default=True, help="If >0, blend in score from the most recent N training days")
@click.option("--recent-weight", type=float, default=0.0, show_default=True, help="Weight (0..1) for recent-days score when selecting config")
@click.option("--min-bets-recent", type=int, default=20, show_default=True, help="Minimum graded bets required in the recent slice to use recent score")
@click.option("--top-n", "top_n_list", type=str, default="3,5,8,12", show_default=True, help="Comma list for top_n_per_day")
@click.option("--min-ev", "min_ev_list", type=str, default="none,0.5,1.0", show_default=True, help="Comma list for min_ev; use 'none'")
@click.option("--min-edge", "min_edge_list", type=str, default="none,0.02,0.04", show_default=True, help="Comma list for min_edge; use 'none'")
@click.option(
    "--price-ranges",
    type=str,
    default="none,-200:150,-180:140,-160:130,-140:120,-120:120",
    show_default=True,
    help="Comma list of American odds ranges to test. Use 'none' for no filter. Format: MIN:MAX (e.g., -200:150)",
)
@click.option("--sort-by", "sort_by_list", type=str, default="ev,edge", show_default=True, help="Comma list for sort key")
@click.option(
    "--include-stats",
    "include_stats_sets",
    multiple=True,
    help="Comma list allowlist of stats to include; may be repeated. Use 'all' for no filter.",
)
@click.option(
    "--exclude-stats",
    "exclude_stats_sets",
    multiple=True,
    help="Comma list blocklist of stats to exclude; may be repeated.",
)
@click.option("--try-dd-td/--no-try-dd-td", default=False, show_default=True, help="Include both include_dd_td=False and True in grid")
@click.option("--bookmakers", type=str, default=None, help="Comma list of bookmaker ids to test individually (plus all)")
@click.option(
    "--exclude-bookmakers-fixed",
    type=str,
    default=None,
    help="Comma list of bookmaker ids to always exclude (applied to all configs)",
)
@click.option(
    "--exclude-bookmakers",
    "exclude_bookmakers_sets",
    multiple=True,
    help="Comma list of bookmaker ids to exclude; may be repeated. Use 'none' for no filter.",
)
@click.option("--out", "out_path", type=click.Path(dir_okay=False), default=None, help="Optional output CSV for per-day walk-forward results")
def props_edges_walkforward_cmd(
    preset: str,
    start: str | None,
    end: str | None,
    days: int,
    train_days: int,
    objective: str,
    min_bets_train: int,
    recent_days: int,
    recent_weight: float,
    min_bets_recent: int,
    top_n_list: str,
    min_ev_list: str,
    min_edge_list: str,
    price_ranges: str,
    sort_by_list: str,
    include_stats_sets: tuple[str, ...],
    exclude_stats_sets: tuple[str, ...],
    try_dd_td: bool,
    bookmakers: str | None,
    exclude_bookmakers_fixed: str | None,
    exclude_bookmakers_sets: tuple[str, ...],
    out_path: str | None,
):
    """Walk-forward evaluation: pick best config on trailing window, apply to next day."""
    console.rule("Props Edges Walk-Forward")

    # Apply presets (override / fill defaults)
    if str(preset).lower() == "oos-threes":
        objective = "roi"
        top_n_list = "3"
        min_ev_list = "none,0.5"
        min_edge_list = "none,0.02"
        price_ranges = "-200:300"
        sort_by_list = "edge"
        include_stats_sets = ("threes",)
        # Default exclusion comparison: none vs betmgm (unless user specified)
        if not exclude_bookmakers_sets:
            exclude_bookmakers_sets = ("none", "betmgm")
        # defaults for recency: dynamic weight
        if recent_days == 0:
            recent_days = 7
        if recent_weight == 0.0:
            recent_weight = 0.7
        if min_bets_recent == 20:
            # keep default unless user changed
            min_bets_recent = 20

    if str(preset).lower() == "oos-threes-excl-betmgm":
        objective = "roi"
        top_n_list = "3"
        min_ev_list = "none,0.5"
        min_edge_list = "none,0.02"
        price_ranges = "-200:300"
        sort_by_list = "edge"
        include_stats_sets = ("threes",)
        if not exclude_bookmakers_fixed:
            exclude_bookmakers_fixed = "betmgm"
        # With fixed exclusion, don't add extra exclusion sets unless user provided them.
        if not exclude_bookmakers_sets:
            exclude_bookmakers_sets = ("none",)
        if recent_days == 0:
            recent_days = 7
        if recent_weight == 0.0:
            recent_weight = 0.7

    # Tune only threshold/volume knobs for the fixed BetMGM-excluded threes strategy.
    # This keeps the search space constrained and avoids overfitting across many dimensions.
    if str(preset).lower() == "oos-threes-excl-betmgm-tune":
        objective = "roi"
        # modest grid: volume vs quality
        top_n_list = "1,2,3,4"
        min_ev_list = "none,0.25,0.5"
        min_edge_list = "none,0.01,0.02"
        price_ranges = "-200:300"
        sort_by_list = "edge"
        include_stats_sets = ("threes",)
        if not exclude_bookmakers_fixed:
            exclude_bookmakers_fixed = "betmgm"
        if not exclude_bookmakers_sets:
            exclude_bookmakers_sets = ("none",)
        if recent_days == 0:
            recent_days = 7
        if recent_weight == 0.0:
            recent_weight = 0.7

    # Same as the tune preset, but selects configs by profit instead of ROI.
    if str(preset).lower() == "oos-threes-excl-betmgm-tune-profit":
        objective = "profit"
        top_n_list = "1,2,3,4"
        min_ev_list = "none,0.25,0.5"
        min_edge_list = "none,0.01,0.02"
        price_ranges = "-200:300"
        sort_by_list = "edge"
        include_stats_sets = ("threes",)
        if not exclude_bookmakers_fixed:
            exclude_bookmakers_fixed = "betmgm"
        if not exclude_bookmakers_sets:
            exclude_bookmakers_sets = ("none",)
        if recent_days == 0:
            recent_days = 7
        if recent_weight == 0.0:
            recent_weight = 0.7

    if train_days <= 0:
        console.print("--train-days must be > 0", style="red")
        raise SystemExit(2)

    if recent_days < 0:
        console.print("--recent-days must be >= 0", style="red")
        raise SystemExit(2)
    if recent_weight < 0.0 or recent_weight > 1.0:
        console.print("--recent-weight must be within [0, 1]", style="red")
        raise SystemExit(2)
    if min_bets_recent < 0:
        console.print("--min-bets-recent must be >= 0", style="red")
        raise SystemExit(2)

    # Resolve test range (through last completed day)
    try:
        if start and end:
            test_start = str(start)
            test_end = str(end)
        else:
            if days <= 0:
                raise ValueError("--days must be > 0")
            test_end = (pd.Timestamp(_date.today()) - pd.Timedelta(days=1)).date().isoformat() if not end else str(end)
            test_start = (pd.to_datetime(test_end) - pd.Timedelta(days=int(days) - 1)).date().isoformat() if not start else str(start)
    except Exception as e:
        console.print(f"Invalid date args: {e}", style="red")
        raise SystemExit(2)

    def _parse_num_list(s: str) -> list[float | None]:
        out: list[float | None] = []
        for part in (s or "").split(","):
            t = part.strip().lower()
            if not t or t in {"none", "null", "na"}:
                out.append(None)
                continue
            out.append(float(t))
        return out

    def _parse_int_list(s: str) -> list[int]:
        out: list[int] = []
        for part in (s or "").split(","):
            t = part.strip()
            if not t:
                continue
            out.append(int(t))
        return out

    def _parse_sets(raw: tuple[str, ...], default_sets: list[str]) -> list[tuple[str, ...] | None]:
        items = list(raw) if raw else default_sets
        sets: list[tuple[str, ...] | None] = []
        for spec in items:
            spec = (spec or "").strip()
            if not spec or spec.lower() == "all":
                sets.append(None)
                continue
            parts = [p.strip().lower() for p in spec.split(",") if p.strip()]
            sets.append(tuple(parts) if parts else None)
        seen: set[tuple[str, ...] | None] = set()
        uniq: list[tuple[str, ...] | None] = []
        for x in sets:
            if x in seen:
                continue
            seen.add(x)
            uniq.append(x)
        return uniq

    top_ns = _parse_int_list(top_n_list)
    min_evs = _parse_num_list(min_ev_list)
    min_edges = _parse_num_list(min_edge_list)

    def _parse_price_ranges(s: str) -> list[tuple[float | None, float | None]]:
        out: list[tuple[float | None, float | None]] = []
        for part in (s or "").split(","):
            t = part.strip().lower()
            if not t or t in {"none", "null", "na", "all"}:
                out.append((None, None))
                continue
            if ":" not in t:
                raise ValueError(f"Invalid --price-ranges entry '{part}'. Expected MIN:MAX or 'none'.")
            left, right = t.split(":", 1)
            left = left.strip()
            right = right.strip()
            mn = float(left) if left and left not in {"none", "null", "na"} else None
            mx = float(right) if right and right not in {"none", "null", "na"} else None
            out.append((mn, mx))
        # de-dupe preserving order
        seen: set[tuple[float | None, float | None]] = set()
        uniq: list[tuple[float | None, float | None]] = []
        for r in out:
            if r in seen:
                continue
            seen.add(r)
            uniq.append(r)
        return uniq

    price_range_list = _parse_price_ranges(price_ranges)
    sort_bys = [s.strip().lower() for s in (sort_by_list or "").split(",") if s.strip()]
    if not sort_bys:
        sort_bys = ["ev"]

    include_sets = _parse_sets(
        include_stats_sets,
        default_sets=["all", "threes", "pts,threes", "pts,reb,threes", "reb,threes"],
    )
    exclude_sets = _parse_sets(exclude_stats_sets, default_sets=["all"])

    def _parse_bookmaker_sets(raw: tuple[str, ...], default_sets: list[str]) -> list[tuple[str, ...] | None]:
        items = list(raw) if raw else default_sets
        sets: list[tuple[str, ...] | None] = []
        for spec in items:
            spec = (spec or "").strip()
            if not spec or spec.lower() in {"none", "all"}:
                sets.append(None)
                continue
            parts = [p.strip().lower() for p in spec.split(",") if p.strip()]
            sets.append(tuple(parts) if parts else None)
        seen: set[tuple[str, ...] | None] = set()
        uniq: list[tuple[str, ...] | None] = []
        for x in sets:
            if x in seen:
                continue
            seen.add(x)
            uniq.append(x)
        return uniq

    exclude_bm_sets = _parse_bookmaker_sets(exclude_bookmakers_sets, default_sets=["none", "betmgm", "bovada", "betmgm,bovada"])

    fixed_exclude_bms: tuple[str, ...] | None = None
    if exclude_bookmakers_fixed:
        parts = [p.strip().lower() for p in str(exclude_bookmakers_fixed).split(",") if p.strip()]
        fixed_exclude_bms = tuple(parts) if parts else None

    bm_list: list[str | None] = [None]
    if bookmakers:
        bm_list = [None] + [b.strip().lower() for b in bookmakers.split(",") if b.strip()]
    dd_td_vals = [False, True] if try_dd_td else [False]

    grid = list(itertools.product(top_ns, min_evs, min_edges, price_range_list, sort_bys, include_sets, exclude_sets, bm_list, exclude_bm_sets, dd_td_vals))
    console.print({
        "test_start": test_start,
        "test_end": test_end,
        "train_days": int(train_days),
        "grid": int(len(grid)),
        "objective": str(objective).lower(),
        "recent_days": int(recent_days),
        "recent_weight": float(recent_weight),
        "min_bets_recent": int(min_bets_recent),
        "exclude_bookmakers_fixed": (",".join(fixed_exclude_bms) if fixed_exclude_bms else None),
    })

    test_dates = [d.date().isoformat() for d in pd.date_range(start=test_start, end=test_end, freq="D")]

    daily_rows: list[dict[str, object]] = []
    total_profit = 0.0
    total_w = total_l = total_p = total_graded = 0

    obj = str(objective).lower()
    score_col = "profit" if obj == "profit" else "roi_per_bet"

    def _recent_slice_score(train_daily: pd.DataFrame | None) -> tuple[float | None, int | None]:
        """Compute recent-slice score and recent graded bets.

        Note: this function returns a score even when recent bet volume is small; the
        blend weight is scaled separately based on --min-bets-recent.
        """
        if train_daily is None or getattr(train_daily, "empty", True):
            return None, None
        if int(recent_days) <= 0 or float(recent_weight) <= 0.0:
            return None, None
        try:
            d2 = train_daily.copy()
            if "date" in d2.columns:
                d2["date"] = pd.to_datetime(d2["date"], errors="coerce")
                d2 = d2.sort_values("date")
            d2 = d2.tail(int(recent_days))
            if d2.empty:
                return None, None
            bets = int(pd.to_numeric(d2.get("bets_graded"), errors="coerce").fillna(0).sum())
            prof = float(pd.to_numeric(d2.get("profit"), errors="coerce").fillna(0.0).sum())
            if obj == "profit":
                # Scale recent profit up to a train_days-equivalent window.
                return (prof * (float(train_days) / float(recent_days))) if int(recent_days) > 0 else prof, bets
            return (prof / float(bets)) if bets > 0 else None, bets
        except Exception:
            return None, None

    def _blend(full_score: float, recent_score: float | None, recent_bets: int | None) -> float:
        if recent_score is None or float(recent_weight) <= 0.0:
            return float(full_score)
        w = float(recent_weight)
        try:
            if int(min_bets_recent) > 0 and recent_bets is not None:
                w = w * min(1.0, max(0.0, float(recent_bets) / float(min_bets_recent)))
        except Exception:
            pass
        if w <= 0.0:
            return float(full_score)
        return (1.0 - w) * float(full_score) + w * float(recent_score)

    for ds in track(test_dates, description="Walk-forward days"):
        train_end = (pd.to_datetime(ds) - pd.Timedelta(days=1)).date().isoformat()
        train_start = (pd.to_datetime(train_end) - pd.Timedelta(days=int(train_days) - 1)).date().isoformat()

        best_cfg: PropsEdgesBacktestConfig | None = None
        best_score = None
        best_train_summary: dict[str, object] | None = None
        best_train_score_all = None
        best_train_score_recent = None
        best_train_score_final = None
        best_train_recent_bets = None

        for top_n, min_ev, min_edge, pr, sort_by, include_stats, exclude_stats, bm, exclude_bms, include_dd_td in grid:
            mn_price, mx_price = pr

            merged_excludes: tuple[str, ...] | None = None
            if fixed_exclude_bms and exclude_bms:
                merged_excludes = tuple(dict.fromkeys([*fixed_exclude_bms, *exclude_bms]))
            elif fixed_exclude_bms:
                merged_excludes = fixed_exclude_bms
            elif exclude_bms:
                merged_excludes = exclude_bms

            cfg = PropsEdgesBacktestConfig(
                sort_by=str(sort_by),
                top_n_per_day=int(top_n),
                top_n_per_game=None,
                min_ev=min_ev,
                min_edge=min_edge,
                min_price=mn_price,
                max_price=mx_price,
                bookmaker=bm,
                exclude_bookmakers=merged_excludes,
                dedupe_best_book=True,
                include_dd_td=bool(include_dd_td),
                include_stats=include_stats,
                exclude_stats=exclude_stats,
            )
            try:
                _, s_train, d_train = backtest_props_edges(start=train_start, end=train_end, cfg=cfg)
            except Exception:
                continue
            if s_train is None or s_train.empty:
                continue
            tr = s_train.iloc[0].to_dict()
            if int(tr.get("bets_graded") or 0) < int(min_bets_train):
                continue
            sc = tr.get(score_col)
            try:
                sc_all = float(sc)
            except Exception:
                continue
            if not np.isfinite(sc_all):
                continue

            sc_recent, recent_bets = _recent_slice_score(d_train)
            sc_final = _blend(sc_all, sc_recent, recent_bets)
            scv = float(sc_final)
            if not np.isfinite(scv):
                continue
            if best_score is None or scv > float(best_score):
                best_score = scv
                best_cfg = cfg
                best_train_summary = tr
                best_train_score_all = sc_all
                best_train_score_recent = sc_recent
                best_train_score_final = sc_final
                best_train_recent_bets = recent_bets

        if best_cfg is None:
            daily_rows.append({"date": ds, "note": "no_valid_config"})
            continue

        # Apply to the test day
        try:
            ledger, s_test, _ = backtest_props_edges(start=ds, end=ds, cfg=best_cfg)
        except Exception:
            daily_rows.append({"date": ds, "note": "test_failed"})
            continue

        if s_test is None or s_test.empty:
            # Files might exist but filters produced zero picks.
            try:
                proc = paths.data_processed
                has_edges = (proc / f"props_edges_{ds}.csv").exists()
                has_recon = (proc / f"recon_props_{ds}.csv").exists()
                if has_edges and has_recon:
                    daily_rows.append({"date": ds, "note": "no_picks_after_filters"})
                elif (not has_edges) and (not has_recon):
                    daily_rows.append({"date": ds, "note": "missing_edges_and_recon"})
                elif not has_edges:
                    daily_rows.append({"date": ds, "note": "missing_edges"})
                else:
                    daily_rows.append({"date": ds, "note": "missing_recon"})
            except Exception:
                daily_rows.append({"date": ds, "note": "no_test_data"})
            continue

        te = s_test.iloc[0].to_dict()
        profit = float(te.get("profit") or 0.0) if str(te.get("profit")) != "nan" else 0.0
        bets_graded = int(te.get("bets_graded") or 0)
        wins = int(te.get("wins") or 0)
        losses = int(te.get("losses") or 0)
        pushes = int(te.get("pushes") or 0)

        total_profit += profit
        total_graded += bets_graded
        total_w += wins
        total_l += losses
        total_p += pushes

        daily_rows.append(
            {
                "date": ds,
                "train_start": train_start,
                "train_end": train_end,
                "train_score": best_score,
                "train_score_all": best_train_score_all,
                "train_score_recent": best_train_score_recent,
                "train_score_final": best_train_score_final,
                "train_recent_bets": best_train_recent_bets,
                "train_bets_graded": (best_train_summary.get("bets_graded") if best_train_summary else None),
                "cfg_sort_by": best_cfg.sort_by,
                "cfg_top_n_per_day": best_cfg.top_n_per_day,
                "cfg_min_ev": best_cfg.min_ev,
                "cfg_min_edge": best_cfg.min_edge,
                "cfg_min_price": best_cfg.min_price,
                "cfg_max_price": best_cfg.max_price,
                "cfg_include_stats": (",".join(best_cfg.include_stats) if best_cfg.include_stats else None),
                "cfg_exclude_stats": (",".join(best_cfg.exclude_stats) if best_cfg.exclude_stats else None),
                "cfg_bookmaker": best_cfg.bookmaker,
                "cfg_exclude_bookmakers": (",".join(best_cfg.exclude_bookmakers) if getattr(best_cfg, "exclude_bookmakers", None) else None),
                "cfg_include_dd_td": bool(best_cfg.include_dd_td),
                "test_bets_graded": bets_graded,
                "test_wins": wins,
                "test_losses": losses,
                "test_pushes": pushes,
                "test_profit": profit,
                "test_roi_per_bet": (profit / bets_graded) if bets_graded else np.nan,
            }
        )

    out_df = pd.DataFrame(daily_rows)
    console.rule("Walk-Forward Summary")
    hit = (total_w / (total_w + total_l)) if (total_w + total_l) else float("nan")
    roi = (total_profit / total_graded) if total_graded else float("nan")
    console.print({"test_start": test_start, "test_end": test_end, "train_days": int(train_days), "graded_bets": int(total_graded), "wins": int(total_w), "losses": int(total_l), "profit": float(total_profit), "hit_rate": hit, "roi_per_bet": roi})

    # Show most recent days
    try:
        show = out_df.copy()
        if "date" in show.columns:
            show = show.sort_values(by=["date"], ascending=True)
        console.rule("Per-Day (tail)")
        cols = [c for c in ["date", "cfg_sort_by", "cfg_include_stats", "cfg_min_edge", "cfg_min_ev", "cfg_min_price", "cfg_max_price", "cfg_exclude_bookmakers", "cfg_top_n_per_day", "test_profit", "test_roi_per_bet", "test_bets_graded"] if c in show.columns]
        console.print(show[cols].tail(20) if cols else show.tail(20))
    except Exception:
        pass

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        console.print(f"Wrote walk-forward results: {out_path}", style="green")
    else:
        try:
            out = paths.data_processed / f"props_edges_walkforward_{test_start}_to_{test_end}_train{int(train_days)}.csv"
            out_df.to_csv(out, index=False)
            console.print(f"Wrote walk-forward results: {out}", style="green")
        except Exception:
            pass


@cli.command("predict-props")
@click.option("--date", "date_str", type=str, required=True, help="Prediction date YYYY-MM-DD (features built up to the day before)")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=False, help="Output CSV path (default props_predictions_YYYY-MM-DD.csv)")
@click.option("--slate-only/--no-slate-only", default=True, show_default=True, help="Filter predictions to teams on the scoreboard slate and add opponent/home flags")
@click.option("--calibrate/--no-calibrate", default=True, show_default=True, help="Apply rolling bias calibration from recent recon vs predictions")
@click.option("--calib-window", type=int, default=7, show_default=True, help="Lookback days for calibration window (excludes today)")
@click.option("--calibrate-player/--no-calibrate-player", default=False, show_default=True, help="Apply per-player rolling bias calibration when enough matches exist")
@click.option("--player-calib-window", type=int, default=30, show_default=True, help="Lookback days for per-player calibration (excludes today)")
@click.option("--player-min-pairs", type=int, default=6, show_default=True, help="Minimum matched rows per player/stat to apply adjustment")
@click.option("--player-shrink-k", type=int, default=8, show_default=True, help="Empirical-Bayes shrinkage K for per-player adjustments")
@click.option("--player-shrink-k-by-stat", type=str, required=False, help="Optional per-stat shrinkage mapping, e.g., 'reb:12,ast:12' (others default to --player-shrink-k)")
@click.option("--player-min-pairs-by-stat", type=str, required=False, help="Optional per-stat min-pairs mapping, e.g., 'reb:8,ast:8' (others default to --player-min-pairs)")
@click.option("--use-pure-onnx/--no-use-pure-onnx", default=True, show_default=True, help="Use pure ONNX models with NPU acceleration (no sklearn dependency)")
@click.option("--use-smart-sim/--no-use-smart-sim", default=True, show_default=True, help="Derive prop stat means from SmartSim (minutes/rotations-aware simulation) when possible")
@click.option("--smart-sim-n-sims", type=int, default=2000, show_default=True, help="SmartSim simulations per game")
@click.option("--smart-sim-pbp/--no-smart-sim-pbp", default=True, show_default=True, help="Use possession-level SmartSim mode")
@click.option("--smart-sim-workers", type=int, default=1, show_default=True, help="Parallel workers (per-game) for SmartSim during predict-props")
@click.option("--smart-sim-overwrite", is_flag=True, default=False, help="Overwrite existing smart_sim_<date>_*.json outputs")
@click.option("--smart-sim-roster-mode", type=str, default="historical", show_default=True, help="Roster sourcing mode for SmartSim: historical or pregame")
@click.option("--smart-sim-out-prefix", type=str, default="smart_sim", show_default=True, help="Output filename prefix for SmartSim artifacts consumed by predict-props")
def predict_props_cmd(date_str: str, out_path: str | None, slate_only: bool, calibrate: bool, calib_window: int,
                      calibrate_player: bool, player_calib_window: int, player_min_pairs: int, player_shrink_k: int,
                      player_shrink_k_by_stat: str | None, player_min_pairs_by_stat: str | None,
                      use_pure_onnx: bool,
                      use_smart_sim: bool,
                      smart_sim_n_sims: int,
                      smart_sim_pbp: bool,
                      smart_sim_workers: int,
                      smart_sim_overwrite: bool,
                      smart_sim_roster_mode: str,
                      smart_sim_out_prefix: str):
    """Predict player props for a slate date using rolling-history models.

    Note: This version builds features from history only and returns predictions for all players seen in logs. A later enhancement can filter to the actual slate roster for the date and merge odds.
    """
    console.rule("Predict Props")
    
    # Build features (no sklearn required). If a pure builder exists, use it; else use standard builder.
    try:
        if use_pure_onnx:
            try:
                from .props_features_pure import build_features_for_date_pure
                console.print("Building features with pure method (no sklearn)...", style="cyan")
                feats = build_features_for_date_pure(date_str)
            except ModuleNotFoundError:
                # Fall back to standard feature builder (still sklearn-free)
                feats = build_features_for_date(date_str)
            except Exception as e:
                # Do not fall back to sklearn; fail fast in ONNX-only mode
                raise click.ClickException(f"Failed to build features in pure mode: {e}")
        else:
            feats = build_features_for_date(date_str)
    except Exception as e:
        raise click.ClickException(f"Failed to build features for {date_str}: {e}")
    # League-wide team cleanup (before slate filter): fix feats['team'] using latest roster, overrides, NBA API fallback; also integrate league_status
    try:
        import re as _re
        from .config import paths as _paths
        from .teams import to_tricode as _to_tri
        import pandas as _pd
        from pathlib import Path as _Path
        # Load latest roster file (avoid cross-season contamination)
        def _load_latest_roster_df():
            try:
                proc = _paths.data_processed
                files = list(proc.glob("rosters_*.csv"))
                if not files:
                    return _pd.DataFrame(columns=["PLAYER","PLAYER_ID","TEAM_ABBREVIATION"])

                # Prefer the season-spanning roster for this slate date (e.g., rosters_2025-26.csv)
                try:
                    d = _pd.to_datetime(date_str, errors="coerce")
                    if d is not None and not _pd.isna(d):
                        start_year = int(d.year) if int(d.month) >= 7 else int(d.year) - 1
                        season = f"{start_year}-{str(start_year+1)[-2:]}"
                        exact = proc / f"rosters_{season}.csv"
                        if exact.exists():
                            files = [exact]
                except Exception:
                    pass

                # Next preference: season-format files (contain '-') sorted by mtime
                season_files = [f for f in files if '-' in f.stem]
                if season_files:
                    season_files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
                    p = season_files[0]
                else:
                    files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
                    p = files[0]

                df = _pd.read_csv(p)
                if df is None or df.empty:
                    return _pd.DataFrame(columns=["PLAYER","PLAYER_ID","TEAM_ABBREVIATION"])
                cols = {c.upper(): c for c in df.columns}
                if "TEAM_ABBREVIATION" not in cols:
                    df["TEAM_ABBREVIATION"] = None
                return df
            except Exception:
                return _pd.DataFrame(columns=["PLAYER","PLAYER_ID","TEAM_ABBREVIATION"])
        # Normalize name
        _SUFFIXES = {"jr","sr","ii","iii","iv","v"}
        def _norm_name_key(s: str) -> str:
            s = (s or "").strip().lower()
            try:
                import unicodedata as _ud
                s = _ud.normalize("NFKD", s)
                s = s.encode("ascii", "ignore").decode("ascii")
            except Exception:
                pass
            s = _re.sub(r"[^a-z0-9\s]", "", s)
            s = _re.sub(r"\s+", " ", s).strip()
            toks = [t for t in s.split(" ") if t and t not in _SUFFIXES]
            return " ".join(toks)
        # Build pid->team and name->team.
        # Authoritative source: player_logs (last-known team at/before date).
        # This avoids bad/corrupt roster files poisoning the cache.
        pid_to_tri, name_to_tri = {}, {}
        try:
            logs_pq = _paths.data_processed / "player_logs.parquet"
            logs_csv = _paths.data_processed / "player_logs.csv"
            logs = None
            if logs_csv.exists():
                logs = _pd.read_csv(logs_csv)
            elif logs_pq.exists():
                try:
                    logs = _pd.read_parquet(logs_pq)
                except Exception:
                    logs = None
            if isinstance(logs, _pd.DataFrame) and not logs.empty:
                lcols = {c.upper(): c for c in logs.columns}
                pid_c = lcols.get("PLAYER_ID")
                tri_c = lcols.get("TEAM_ABBREVIATION")
                name_c = lcols.get("PLAYER_NAME")
                date_c = lcols.get("GAME_DATE") or lcols.get("DATE")
                if pid_c and tri_c and date_c:
                    tmp = logs[[pid_c, tri_c, date_c] + ([name_c] if name_c else [])].copy()
                    tmp[pid_c] = _pd.to_numeric(tmp[pid_c], errors="coerce")
                    tmp[tri_c] = tmp[tri_c].astype(str).map(lambda x: (_to_tri(str(x)) or str(x).strip().upper()))
                    tmp[date_c] = _pd.to_datetime(tmp[date_c], errors="coerce")
                    # Use last known team at/before date_str
                    cut = _pd.to_datetime(date_str, errors="coerce")
                    if _pd.notna(cut):
                        tmp = tmp[tmp[date_c].notna() & (tmp[date_c] <= cut)]
                    tmp = tmp.dropna(subset=[pid_c])
                    tmp = tmp[tmp[tri_c].astype(str).str.len() > 0]
                    if not tmp.empty:
                        tmp = tmp.sort_values(date_c)
                        last = tmp.groupby(pid_c, as_index=False).tail(1)
                        for _, r in last.iterrows():
                            try:
                                pid = int(r[pid_c])
                            except Exception:
                                continue
                            tri = str(r[tri_c]).strip().upper()
                            if pid and tri:
                                pid_to_tri[pid] = tri
                            if name_c:
                                try:
                                    nkey = _norm_name_key(str(r.get(name_c) or ""))
                                    if nkey and tri:
                                        name_to_tri[nkey] = tri
                                except Exception:
                                    pass
        except Exception:
            pass

        # Secondary source: latest roster file (ONLY fills missing entries)
        rost = _load_latest_roster_df()
        if not rost.empty:
            rcols = {c.upper(): c for c in rost.columns}
            pid_col = rcols.get("PLAYER_ID"); name_col = rcols.get("PLAYER"); tri_col = rcols.get("TEAM_ABBREVIATION")
            if pid_col and name_col and tri_col:
                tmp = rost[[pid_col, name_col, tri_col]].copy()
                tmp[tri_col] = tmp[tri_col].astype(str).map(lambda x: (_to_tri(str(x)) or str(x).strip().upper()))
                for _, r in tmp.iterrows():
                    try:
                        tri = str(r[tri_col]).strip().upper()
                        if not tri:
                            continue
                        try:
                            pid = int(_pd.to_numeric(r[pid_col], errors="coerce"))
                        except Exception:
                            pid = None
                        if pid and (pid not in pid_to_tri):
                            pid_to_tri[pid] = tri
                        try:
                            nkey = _norm_name_key(str(r[name_col]))
                            if nkey and (nkey not in name_to_tri):
                                name_to_tri[nkey] = tri
                        except Exception:
                            pass
                    except Exception:
                        continue
        # Apply global roster overrides if present
        try:
            ov = _paths.root / "data" / "overrides" / "roster_overrides.csv"
            if ov.exists():
                odf = _pd.read_csv(ov)
                if odf is not None and not odf.empty:
                    ocols = {c.upper(): c for c in odf.columns}
                    opid = ocols.get("PLAYER_ID"); oname = ocols.get("PLAYER"); otri = ocols.get("TEAM_ABBREVIATION")
                    if otri and (opid or oname):
                        tmpo = odf[[c for c in [opid, oname, otri] if c]].copy()
                        tmpo[otri] = tmpo[otri].astype(str).map(lambda x: (_to_tri(str(x)) or str(x).strip().upper()))
                        for _, r in tmpo.iterrows():
                            try:
                                tri = str(r[otri]).strip().upper()
                                if not tri:
                                    continue
                                if opid and _pd.notna(r.get(opid)):
                                    try:
                                        pid = int(_pd.to_numeric(r[opid], errors="coerce"))
                                        pid_to_tri[pid] = tri
                                    except Exception:
                                        pass
                                if oname and _pd.notna(r.get(oname)):
                                    try:
                                        nkey = _norm_name_key(str(r[oname]))
                                        if nkey:
                                            name_to_tri[nkey] = tri
                                    except Exception:
                                        pass
                            except Exception:
                                continue
        except Exception:
            pass
        # Load cache and resolver using nba_api CommonPlayerInfo
        cache_p = _paths.data_processed / "player_team_cache.csv"
        cache = {}
        if cache_p.exists():
            try:
                cdf = _pd.read_csv(cache_p)
                if cdf is not None and not cdf.empty and {"player_id","team"}.issubset(set(cdf.columns)):
                    for _, r in cdf.iterrows():
                        try:
                            cache[int(_pd.to_numeric(r["player_id"], errors="coerce"))] = str(r["team"]).strip().upper()
                        except Exception:
                            continue
            except Exception:
                pass

        # If we have authoritative pid_to_tri from logs, make it override cache.
        # This keeps results predictable and self-heals poisoned cache entries.
        try:
            if pid_to_tri:
                for pid, tri in pid_to_tri.items():
                    if pid and tri:
                        cache[int(pid)] = str(tri).strip().upper()
        except Exception:
            pass
        def _resolve_pid_team(pid: int) -> str | None:
            if pid in pid_to_tri:
                return pid_to_tri[pid]
            if pid in cache:
                return cache[pid]
            # NBA API call with small timeout and graceful failure
            try:
                from nba_api.stats.endpoints import commonplayerinfo as _cpi
                resp = _cpi.CommonPlayerInfo(player_id=int(pid), timeout=10)
                nd = resp.get_normalized_dict()
                rows = nd.get("CommonPlayerInfo", [])
                if rows:
                    tri = str(rows[0].get("TEAM_ABBREVIATION") or "").strip().upper()
                    if tri:
                        cache[pid] = tri
                        return tri
            except Exception:
                return None
            return None
        # Apply mapping to feats
        if not feats.empty and ("team" in feats.columns):
            try:
                feats = feats.copy()
                # Prefer pid, then name
                if "player_id" in feats.columns:
                    feats["player_id"] = _pd.to_numeric(feats["player_id"], errors="coerce")
                if "player_name" in feats.columns:
                    feats["_name_key"] = feats["player_name"].astype(str).map(_norm_name_key)
                # Resolve team
                def _team_row(row):
                    # roster override via pid
                    pid = row.get("player_id") if "player_id" in row else None
                    tri = None
                    if pid is not None and _pd.notna(pid):
                        tri = _resolve_pid_team(int(pid))
                    if not tri and "_name_key" in row and row.get("_name_key"):
                        tri = name_to_tri.get(row.get("_name_key"))
                    if not tri:
                        # keep existing but normalize
                        tri = _to_tri(str(row.get("team"))) or str(row.get("team") or "").strip().upper()
                    return tri
                feats["team"] = feats.apply(_team_row, axis=1)
                # Clean up
                feats.drop(columns=["_name_key"], inplace=True, errors="ignore")
            except Exception:
                pass
        # Persist cache (logs-derived mapping overrides older entries)
        try:
            if cache:
                rows = [(k, v) for k, v in cache.items()]
                _pd.DataFrame(rows, columns=["player_id","team"]).to_csv(cache_p, index=False)
        except Exception:
            pass
        # Integrate league_status (ensures FA/non-playing dropped and team corrected)
        try:
            # Prefer the pipeline-built artifact for determinism
            ls_path = _paths.data_processed / f"league_status_{date_str}.csv"
            if ls_path.exists():
                ls = _pd.read_csv(ls_path)
            else:
                ls = build_league_status(date_str)
            if ls is not None and not ls.empty and {"player_id","team","injury_status","team_on_slate","playing_today"}.issubset(set(ls.columns)) and ("player_id" in feats.columns):
                feats = feats.copy()
                feats["player_id"] = _pd.to_numeric(feats["player_id"], errors="coerce")
                ls2 = ls[["player_id","team","injury_status","team_on_slate","playing_today"]].copy()
                ls2["player_id"] = _pd.to_numeric(ls2["player_id"], errors="coerce")
                feats = feats.merge(ls2, on="player_id", how="left", suffixes=("","_ls"))
                # league_status is authoritative for the slate date; override team when present
                feats["team_ls"] = feats["team_ls"].fillna("").astype(str).str.upper().str.strip()
                feats["team"] = feats["team_ls"].where(feats["team_ls"].astype(str).str.len() > 0, feats["team"])
                def _ok(row):
                    t = str(row.get("team") or "").strip().upper()
                    if not t:
                        return False
                    pt = row.get("playing_today")
                    if _pd.notna(pt) and (pt is False):
                        return False
                    return True
                feats = feats[feats.apply(_ok, axis=1)].drop(columns=[c for c in feats.columns if c.endswith("_ls")], errors="ignore")
        except Exception as _ee:
            pass
    except Exception:
        pass

    # Optional slate filter using ScoreboardV2, with fallback to OddsAPI game odds
    if slate_only:
        slate_applied = False
        # Primary: NBA ScoreboardV2
        try:
            sb = scoreboardv2.ScoreboardV2(game_date=date_str, day_offset=0, timeout=30)
            nd = sb.get_normalized_dict()
            gh = pd.DataFrame(nd.get("GameHeader", []))
            ls = pd.DataFrame(nd.get("LineScore", []))
            if not gh.empty and not ls.empty:
                ls_cols = {c.upper(): c for c in ls.columns}
                if {"TEAM_ID","TEAM_ABBREVIATION"}.issubset(ls_cols.keys()):
                    team_map = {}
                    for _, r in ls.iterrows():
                        try:
                            team_map[int(r[ls_cols["TEAM_ID"]])] = str(r[ls_cols["TEAM_ABBREVIATION"]]).upper()
                        except Exception:
                            continue
                    gh_cols = {c.upper(): c for c in gh.columns}
                    if {"HOME_TEAM_ID","VISITOR_TEAM_ID"}.issubset(gh_cols.keys()):
                        games = []
                        for _, g in gh.iterrows():
                            try:
                                hid = int(g[gh_cols["HOME_TEAM_ID"]]); vid = int(g[gh_cols["VISITOR_TEAM_ID"]])
                                h = team_map.get(hid); v = team_map.get(vid)
                                if h and v:
                                    games.append({"team": h, "opponent": v, "home": True})
                                    games.append({"team": v, "opponent": h, "home": False})
                            except Exception:
                                continue
                        slate = pd.DataFrame(games)
                        if not slate.empty and "team" in feats.columns:
                            feats["team"] = feats["team"].astype(str).str.upper()
                            feats = feats.merge(slate, on="team", how="inner")
                            slate_applied = True
        except Exception:
            pass
        # Fallback: use standardized OddsAPI game odds CSV written earlier in the pipeline
        if not slate_applied:
            try:
                from .config import paths as _paths
                from .teams import to_tricode as _to_tri
                go_path = _paths.data_processed / f"game_odds_{date_str}.csv"
                if go_path.exists():
                    go = pd.read_csv(go_path)
                    if not go.empty:
                        games = []
                        # Support both visitor_team and away_team column names
                        home_col = "home_team" if "home_team" in go.columns else None
                        away_col = "visitor_team" if "visitor_team" in go.columns else ("away_team" if "away_team" in go.columns else None)
                        if home_col and away_col:
                            for _, r in go.iterrows():
                                try:
                                    h_raw = str(r.get(home_col) or "").strip()
                                    a_raw = str(r.get(away_col) or "").strip()
                                    h = _to_tri(h_raw)
                                    a = _to_tri(a_raw)
                                    if h and a:
                                        games.append({"team": h, "opponent": a, "home": True})
                                        games.append({"team": a, "opponent": h, "home": False})
                                except Exception:
                                    continue
                        slate = pd.DataFrame(games)
                        if not slate.empty and "team" in feats.columns:
                            feats["team"] = feats["team"].astype(str).str.upper()
                            feats = feats.merge(slate, on="team", how="inner")
                            slate_applied = True
            except Exception:
                pass
        try:
            if "home" in feats.columns:
                feats = feats.copy()
                feats["is_home"] = pd.to_numeric(feats["home"], errors="coerce").fillna(0.0).astype(float)
        except Exception:
            pass
    
    # Optional: exclude clearly inactive players based on ESPN injuries DB (OUT)
    try:
        import pandas as _pd
        from .teams import to_tricode as _to_tri
        inj_path = paths.data_raw / "injuries.csv"
        if not feats.empty:
            # Consistent name normalization for matching injuries -> features
            import re as _re
            _SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
            def _norm_name_key(s: str) -> str:
                s = (s or "").strip().lower()
                try:
                    import unicodedata as _ud
                    s = _ud.normalize("NFKD", s)
                    s = s.encode("ascii", "ignore").decode("ascii")
                except Exception:
                    pass
                # remove punctuation
                s = _re.sub(r"[^a-z0-9\s]", "", s)
                # collapse whitespace
                s = _re.sub(r"\s+", " ", s).strip()
                # drop common suffix tokens
                toks = [t for t in s.split(" ") if t and t not in _SUFFIXES]
                return " ".join(toks)
            day_out = _pd.DataFrame()
            # Primary source: scraped injuries DB
            if inj_path.exists():
                inj = _pd.read_csv(inj_path)
                if not inj.empty and {"team","player","status","date"}.issubset(set(inj.columns)):
                    # Consider the latest status up to the target date, not just exact date rows
                    inj["date"] = _pd.to_datetime(inj["date"], errors="coerce").dt.date
                    cutoff = _pd.to_datetime(date_str).date()
                    inj = inj[inj["date"].notna()]
                    inj = inj[inj["date"] <= cutoff].copy()
                    if not inj.empty:
                        # take latest per player+team
                        inj = inj.sort_values(["date"])  # ascending so tail(1) gives latest
                        grp_cols = [c for c in ["player","team"] if c in inj.columns]
                        if not grp_cols:
                            grp_cols = ["player"]
                        latest = inj.groupby(grp_cols, as_index=False).tail(1)
                        # Ensure a standalone copy before column assignments to avoid SettingWithCopyWarning
                        latest = latest.copy()
                        latest["team_tri"] = latest["team"].astype(str).map(lambda x: _to_tri(str(x)))
                        latest["player_key"] = latest["player"].astype(str).map(_norm_name_key)
                        latest["status_norm"] = latest["status"].astype(str).str.upper()
                        latest["injury_norm"] = latest.get("injury", "").astype(str).str.upper() if ("injury" in latest.columns) else ""
                        # Exclusion logic: exact statuses plus season-long/indefinite phrasing
                        def _excluded_status(u: str) -> bool:
                            try:
                                u = str(u).upper()
                            except Exception:
                                return False
                            if u in {"OUT","DOUBTFUL","SUSPENDED","INACTIVE","REST"}:
                                return True
                            if ("OUT" in u and ("SEASON" in u or "INDEFINITE" in u)) or ("SEASON-ENDING" in u):
                                return True
                            return False
                        # Recency window to avoid stale OUT labels persisting forever when the feed
                        # isn't updated for return-to-play. Still allow season-long/indefinite outs.
                        try:
                            recency_days = 30
                            d0 = _pd.to_datetime(date_str, errors="coerce")
                            if not _pd.isna(d0) and ("date" in latest.columns):
                                latest["_date"] = _pd.to_datetime(latest["date"], errors="coerce").dt.date
                                latest = latest[latest["_date"].notna()].copy()
                                fresh_cutoff = (d0 - _pd.Timedelta(days=int(recency_days))).date()
                                season = latest["status_norm"].astype(str).str.contains("SEASON", na=False) | \
                                         latest["status_norm"].astype(str).str.contains("INDEFINITE", na=False) | \
                                         latest["status_norm"].astype(str).str.contains("SEASON-ENDING", na=False)
                                if isinstance(latest.get("injury_norm"), _pd.Series):
                                    season = season | latest["injury_norm"].astype(str).str.contains("OUT FOR SEASON", na=False) | \
                                             latest["injury_norm"].astype(str).str.contains("SEASON-ENDING", na=False) | \
                                             latest["injury_norm"].astype(str).str.contains("INDEFINITE", na=False)
                                latest = latest[(latest["_date"] >= fresh_cutoff) | season].copy()
                        except Exception:
                            pass
                        day_out = latest[latest["status_norm"].map(_excluded_status)].copy()
            # Manual override per-day file support
            try:
                override_path = paths.data_raw / f"injuries_overrides_{date_str}.csv"
                if override_path.exists():
                    ovr = _pd.read_csv(override_path)
                    if not ovr.empty and {"team","player","status"}.issubset(set(ovr.columns)):
                        ovr = ovr.copy()
                        ovr["team_tri"] = ovr["team"].astype(str).map(lambda x: _to_tri(str(x)))
                        ovr["player_key"] = ovr["player"].astype(str).map(_norm_name_key)
                        ovr_status = ovr["status"].astype(str).str.upper().str.strip()
                        # Allow-list statuses remove a player from exclusions.
                        allow = {"IN", "ACTIVE", "AVAILABLE"}
                        try:
                            allow_set = set(
                                (str(r.get("player_key") or ""), str(r.get("team_tri") or ""))
                                for _, r in ovr[ovr_status.isin(allow)].iterrows()
                            )
                            if allow_set and not day_out.empty:
                                day_out = day_out.copy()
                                day_out["_team_tri"] = day_out.get("team_tri", "").astype(str)
                                day_out["_pkey"] = day_out.get("player_key", "").astype(str)
                                day_out = day_out[~day_out.apply(lambda r: (str(r.get("_pkey")), str(r.get("_team_tri"))) in allow_set, axis=1)].copy()
                        except Exception:
                            pass
                        # Apply exclusion logic to override rows.
                        ovr_mask = ovr_status.isin({"OUT","DOUBTFUL","SUSPENDED","INACTIVE","REST"}) | \
                                   (ovr_status.str.contains("OUT", na=False) & (ovr_status.str.contains("SEASON", na=False) | ovr_status.str.contains("INDEFINITE", na=False))) | \
                                   (ovr_status.str.contains("SEASON-ENDING", na=False))
                        ovr_out = ovr[ovr_mask]
                        day_out = _pd.concat([day_out, ovr_out], ignore_index=True) if not day_out.empty else ovr_out
            except Exception:
                pass

            if not day_out.empty and {"team","player_name"}.issubset(set(feats.columns)):
                tmp = feats.copy()
                tmp["team_tri"] = tmp["team"].astype(str).map(lambda x: _to_tri(str(x)))
                tmp["player_key"] = tmp["player_name"].astype(str).map(_norm_name_key)

                # Defense-in-depth allow-lists:
                # - If we have already-played boxscores for this date, never exclude players who logged minutes.
                # - If OddsAPI props contain a player, never exclude them (feed/team mismatches happen).
                try:
                    allow_pairs: set[tuple[str, str]] = set()
                    allow_players: set[str] = set()

                    # Boxscores allow-list (completed games only)
                    try:
                        bs_p = paths.data_processed / f"boxscores_{date_str}.csv"
                        if bs_p.exists():
                            bs = _pd.read_csv(bs_p)
                            if bs is not None and (not bs.empty):
                                name_col = next((c for c in bs.columns if c.upper() in {"PLAYER_NAME","PLAYER"}), None)
                                min_col = next((c for c in bs.columns if c.upper() == "MIN"), None)
                                team_col = next((c for c in bs.columns if c.upper() in {"TEAM_ABBREVIATION","TEAM"}), None)
                                if name_col and min_col:
                                    bs = bs.copy()
                                    bs[min_col] = _pd.to_numeric(bs[min_col], errors="coerce").fillna(0.0)
                                    played = bs[bs[min_col] > 0].copy()
                                    if not played.empty:
                                        played["_pkey"] = played[name_col].astype(str).map(_norm_name_key)
                                        allow_players |= set(played["_pkey"].dropna().astype(str).tolist())
                                        if team_col:
                                            played["_tri"] = played[team_col].astype(str).map(lambda x: _to_tri(str(x)) or str(x))
                                            played["_tri"] = played["_tri"].astype(str).str.upper().str.strip()
                                            allow_pairs |= set(
                                                (str(r.get("_pkey") or ""), str(r.get("_tri") or ""))
                                                for _, r in played[["_pkey", "_tri"]].iterrows()
                                                if str(r.get("_pkey") or "")
                                            )
                    except Exception:
                        pass

                    # OddsAPI props allow-list (pre-game)
                    try:
                        raw_props = paths.data_raw / f"odds_nba_player_props_{date_str}.csv"
                        if raw_props.exists():
                            pr = _pd.read_csv(raw_props)
                            if pr is not None and (not pr.empty):
                                name_col = next((c for c in pr.columns if c.lower() in ("player", "player_name", "name")), None)
                                team_col = next((c for c in pr.columns if c.lower() in ("team", "team_abbr", "team_abbrev", "team_abbreviation")), None)
                                if name_col:
                                    pr = pr.copy()
                                    pr["_pkey"] = pr[name_col].astype(str).map(_norm_name_key)
                                    allow_players |= set(pr["_pkey"].dropna().astype(str).tolist())
                                    if team_col:
                                        pr["_tri"] = pr[team_col].astype(str).map(lambda x: _to_tri(str(x)) or str(x))
                                        pr["_tri"] = pr["_tri"].astype(str).str.upper().str.strip()
                                        allow_pairs |= set(
                                            (str(r.get("_pkey") or ""), str(r.get("_tri") or ""))
                                            for _, r in pr[["_pkey", "_tri"]].iterrows()
                                            if str(r.get("_pkey") or "")
                                        )
                    except Exception:
                        pass

                    if (allow_pairs or allow_players) and (not day_out.empty):
                        day_out = day_out.copy()
                        day_out["_team_tri"] = day_out.get("team_tri", "").astype(str).str.upper().str.strip()
                        day_out["_pkey"] = day_out.get("player_key", "").astype(str)
                        day_out = day_out[
                            ~day_out.apply(
                                lambda r: (
                                    (str(r.get("_pkey") or ""), str(r.get("_team_tri") or "")) in allow_pairs
                                    or (str(r.get("_pkey") or "") in allow_players)
                                ),
                                axis=1,
                            )
                        ].copy()
                        day_out = day_out.drop(columns=["_team_tri", "_pkey"], errors="ignore")
                except Exception:
                    pass

                # Repair/standardize injury-feed team assignments using processed rosters for the
                # slate date (season-appropriate). This prevents mis-tagged rows (e.g., feed glitches
                # around trades) from excluding the wrong team and from writing bad
                # injuries_excluded_<date>.csv.
                try:
                    if isinstance(day_out, _pd.DataFrame) and (not day_out.empty) and ("player_key" in day_out.columns):
                        day_out = day_out.copy()
                        if "team_tri" not in day_out.columns:
                            if "team" in day_out.columns:
                                day_out["team_tri"] = day_out["team"].astype(str).map(lambda x: _to_tri(str(x)))
                            else:
                                day_out["team_tri"] = ""
                        day_out["team_tri"] = day_out["team_tri"].astype(str).str.upper().str.strip()

                        roster_map: dict[str, str] = {}
                        try:
                            proc = paths.data_processed
                            d = _pd.to_datetime(date_str, errors="coerce")
                            if d is not None and (not _pd.isna(d)):
                                start_year = int(d.year) if int(d.month) >= 7 else int(d.year) - 1
                                season = f"{start_year}-{str(start_year+1)[-2:]}"
                                cand = proc / f"rosters_{season}.csv"
                            else:
                                cand = None
                            roster_file = cand if (cand is not None and cand.exists()) else None
                            if roster_file is None:
                                files = list(proc.glob("rosters_*.csv"))
                                season_files = [f for f in files if "-" in f.stem]
                                if season_files:
                                    season_files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
                                    roster_file = season_files[0]
                                elif files:
                                    files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
                                    roster_file = files[0]
                            if roster_file is not None and roster_file.exists():
                                rdf = _pd.read_csv(roster_file)
                                if rdf is not None and (not rdf.empty):
                                    cols = {c.upper(): c for c in rdf.columns}
                                    name_col = cols.get("PLAYER")
                                    tri_col = cols.get("TEAM_ABBREVIATION")
                                    if name_col and tri_col:
                                        for _, rr in rdf[[name_col, tri_col]].dropna().iterrows():
                                            try:
                                                pk = _norm_name_key(str(rr.get(name_col) or ""))
                                                tri = str(_to_tri(str(rr.get(tri_col) or "")) or "").strip().upper()
                                                if pk and tri:
                                                    roster_map[pk] = tri
                                            except Exception:
                                                continue
                        except Exception:
                            roster_map = {}

                        def _fix_tri(r):
                            try:
                                pk = str(r.get("player_key") or "")
                                tri = str(r.get("team_tri") or "").strip().upper()
                                corr = roster_map.get(pk)
                                if corr and (not tri or tri != corr):
                                    return corr
                                return tri
                            except Exception:
                                return str(r.get("team_tri") or "").strip().upper()

                        day_out["team_tri"] = day_out.apply(_fix_tri, axis=1)
                        if "team" in day_out.columns:
                            # Keep a simple, consistent team column for downstream consumers.
                            day_out["team"] = day_out["team_tri"].where(day_out["team_tri"].astype(str).str.len() > 0, day_out["team"])
                except Exception:
                    pass

                try:
                    _, _, ls_allowed = _league_status_player_maps_for_date(date_str)
                    if ls_allowed and not day_out.empty:
                        day_out = day_out.copy()
                        day_out["_team_tri"] = day_out.get("team_tri", "").astype(str).str.upper().str.strip()
                        day_out["_pkey"] = day_out.get("player_key", "").astype(str).str.upper().str.strip()
                        day_out = day_out[
                            ~day_out.apply(
                                lambda r: str(r.get("_pkey") or "") in ls_allowed.get(str(r.get("_team_tri") or ""), set()),
                                axis=1,
                            )
                        ].copy()
                        day_out = day_out.drop(columns=["_team_tri", "_pkey"], errors="ignore")
                except Exception:
                    pass

                ban = set((str(r.get("player_key")), str(r.get("team_tri"))) for _, r in day_out.iterrows())
                before = len(tmp)
                tmp = tmp[~tmp.apply(lambda r: (str(r.get("player_key")), str(r.get("team_tri"))) in ban, axis=1)]
                if len(tmp) < before:
                    console.print(f"Filtered OUT players by injuries/overrides: removed {before-len(tmp)} rows", style="yellow")
                # Persist a small diagnostics file to aid debugging
                try:
                    diag_cols = [c for c in ["date","team","team_tri","player","status","injury"] if c in day_out.columns]
                    diag = day_out[diag_cols].copy() if diag_cols else day_out.copy()
                    out_diag = paths.data_processed / f"injuries_excluded_{date_str}.csv"
                    diag.to_csv(out_diag, index=False)
                except Exception:
                    pass
                feats = tmp.drop(columns=["team_tri","player_key"], errors="ignore")
    except Exception as _e:
        console.print(f"Injury filter skipped: {_e}", style="yellow")

    # Enforce slate participants via OddsAPI player props.
    # IMPORTANT: this must remain a *soft* cleanup only. OddsAPI player-props feeds can be incomplete,
    # so never prune rows that are already protected by league_status / slate membership (or completed boxscores).
    try:
        from .config import paths as _paths
        import pandas as _pd
        import re as _re
        raw_props = _paths.data_raw / f"odds_nba_player_props_{date_str}.csv"
        if raw_props.exists() and not feats.empty and ("player_name" in feats.columns):
            props = _pd.read_csv(raw_props)
            if props is not None and not props.empty:
                name_col = next((c for c in props.columns if c.lower() in ("player","player_name","name")), None)
                if name_col:
                    def _norm(s: str) -> str:
                        s = (s or "").strip().lower()
                        s = _re.sub(r"[^a-z0-9\s]", "", s)
                        s = _re.sub(r"\s+", " ", s).strip()
                        toks = [t for t in s.split(" ") if t not in {"jr","sr","ii","iii","iv","v"}]
                        return " ".join(toks)
                    pset = set(props[name_col].astype(str).map(_norm).tolist())
                    feats = feats.copy()
                    feats["_pkey"] = feats["player_name"].astype(str).map(_norm)
                    before = int(len(feats))
                    odds_match = feats["_pkey"].isin(pset)
                    keep_mask = odds_match.copy()

                    try:
                        if "playing_today" in feats.columns:
                            pt = feats["playing_today"]
                            pt_keep = pt.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})
                            try:
                                pt_num = _pd.to_numeric(pt, errors="coerce")
                                pt_keep = pt_keep | (pt_num.fillna(0.0).astype(float) > 0.5)
                            except Exception:
                                pass
                            keep_mask = keep_mask | pt_keep.astype(bool)
                    except Exception:
                        pass

                    try:
                        if "team_on_slate" in feats.columns:
                            tos = feats["team_on_slate"]
                            tos_keep = tos.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})
                            try:
                                tos_num = _pd.to_numeric(tos, errors="coerce")
                                tos_keep = tos_keep | (tos_num.fillna(0.0).astype(float) > 0.5)
                            except Exception:
                                pass
                            keep_mask = keep_mask | tos_keep.astype(bool)
                    except Exception:
                        pass

                    try:
                        box_p = _paths.data_processed / f"boxscores_{date_str}.csv"
                        if box_p.exists() and ("player_id" in feats.columns):
                            bdf = _pd.read_csv(box_p, usecols=lambda c: str(c).strip().upper() in {"PLAYER_ID", "MIN"})
                            if ("PLAYER_ID" in bdf.columns) and ("MIN" in bdf.columns):
                                bdf["PLAYER_ID"] = _pd.to_numeric(bdf["PLAYER_ID"], errors="coerce")
                                bdf["MIN"] = _pd.to_numeric(bdf["MIN"], errors="coerce").fillna(0.0)
                                played_ids = set(bdf.loc[(bdf["PLAYER_ID"].notna()) & (bdf["MIN"] > 0.0), "PLAYER_ID"].astype(int).tolist())
                                if played_ids:
                                    feat_pid = _pd.to_numeric(feats["player_id"], errors="coerce")
                                    keep_mask = keep_mask | feat_pid.isin(played_ids)
                    except Exception:
                        pass

                    filtered = feats[keep_mask].drop(columns=["_pkey"], errors="ignore")
                    kept = int(len(filtered))
                    removed = before - kept

                    # Apply only when Odds coverage is sufficiently high.
                    # Heuristics:
                    # - At least 80 unique players in OddsAPI props, OR
                    # - At least ~35% of feature rows match OddsAPI names,
                    # AND
                    # - The filter doesn't remove more than 40% of rows.
                    cov_players = int(len(pset))
                    cov_ratio = (int(odds_match.sum()) / before) if before > 0 else 0.0
                    safe_to_apply = (
                        (cov_players >= 80 or cov_ratio >= 0.35)
                        and (removed <= int(0.40 * max(1, before)))
                        and (kept >= 40)
                    )
                    if safe_to_apply:
                        feats = filtered
                        if removed > 0:
                            console.print(
                                f"Pruned non-participants via OddsAPI props: removed {removed} rows (kept {kept}/{before})",
                                style="yellow",
                            )
                    else:
                        feats = feats.drop(columns=["_pkey"], errors="ignore")
                        console.print(
                            f"Participants filter skipped (Odds coverage too low): odds_players={cov_players}, kept={kept}/{before}",
                            style="yellow",
                        )
    except Exception as _e:
        console.print(f"Participants filter skipped: {_e}", style="yellow")

    # Sanity filter using latest team from player logs (and name->id fallback):
    # drop rows whose last known team is neither their current team nor one of today's slate teams
    try:
        from .config import paths as _paths
        from .teams import to_tricode as _to_tri
        import pandas as _pd
        if not feats.empty and ("player_id" in feats.columns) and ("team" in feats.columns):
            logs_p = _paths.data_processed / "player_logs.csv"
            if logs_p.exists():
                lg = _pd.read_csv(logs_p)
                if lg is not None and not lg.empty:
                    c = {c.upper(): c for c in lg.columns}
                    pid_c = c.get("PLAYER_ID"); team_c = c.get("TEAM_ABBREVIATION"); date_c = c.get("GAME_DATE") or c.get("GAME_DATE_EST")
                    if pid_c and team_c:
                        tmp = lg.copy()
                        if date_c:
                            tmp[date_c] = _pd.to_datetime(tmp[date_c], errors="coerce")
                            tmp = tmp.sort_values([pid_c, date_c])
                            last = tmp.groupby(pid_c, as_index=False).tail(1)
                        else:
                            last = tmp.drop_duplicates(subset=[pid_c], keep="last")
                        last = last[[pid_c, team_c]].copy().rename(columns={pid_c:"player_id", team_c:"last_team"})
                        last["player_id"] = _pd.to_numeric(last["player_id"], errors="coerce")
                        last["last_team"] = last["last_team"].astype(str).map(lambda x: (_to_tri(str(x)) or str(x).strip().upper()))
                        feats = feats.merge(last, on="player_id", how="left")
                        # Fallback: resolve missing last_team by mapping name -> player_id via nba_api static players
                        if "last_team" in feats.columns:
                            missing = feats["last_team"].isna()
                        else:
                            feats["last_team"] = _pd.NA
                            missing = feats["last_team"].isna()
                        if missing.any():
                            try:
                                from nba_api.stats.static import players as _static_players
                                plist = _static_players.get_players()
                                if plist:
                                    pdf = _pd.DataFrame(plist)
                                    cpl = {c.lower(): c for c in pdf.columns}
                                    if cpl.get("id") and cpl.get("full_name"):
                                        pdf = pdf.rename(columns={cpl["id"]: "pid", cpl["full_name"]: "pname"})
                                        def _norm(s: str) -> str:
                                            import re as __re
                                            s = (s or "").strip().lower()
                                            s = __re.sub(r"[^a-z0-9\s]", "", s)
                                            s = __re.sub(r"\s+", " ", s).strip()
                                            toks = [t for t in s.split(" ") if t not in {"jr","sr","ii","iii","iv","v"}]
                                            return " ".join(toks)
                                        pdf["_key"] = pdf["pname"].astype(str).map(_norm)
                                        feats["_key"] = feats["player_name"].astype(str).map(_norm)
                                        kmap = pdf[["_key","pid"]].drop_duplicates()
                                        feats = feats.merge(kmap, on="_key", how="left")
                                        # fill last_team using logs by name-resolved pid
                                        last2 = last.rename(columns={"player_id":"pid"})
                                        feats = feats.merge(last2, on="pid", how="left", suffixes=("","_byname"))
                                        feats["last_team"] = feats["last_team"].fillna(feats.get("last_team_byname"))
                                        feats = feats.drop(columns=[c for c in ["_key","pid","last_team_byname"] if c in feats.columns], errors="ignore")
                            except Exception:
                                pass
                        # Build today's event team set
                        slate_teams = set()
                        try:
                            go_path = _paths.data_processed / f"game_odds_{date_str}.csv"
                            go = _pd.read_csv(go_path) if go_path.exists() else _pd.DataFrame()
                            if not go.empty:
                                hcol = "home_team" if "home_team" in go.columns else None
                                acol = "visitor_team" if "visitor_team" in go.columns else ("away_team" if "away_team" in go.columns else None)
                                if hcol and acol:
                                    for _, r in go.iterrows():
                                        h = _to_tri(str(r.get(hcol) or "")); a = _to_tri(str(r.get(acol) or ""))
                                        if h: slate_teams.add(h)
                                        if a: slate_teams.add(a)
                        except Exception:
                            pass
                        def _keep(row):
                            t = str(row.get("team") or "").strip().upper()
                            lt = str(row.get("last_team") or "").strip().upper()
                            if not lt:
                                return True
                            if lt == t:
                                return True
                            if slate_teams and (lt in slate_teams):
                                # allow if last team is one of today's teams (recent trade edge case)
                                return True
                            return False
                        before = len(feats)
                        feats = feats[feats.apply(_keep, axis=1)]
                        dropped = before - len(feats)
                        if dropped > 0:
                            console.print(f"Dropped {dropped} rows by last-team sanity filter", style="yellow")
                        feats = feats.drop(columns=["last_team"], errors="ignore")
    except Exception as _e:
        console.print(f"Logs-based sanity filter skipped: {_e}", style="yellow")

    # Predict using pure ONNX only
    # Predict using pure ONNX only
    try:
        from .props_onnx_pure import predict_props_pure_onnx
        console.print("Using pure ONNX models with NPU acceleration...", style="cyan")
        preds = predict_props_pure_onnx(feats)
        console.print(f"Pure ONNX predictions generated for {len(preds)} players", style="green")
    except Exception as e:
        raise click.ClickException(f"Failed to run pure ONNX predictions: {e}")

    # Preserve the base ONNX outputs in pred_* and use mean_* as the downstream-consumed
    # columns (optionally overwritten by SmartSim). This prevents feedback loops where
    # SmartSim-derived means overwrite the model predictions file.
    for stat in ("pts", "reb", "ast", "threes", "pra", "stl", "blk", "tov"):
        pred_col = f"pred_{stat}"
        mean_col = f"mean_{stat}"
        if pred_col in preds.columns and mean_col not in preds.columns:
            preds[mean_col] = pd.to_numeric(preds[pred_col], errors="coerce")
    # Optional light calibration (rolling intercept per stat)
    # Optionally override predicted prop means with SmartSim simulation outputs.
    # This uses minutes/rotations when available and produces more realistic distributions.
    if use_smart_sim:
        try:
            import json as _json
            import re as _re
            from pathlib import Path as _Path

            # Ensure leakage-free, as-of priors exist for this date before running SmartSim.
            # (Best-effort: do not fail props pipeline if priors refresh fails.)
            try:
                season = _season_year_from_date_str(date_str)
                p = _ensure_team_advanced_stats_asof(season=season, as_of=date_str)
                if p is not None:
                    console.print(f"SmartSim priors: ensured as-of file {p}", style="cyan")
                else:
                    console.print("SmartSim priors: as-of file not created (continuing)", style="yellow")
            except Exception as _e_priors:
                console.print(f"SmartSim priors refresh failed (continuing): {_e_priors}", style="yellow")

            # Ensure SmartSim can load the props predictions file.
            default_pp = paths.data_processed / f"props_predictions_{date_str}.csv"
            try:
                default_pp.parent.mkdir(parents=True, exist_ok=True)
                preds.to_csv(default_pp, index=False)
            except Exception:
                pass

            # Run SmartSim for all games on this date (writes smart_sim_<date>_<HOME>_<AWAY>.json files)
            try:
                smart_sim_prefix = str(smart_sim_out_prefix or "smart_sim").strip() or "smart_sim"
                summary = _smart_sim_run_date(
                    date_str=date_str,
                    n_sims=int(smart_sim_n_sims),
                    seed=None,
                    max_games=None,
                    overwrite=bool(smart_sim_overwrite),
                    pbp=bool(smart_sim_pbp),
                    workers=int(smart_sim_workers),
                    roster_mode=str(smart_sim_roster_mode or "historical"),
                    out_prefix=smart_sim_prefix,
                )
                console.print({"smart_sim": summary})
            except KeyboardInterrupt:
                raise
            except BaseException as _e:
                console.print(
                    f"SmartSim run skipped due to error ({type(_e).__name__}): {_e}",
                    style="yellow",
                )
                summary = None

            # Parse SmartSim outputs and merge into preds.
            sim_files = sorted(paths.data_processed.glob(f"{smart_sim_prefix}_{date_str}_*.json"))
            if sim_files:
                def _norm_name_key(s: str) -> str:
                    s = (s or "").strip().upper()
                    if "(" in s:
                        s = s.split("(", 1)[0]
                    s = s.replace("-", " ")
                    try:
                        import unicodedata as _ud
                        s = _ud.normalize("NFKD", s)
                        s = s.encode("ascii", "ignore").decode("ascii")
                    except Exception:
                        pass
                    s = _re.sub(r"[^A-Z0-9\s]", "", s)
                    s = _re.sub(r"\s+", " ", s).strip()
                    for suf in (" JR", " SR", " II", " III", " IV"):
                        if s.endswith(suf):
                            s = s[: -len(suf)].strip()
                    return s

                sim_rows: list[dict] = []
                for fp in sim_files:
                    try:
                        obj = _json.loads(_Path(fp).read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    if not isinstance(obj, dict) or obj.get("error"):
                        continue
                    home_tri = str(obj.get("home") or "").strip().upper()
                    away_tri = str(obj.get("away") or "").strip().upper()
                    players = obj.get("players") or {}
                    for side, team_tri in (("home", home_tri), ("away", away_tri)):
                        arr = players.get(side) or []
                        if not isinstance(arr, list):
                            continue
                        for r in arr:
                            if not isinstance(r, dict):
                                continue
                            name = str(r.get("player_name") or "").strip()
                            if not name:
                                continue
                            pid = r.get("player_id")
                            try:
                                pid = int(pd.to_numeric(pid, errors="coerce")) if pid is not None else None
                            except Exception:
                                pid = None
                            opp = away_tri if side == "home" else home_tri
                            sim_rows.append({
                                "team": team_tri,
                                "player_id": pid,
                                "player_name": name,
                                "name_key": _norm_name_key(name),
                                "opponent": opp,
                                "home": True if side == "home" else False,
                                "mean_pts": r.get("pts_mean"),
                                "mean_reb": r.get("reb_mean"),
                                "mean_ast": r.get("ast_mean"),
                                "mean_threes": r.get("threes_mean"),
                                "mean_pra": r.get("pra_mean"),
                                "mean_stl": r.get("stl_mean"),
                                "mean_blk": r.get("blk_mean"),
                                "mean_tov": r.get("tov_mean"),
                                "sd_pts": r.get("pts_sd"),
                                "sd_reb": r.get("reb_sd"),
                                "sd_ast": r.get("ast_sd"),
                                "sd_threes": r.get("threes_sd"),
                                "sd_pra": r.get("pra_sd"),
                                "sd_stl": r.get("stl_sd"),
                                "sd_blk": r.get("blk_sd"),
                                "sd_tov": r.get("tov_sd"),
                            })

                sim_df = pd.DataFrame(sim_rows)
                if sim_df is not None and not sim_df.empty:
                    # Clean + dedupe (keep the highest-minutes-ish estimate by taking max minutes proxy: pts_mean)
                    sim_df["team"] = sim_df["team"].astype(str).str.upper().str.strip()
                    sim_df["name_key"] = sim_df["name_key"].astype(str).str.upper().str.strip()
                    if "player_id" in sim_df.columns:
                        sim_df["player_id"] = pd.to_numeric(sim_df["player_id"], errors="coerce").astype("Int64")
                    for c in [
                        "mean_pts","mean_reb","mean_ast","mean_threes","mean_pra","mean_stl","mean_blk","mean_tov",
                        "sd_pts","sd_reb","sd_ast","sd_threes","sd_pra","sd_stl","sd_blk","sd_tov",
                    ]:
                        if c in sim_df.columns:
                            sim_df[c] = pd.to_numeric(sim_df[c], errors="coerce")

                    # Prefer stable player_id for matching when available.
                    sim_has_pid = "player_id" in sim_df.columns and sim_df["player_id"].notna().any()
                    if sim_has_pid:
                        sim_df_pid = sim_df[sim_df["player_id"].notna()].copy()
                        sim_df_pid = sim_df_pid.drop_duplicates(subset=["player_id"], keep="last")
                    else:
                        sim_df_pid = pd.DataFrame()
                    sim_df_name = sim_df.drop_duplicates(subset=["team","name_key"], keep="last")

                    # Merge into preds by team + normalized name.
                    preds = preds.copy()
                    if "team" in preds.columns:
                        preds["team"] = preds["team"].astype(str).str.upper().str.strip()
                    if "player_name" not in preds.columns:
                        preds["player_name"] = None
                    preds["_name_key"] = preds["player_name"].astype(str).map(_norm_name_key)

                    # Coverage tracking
                    merge_report: dict[str, Any] = {
                        "date": str(date_str),
                        "smart_sim_files": int(len(sim_files)),
                        "smart_sim_players": int(len(sim_df_name)),
                        "matched_by": {"player_id": 0, "team_name": 0},
                        "pred_rows": int(len(preds)),
                    }

                    # First merge on player_id when present.
                    merged = preds
                    if ("player_id" in merged.columns) and (not sim_df_pid.empty):
                        merged["player_id"] = pd.to_numeric(merged["player_id"], errors="coerce").astype("Int64")
                        merged = merged.merge(sim_df_pid, on=["player_id"], how="left", suffixes=("", "_sim"))
                        merge_report["matched_by"]["player_id"] = int(merged["mean_pts_sim"].notna().sum()) if "mean_pts_sim" in merged.columns else int(merged.filter(like="_sim").notna().any(axis=1).sum())
                    else:
                        merged = merged.copy()

                    # Fill remaining via team+name_key match.
                    need_name_match = pd.Series([True] * len(merged), index=merged.index)
                    if merged.filter(like="_sim").shape[1] > 0:
                        need_name_match = ~merged.filter(like="_sim").notna().any(axis=1)
                    if bool(need_name_match.any()):
                        sub_idx = merged.index[need_name_match]
                        m2 = merged.loc[need_name_match].merge(sim_df_name, left_on=["team", "_name_key"], right_on=["team", "name_key"], how="left", suffixes=("", "_sim2"))
                        m2.index = sub_idx
                        # Copy sim2 into sim columns where missing
                        for col in [
                            "mean_pts","mean_reb","mean_ast","mean_threes","mean_pra","mean_stl","mean_blk","mean_tov",
                            "sd_pts","sd_reb","sd_ast","sd_threes","sd_pra","sd_stl","sd_blk","sd_tov",
                        ]:
                            c2 = f"{col}_sim2"
                            c1 = f"{col}_sim"
                            if c2 in m2.columns:
                                if c1 not in m2.columns:
                                    m2[c1] = np.nan
                                m2[c1] = m2[c2].where(m2[c2].notna(), m2[c1])

                        # Also fix identifiers/metadata from SmartSim when name-matching.
                        for base_col in ("player_id", "player_name", "opponent", "home"):
                            c2 = f"{base_col}_sim2"
                            if c2 in m2.columns:
                                if base_col not in m2.columns:
                                    m2[base_col] = np.nan
                                m2[base_col] = m2[c2].where(m2[c2].notna(), m2[base_col])
                        m2 = m2.drop(columns=[c for c in m2.columns if c.endswith("_sim2")], errors="ignore")

                        # Assign back in a dtype-safe way.
                        # Avoid setting an entire mixed-dtype row-block from a NumPy array (can force object dtype
                        # and trigger pandas FutureWarning: "Setting an item of incompatible dtype").
                        cols_to_update: list[str] = []
                        cols_to_update.extend([c for c in merged.columns if c.endswith("_sim") and c in m2.columns])
                        cols_to_update.extend([c for c in ("player_id", "player_name", "opponent", "home") if (c in merged.columns and c in m2.columns)])
                        cols_to_update = list(dict.fromkeys(cols_to_update))

                        if cols_to_update:
                            m2_cast = m2[cols_to_update].copy()
                            for c in cols_to_update:
                                if pd.api.types.is_integer_dtype(merged[c]) or str(merged[c].dtype) == "Int64":
                                    m2_cast[c] = pd.to_numeric(m2_cast[c], errors="coerce").astype("Int64")
                                elif pd.api.types.is_numeric_dtype(merged[c]):
                                    m2_cast[c] = pd.to_numeric(m2_cast[c], errors="coerce")
                                elif str(merged[c].dtype) in {"bool", "boolean"}:
                                    # Keep <NA> support for optional booleans
                                    m2_cast[c] = m2_cast[c].astype("boolean")
                                else:
                                    m2_cast[c] = m2_cast[c].astype(object)
                            merged.loc[sub_idx, cols_to_update] = m2_cast

                    merge_report["matched_by"]["team_name"] = int(merged.filter(like="_sim").notna().any(axis=1).sum()) - int(merge_report["matched_by"]["player_id"])

                    # Replace mean_* columns when sim is available, leaving base pred_* untouched.
                    for col in [
                        "mean_pts","mean_reb","mean_ast","mean_threes","mean_pra","mean_stl","mean_blk","mean_tov",
                        "sd_pts","sd_reb","sd_ast","sd_threes","sd_pra","sd_stl","sd_blk","sd_tov",
                    ]:
                        sim_col = f"{col}_sim"
                        if sim_col in merged.columns:
                            if col not in merged.columns:
                                merged[col] = np.nan
                            merged[col] = merged[sim_col].where(merged[sim_col].notna(), merged[col])

                    preds = merged.drop(columns=[c for c in merged.columns if c.endswith("_sim") or c in {"_name_key", "name_key"}], errors="ignore")

                    # Expand coverage: append SmartSim-only players not present in preds.
                    # This prevents stale/missing feature rows from dropping real participants on completed slates.
                    try:
                        # Availability map (prefer local league_status file to avoid any network calls).
                        ls_playing_by_pid: dict[int, bool | None] = {}
                        ls_team_on_slate_by_pid: dict[int, bool | None] = {}
                        ls_injury_by_pid: dict[int, str] = {}
                        try:
                            ls_path = paths.data_processed / f"league_status_{date_str}.csv"
                            lsdf = None
                            if ls_path.exists():
                                lsdf = pd.read_csv(ls_path)
                            else:
                                lsdf = build_league_status(date_str)
                            if isinstance(lsdf, pd.DataFrame) and (not lsdf.empty) and ("player_id" in lsdf.columns):
                                tmp_ls = lsdf.copy()
                                tmp_ls["player_id"] = pd.to_numeric(tmp_ls["player_id"], errors="coerce")
                                tmp_ls = tmp_ls.dropna(subset=["player_id"]).copy()
                                if not tmp_ls.empty:
                                    tmp_ls["_pid"] = tmp_ls["player_id"].astype(int)
                                    if "playing_today" in tmp_ls.columns:
                                        for pid, v in zip(tmp_ls["_pid"].tolist(), tmp_ls["playing_today"].tolist()):
                                            ls_playing_by_pid[int(pid)] = (bool(v) if (v is not None and not pd.isna(v)) else None)
                                    if "team_on_slate" in tmp_ls.columns:
                                        for pid, v in zip(tmp_ls["_pid"].tolist(), tmp_ls["team_on_slate"].tolist()):
                                            ls_team_on_slate_by_pid[int(pid)] = (bool(v) if (v is not None and not pd.isna(v)) else None)
                                    if "injury_status" in tmp_ls.columns:
                                        for pid, v in zip(tmp_ls["_pid"].tolist(), tmp_ls["injury_status"].tolist()):
                                            ls_injury_by_pid[int(pid)] = str(v or "")
                        except Exception:
                            ls_playing_by_pid = {}
                            ls_team_on_slate_by_pid = {}
                            ls_injury_by_pid = {}

                        pred_pid: set[int] = set()
                        if "player_id" in preds.columns:
                            pred_pid = set(pd.to_numeric(preds["player_id"], errors="coerce").dropna().astype(int).tolist())
                        pred_key_to_pids: dict[tuple[str, str], set[int | None]] = {}
                        if ("team" in preds.columns) and ("player_name" in preds.columns):
                            _t = preds["team"].astype(str).str.upper().str.strip()
                            _nk = preds["player_name"].astype(str).map(_norm_name_key)
                            _pid = pd.to_numeric(preds.get("player_id"), errors="coerce") if "player_id" in preds.columns else pd.Series([np.nan] * len(preds), index=preds.index)
                            for tt, nn, pp in zip(_t.tolist(), _nk.tolist(), _pid.tolist()):
                                key = (str(tt).upper().strip(), str(nn).upper().strip())
                                if key not in pred_key_to_pids:
                                    pred_key_to_pids[key] = set()
                                try:
                                    pred_key_to_pids[key].add(int(pp) if pp is not None and (not pd.isna(pp)) else None)
                                except Exception:
                                    pred_key_to_pids[key].add(None)

                        sim_add = sim_df_name.copy()
                        sim_add["team"] = sim_add["team"].astype(str).str.upper().str.strip()
                        sim_add["name_key"] = sim_add["name_key"].astype(str).str.upper().str.strip()
                        if "player_id" in sim_add.columns:
                            sim_add["player_id"] = pd.to_numeric(sim_add["player_id"], errors="coerce")

                        def _missing(row) -> bool:
                            try:
                                pid = row.get("player_id")
                                if pid is not None and (not pd.isna(pid)):
                                    if int(pid) in pred_pid:
                                        return False
                                    # Never re-add players who are explicitly NOT playing today.
                                    pt = ls_playing_by_pid.get(int(pid))
                                    if pt is False:
                                        return False
                                key = (str(row.get("team") or "").upper().strip(), str(row.get("name_key") or "").upper().strip())
                                if key in pred_key_to_pids:
                                    # Consider it present only if the matching row has the same player_id.
                                    # If a name-match exists but with a different/missing id, append the SmartSim row.
                                    if pid is not None and (not pd.isna(pid)) and int(pid) in pred_key_to_pids[key]:
                                        return False
                            except Exception:
                                pass
                            return True

                        sim_add = sim_add[sim_add.apply(_missing, axis=1)].copy()
                        if not sim_add.empty:
                            add_rows = pd.DataFrame({
                                "team": sim_add.get("team"),
                                "player_id": sim_add.get("player_id"),
                                "player_name": sim_add.get("player_name"),
                                "mean_pts": sim_add.get("mean_pts"),
                                "mean_reb": sim_add.get("mean_reb"),
                                "mean_ast": sim_add.get("mean_ast"),
                                "mean_threes": sim_add.get("mean_threes"),
                                "mean_pra": sim_add.get("mean_pra"),
                                "mean_stl": sim_add.get("mean_stl"),
                                "mean_blk": sim_add.get("mean_blk"),
                                "mean_tov": sim_add.get("mean_tov"),
                                "sd_pts": sim_add.get("sd_pts"),
                                "sd_reb": sim_add.get("sd_reb"),
                                "sd_ast": sim_add.get("sd_ast"),
                                "sd_threes": sim_add.get("sd_threes"),
                                "sd_pra": sim_add.get("sd_pra"),
                                "sd_stl": sim_add.get("sd_stl"),
                                "sd_blk": sim_add.get("sd_blk"),
                                "sd_tov": sim_add.get("sd_tov"),
                            })
                            if "opponent" in preds.columns and "opponent" in sim_add.columns:
                                add_rows["opponent"] = sim_add.get("opponent")
                            if "home" in preds.columns and "home" in sim_add.columns:
                                add_rows["home"] = sim_add.get("home")
                            if "asof_date" in preds.columns:
                                add_rows["asof_date"] = date_str
                            elif "date" in preds.columns:
                                add_rows["date"] = date_str
                            # Fill availability fields from league_status when possible; otherwise leave as NaN.
                            if "player_id" in add_rows.columns:
                                try:
                                    pids = pd.to_numeric(add_rows["player_id"], errors="coerce")
                                    if "playing_today" in preds.columns:
                                        add_rows["playing_today"] = [ls_playing_by_pid.get(int(x)) if (x is not None and not pd.isna(x)) else None for x in pids.tolist()]
                                    if "team_on_slate" in preds.columns:
                                        add_rows["team_on_slate"] = [ls_team_on_slate_by_pid.get(int(x)) if (x is not None and not pd.isna(x)) else None for x in pids.tolist()]
                                    if "injury_status" in preds.columns:
                                        add_rows["injury_status"] = [ls_injury_by_pid.get(int(x), "") if (x is not None and not pd.isna(x)) else "" for x in pids.tolist()]
                                except Exception:
                                    pass

                            add_rows = add_rows.reindex(columns=preds.columns, fill_value=np.nan)
                            preds = pd.concat([preds, add_rows], ignore_index=True)
                            merge_report["added_smart_sim_only_rows"] = int(len(add_rows))
                    except Exception:
                        pass

                    # Reduce false positives in downstream coverage checks:
                    # if SmartSim ran successfully for this date, drop any rows that claim to belong to a
                    # SmartSim team but whose player_id is not present in SmartSim for that team.
                    try:
                        if ("player_id" in preds.columns) and ("team" in preds.columns) and (not sim_df.empty) and ("player_id" in sim_df.columns):
                            sim_team_ids: dict[str, set[int]] = {}
                            sdf = sim_df[["team", "player_id"]].copy()
                            sdf["team"] = sdf["team"].astype(str).str.upper().str.strip()
                            sdf["player_id"] = pd.to_numeric(sdf["player_id"], errors="coerce")
                            sdf = sdf.dropna(subset=["player_id"])
                            if not sdf.empty:
                                for t, g in sdf.groupby("team"):
                                    try:
                                        sim_team_ids[str(t)] = set(g["player_id"].astype(int).tolist())
                                    except Exception:
                                        continue

                            if sim_team_ids:
                                p = preds.copy()
                                p["_team"] = p["team"].astype(str).str.upper().str.strip()
                                p["_pid"] = pd.to_numeric(p["player_id"], errors="coerce")
                                on_sim_team = p["_team"].isin(set(sim_team_ids.keys())) & p["_pid"].notna()
                                if bool(on_sim_team.any()):
                                    protected = pd.Series(False, index=p.index)

                                    for flag_col in ("playing_today", "team_on_slate"):
                                        if flag_col not in p.columns:
                                            continue
                                        try:
                                            flag_vals = p[flag_col]
                                            flag_keep = flag_vals.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})
                                            try:
                                                flag_num = pd.to_numeric(flag_vals, errors="coerce")
                                                flag_keep = flag_keep | (flag_num.fillna(0.0).astype(float) > 0.5)
                                            except Exception:
                                                pass
                                            protected = protected | flag_keep.astype(bool)
                                        except Exception:
                                            pass

                                    try:
                                        box_p = paths.data_processed / f"boxscores_{date_str}.csv"
                                        if box_p.exists():
                                            bdf = pd.read_csv(box_p, usecols=lambda c: str(c).strip().upper() in {"PLAYER_ID", "MIN"})
                                            if ("PLAYER_ID" in bdf.columns) and ("MIN" in bdf.columns):
                                                bdf["PLAYER_ID"] = pd.to_numeric(bdf["PLAYER_ID"], errors="coerce")
                                                bdf["MIN"] = pd.to_numeric(bdf["MIN"], errors="coerce").fillna(0.0)
                                                played_ids = set(bdf.loc[(bdf["PLAYER_ID"].notna()) & (bdf["MIN"] > 0.0), "PLAYER_ID"].astype(int).tolist())
                                                if played_ids:
                                                    protected = protected | p["_pid"].isin(played_ids)
                                    except Exception:
                                        pass

                                    def _in_sim(row) -> bool:
                                        try:
                                            t = str(row.get("_team") or "")
                                            pid = row.get("_pid")
                                            if pid is None or pd.isna(pid):
                                                return True
                                            return int(pid) in sim_team_ids.get(t, set())
                                        except Exception:
                                            return True
                                    keep = pd.Series(True, index=p.index)
                                    keep.loc[on_sim_team] = (
                                        p.loc[on_sim_team].apply(_in_sim, axis=1).astype(bool)
                                        | protected.loc[on_sim_team].astype(bool)
                                    )
                                    dropped = int((~keep).sum())
                                    preds = preds.loc[keep].copy()
                                    merge_report["dropped_not_in_smartsim_team"] = dropped

                            # Do NOT force playing_today=True based on SmartSim team membership;
                            # playing_today is an availability signal that should come from league_status/injuries.

                            # Ensure asof_date is set for appended rows.
                            if "asof_date" in preds.columns:
                                try:
                                    preds = preds.copy()
                                    preds["asof_date"] = preds["asof_date"].where(preds["asof_date"].notna(), date_str)
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    # Ensure sd_* exist and are sane for downstream edges.
                    sd_defaults = {
                        "sd_pts": 7.5,
                        "sd_reb": 3.0,
                        "sd_ast": 2.5,
                        "sd_threes": 1.3,
                        "sd_pra": 9.0,
                        "sd_stl": 1.2,
                        "sd_blk": 1.3,
                        "sd_tov": 1.5,
                    }
                    for c, dflt in sd_defaults.items():
                        if c not in preds.columns:
                            preds[c] = np.nan
                        preds[c] = pd.to_numeric(preds[c], errors="coerce")
                        bad = preds[c].isna() | (preds[c] <= 0)
                        if bool(bad.any()):
                            preds.loc[bad, c] = float(dflt)
                        # light clamp
                        preds[c] = preds[c].clip(lower=0.25, upper=float(max(3.0 * dflt, 25.0)))

                    # Write merge report (best effort)
                    try:
                        merge_report["sd_coverage_ratio"] = float(
                            preds[[c for c in sd_defaults.keys() if c in preds.columns]].notna().all(axis=1).mean()
                        )
                    except Exception:
                        merge_report["sd_coverage_ratio"] = None
                    try:
                        rp = paths.data_processed / f"smart_sim_merge_report_{date_str}.json"
                        rp.write_text(_json.dumps(merge_report, indent=2), encoding="utf-8")
                    except Exception:
                        pass

                    console.print({"smart_sim_players": int(len(sim_df_name)), "smart_sim_files": int(len(sim_files))})
        except Exception as _e:
            console.print(f"SmartSim integration skipped due to error: {_e}", style="yellow")

    if calibrate:
        try:
            from .props_calibration import compute_biases, apply_biases, save_calibration
            biases = compute_biases(anchor_date=date_str, window_days=int(calib_window))
            preds = apply_biases(preds, biases)
            save_calibration(biases, anchor_date=date_str, window_days=int(calib_window))
            console.print({"calibration": biases})
        except Exception as _e:
            console.print(f"Calibration skipped due to error: {_e}", style="yellow")
    # Optional per-player calibration (rolling intercept per player/stat)
    if calibrate_player:
        try:
            from .props_calibration import compute_player_biases, apply_player_biases, save_player_calibration
            # Parse per-stat overrides if provided (format: 'stat:value,stat:value')
            def _parse_map(s: str | None, to_float: bool = True):
                if not s:
                    return None
                out: dict[str, float] | dict[str, int] = {}
                try:
                    parts = [p.strip() for p in s.split(',') if p.strip()]
                    for p in parts:
                        if ':' not in p:
                            continue
                        k, v = p.split(':', 1)
                        k = k.strip().lower()
                        if not k:
                            continue
                        out[k] = float(v) if to_float else int(float(v))
                    return out
                except Exception:
                    return None
            k_map = _parse_map(player_shrink_k_by_stat, to_float=True)
            n_map = _parse_map(player_min_pairs_by_stat, to_float=False)
            # If not provided explicitly, load from config file if available
            if k_map is None and n_map is None:
                cfg_k, cfg_n = _load_player_calib_overrides()
                if cfg_k or cfg_n:
                    k_map = cfg_k or k_map
                    n_map = cfg_n or n_map
            pb = compute_player_biases(
                anchor_date=date_str,
                window_days=int(player_calib_window),
                min_pairs_per_player=int(player_min_pairs),
                shrink_k=float(player_shrink_k),
                shrink_k_by_stat=k_map,  # type: ignore[arg-type]
                min_pairs_by_stat=n_map,  # type: ignore[arg-type]
            )
            if pb is not None and not pb.empty:
                preds = apply_player_biases(preds, pb)
                _saved = save_player_calibration(pb, anchor_date=date_str, window_days=int(player_calib_window))
                console.print({"player_calibration_rows": int(len(pb)), "saved": str(_saved)})
            else:
                console.print("Per-player calibration: no eligible players this window.", style="yellow")
            # Write metadata of calibration parameters used for this date
            try:
                import json as __json
                meta = {
                    "date": date_str,
                    "window_days": int(player_calib_window),
                    "global": {"K": float(player_shrink_k), "min_pairs": int(player_min_pairs)},
                    "by_stat": {
                        **({} if not k_map else {k: {"K": float(v)} for k, v in k_map.items()}),
                    },
                }
                # Merge min_pairs map into by_stat
                if n_map:
                    for sk, nv in n_map.items():
                        if sk not in meta["by_stat"]:
                            meta["by_stat"][sk] = {}
                        try:
                            meta["by_stat"][sk]["min_pairs"] = int(nv)
                        except Exception:
                            pass
                out_meta = paths.data_processed / f"props_player_calibration_used_{date_str}.json"
                with open(out_meta, "w", encoding="utf-8") as f:
                    __json.dump(meta, f, indent=2)
                console.print({"calibration_used": str(out_meta)})
            except Exception as __e:
                console.print(f"Failed to write calibration used meta: {__e}", style="yellow")
        except Exception as _e:
            console.print(f"Per-player calibration skipped due to error: {_e}", style="yellow")
    if not out_path:
        out_path = str(paths.data_processed / f"props_predictions_{date_str}.csv")

    # Final authority pass: force team (and optionally opponent/home) to match the pipeline artifacts.
    # This is especially important for SmartSim-appended rows that may carry stale team metadata.
    try:
        ls_path_final = paths.data_processed / f"league_status_{date_str}.csv"
        if ls_path_final.exists() and ("player_id" in preds.columns):
            lsdf = pd.read_csv(ls_path_final)
            if isinstance(lsdf, pd.DataFrame) and (not lsdf.empty) and {"player_id", "team"}.issubset(set(lsdf.columns)):
                tmp_ls = lsdf[["player_id", "team"]].copy()
                tmp_ls["player_id"] = pd.to_numeric(tmp_ls["player_id"], errors="coerce")
                tmp_ls = tmp_ls.dropna(subset=["player_id"]).copy()
                tmp_ls["_pid"] = tmp_ls["player_id"].astype(int)
                tmp_ls["team"] = tmp_ls["team"].astype(str).str.upper().str.strip()
                pid_to_team = dict(zip(tmp_ls["_pid"].tolist(), tmp_ls["team"].tolist()))

                pids = pd.to_numeric(preds["player_id"], errors="coerce")
                pid_int = pids.where(pids.notna(), pd.NA)
                pid_int = pid_int.astype("Int64")
                cur_team = preds.get("team")
                if cur_team is None:
                    preds["team"] = [pid_to_team.get(int(x)) if (x is not None and not pd.isna(x)) else "" for x in pid_int.tolist()]
                else:
                    cur_team_s = cur_team.astype(str).str.upper().str.strip()
                    new_team = []
                    for pid_val, existing in zip(pid_int.tolist(), cur_team_s.tolist()):
                        if pid_val is not None and (not pd.isna(pid_val)):
                            mapped = pid_to_team.get(int(pid_val))
                            if mapped:
                                new_team.append(mapped)
                                continue
                        new_team.append(existing)
                    preds["team"] = new_team

                # If slate-only, prefer standardized game_odds for opponent/home when available.
                try:
                    if slate_only and ("team" in preds.columns):
                        go_path = paths.data_processed / f"game_odds_{date_str}.csv"
                        if go_path.exists():
                            from .teams import to_tricode as _to_tri2

                            go = pd.read_csv(go_path)
                            if isinstance(go, pd.DataFrame) and (not go.empty):
                                home_col = "home_team" if "home_team" in go.columns else None
                                away_col = "visitor_team" if "visitor_team" in go.columns else ("away_team" if "away_team" in go.columns else None)
                                if home_col and away_col:
                                    opp_map: dict[str, str] = {}
                                    home_map: dict[str, bool] = {}
                                    for _, r in go.iterrows():
                                        h = _to_tri2(str(r.get(home_col) or ""))
                                        a = _to_tri2(str(r.get(away_col) or ""))
                                        if h and a:
                                            opp_map[h] = a
                                            home_map[h] = True
                                            opp_map[a] = h
                                            home_map[a] = False
                                    if opp_map:
                                        t = preds["team"].astype(str).str.upper().str.strip()
                                        if "opponent" in preds.columns:
                                            preds["opponent"] = [opp_map.get(tt, preds.loc[i, "opponent"]) for i, tt in enumerate(t.tolist())]
                                        if "home" in preds.columns:
                                            preds["home"] = [home_map.get(tt, preds.loc[i, "home"]) for i, tt in enumerate(t.tolist())]
                except Exception:
                    pass
    except Exception:
        pass

    # Guardrails: these are count stats; avoid negative predictions/SDs after calibration.
    try:
        preds = preds.copy()
        for col in (
            "pred_pts",
            "pred_reb",
            "pred_ast",
            "pred_threes",
            "pred_stl",
            "pred_blk",
            "pred_tov",
        ):
            if col in preds.columns:
                preds[col] = pd.to_numeric(preds[col], errors="coerce").clip(lower=0.0)

        # PRA should be an exact derived stat in outputs.
        if all(c in preds.columns for c in ("pred_pts", "pred_reb", "pred_ast")):
            preds["pred_pra"] = (
                pd.to_numeric(preds["pred_pts"], errors="coerce").fillna(0.0)
                + pd.to_numeric(preds["pred_reb"], errors="coerce").fillna(0.0)
                + pd.to_numeric(preds["pred_ast"], errors="coerce").fillna(0.0)
            )
            preds["pred_pra"] = pd.to_numeric(preds["pred_pra"], errors="coerce").clip(lower=0.0)

        for col in (
            "sd_pts",
            "sd_reb",
            "sd_ast",
            "sd_threes",
            "sd_pra",
            "sd_stl",
            "sd_blk",
            "sd_tov",
        ):
            if col in preds.columns:
                preds[col] = pd.to_numeric(preds[col], errors="coerce").clip(lower=0.0)

        # De-dupe (same player/team/date) rows.
        if all(c in preds.columns for c in ("asof_date", "team", "player_id")):
            preds["asof_date"] = preds["asof_date"].where(preds["asof_date"].notna(), date_str)
            preds["team"] = preds["team"].astype(str).str.upper().str.strip()
            preds["player_id"] = pd.to_numeric(preds["player_id"], errors="coerce")
            # Rows without a player_id are not reliably joinable/evaluable; drop for determinism.
            preds = preds[preds["player_id"].notna()].copy()
            # Prefer rows with more complete SmartSim-derived distribution columns.
            score_cols = [c for c in ("sd_pts", "sd_reb", "sd_ast", "sd_pra", "pred_pts", "pred_reb", "pred_ast", "pred_pra") if c in preds.columns]
            preds["_row_score"] = preds[score_cols].notna().sum(axis=1) if score_cols else 0
            preds = preds.sort_values(["asof_date", "team", "player_id", "_row_score"]).drop_duplicates(
                subset=["asof_date", "team", "player_id"], keep="last"
            )
            preds = preds.drop(columns=["_row_score"], errors="ignore")
    except Exception:
        pass

    if preds is None or preds.empty:
        raise click.ClickException(f"predict-props produced no rows for {date_str}")

    preds.to_csv(out_path, index=False)
    console.print(f"Saved props predictions to {out_path} (rows={len(preds)}; calibrated={calibrate})")


@cli.command("evaluate-props")
@click.option("--start", type=str, required=True, help="Start date YYYY-MM-DD")
@click.option("--end", type=str, required=True, help="End date YYYY-MM-DD")
@click.option("--slate-only/--no-slate-only", default=True, show_default=True, help="Filter predictions to scoreboard slate for each date")
def evaluate_props_cmd(start: str, end: str, slate_only: bool):
    """Evaluate props predictions vs nbastatR actuals over a date range.

    Builds features (no leakage) per date, predicts, joins to actuals (if present), and computes RMSE/MAE/R2 per stat.
    """
    console.rule("Evaluate Props Models")
    # Parse date inputs early for use in all branches
    import datetime as _dt
    try:
        start_d = _dt.datetime.strptime(start, "%Y-%m-%d").date()
        end_d = _dt.datetime.strptime(end, "%Y-%m-%d").date()
    except Exception:
        console.print("Invalid --start/--end date. Use YYYY-MM-DD.", style="red"); return
    # Load actuals store
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # type: ignore
    import numpy as np
    act_p = paths.data_processed / "props_actuals.parquet"
    actuals = None
    if act_p.exists():
        try:
            actuals = pd.read_parquet(act_p)
        except Exception:
            actuals = None
    if actuals is None:
        # Fallback: combine any daily props_actuals_*.csv files in processed
        try:
            daily = list(paths.data_processed.glob("props_actuals_*.csv"))
            if daily:
                frames = []
                for p in daily:
                    try:
                        frames.append(pd.read_csv(p))
                    except Exception:
                        pass
                if frames:
                    actuals = pd.concat(frames, ignore_index=True)
        except Exception:
            pass
    if actuals is None or actuals.empty:
        # Try per-day nbastatR fetch
        console.print("No consolidated actuals found; attempting to fetch per-day via nbastatR...", style="yellow")
        tmp_frames = []
        try:
            for d in pd.date_range(start_d, end_d, freq="D").date:
                try:
                    df_day = fetch_prop_actuals_via_nbastatr(date=str(d))
                    if df_day is not None and not df_day.empty:
                        tmp_frames.append(df_day)
                except Exception:
                    continue
        except Exception:
            pass
        if tmp_frames:
            actuals = pd.concat(tmp_frames, ignore_index=True)
        else:
            # Final fallback: derive actuals from player_logs (Python only)
            console.print("Falling back to player_logs for actuals.", style="yellow")
            try:
                p = paths.data_processed / "player_logs.parquet"; c = paths.data_processed / "player_logs.csv"
                logs = pd.read_parquet(p) if p.exists() else (pd.read_csv(c) if c.exists() else None)
                if logs is None or logs.empty:
                    console.print("player_logs not found. Run fetch-player-logs.", style="red"); return
                # Identify columns
                def pick(df, cands):
                    m = {x.lower(): x for x in df.columns}
                    for k in cands:
                        if k.lower() in m:
                            return m[k.lower()]
                    return None
                dcol = pick(logs, ["GAME_DATE", "GAME_DATE_EST", "dateGame", "GAME_DATE_PT"]) 
                pid = pick(logs, ["PLAYER_ID", "player_id", "idPlayer"]) 
                pts = pick(logs, ["PTS","pts"]) ; reb = pick(logs, ["REB","reb","TREB","treb"]) ; ast = pick(logs, ["AST","ast"]) ; fg3m = pick(logs, ["FG3M","fg3m"]) 
                stl = pick(logs, ["STL", "stl", "steals"]) ; blk = pick(logs, ["BLK", "blk", "blocks"]) ; tov = pick(logs, ["TOV", "TO", "tov", "to", "turnovers"])
                if not all([dcol, pid, pts, reb, ast, fg3m]):
                    console.print("player_logs missing required columns for actuals.", style="red"); return
                logs[dcol] = pd.to_datetime(logs[dcol]).dt.date
                mask = (logs[dcol] >= start_d) & (logs[dcol] <= end_d)
                keep_cols = [dcol, pid, pts, reb, ast, fg3m] + [c for c in [stl, blk, tov] if c]
                part = logs.loc[mask, keep_cols].copy()
                rename_map = {dcol: "date", pid: "player_id", pts: "pts", reb: "reb", ast: "ast", fg3m: "threes"}
                if stl:
                    rename_map[stl] = "stl"
                if blk:
                    rename_map[blk] = "blk"
                if tov:
                    rename_map[tov] = "tov"
                part.rename(columns=rename_map, inplace=True)
                for ccc in ["pts","reb","ast","threes","stl","blk","tov"]:
                    part[ccc] = pd.to_numeric(part[ccc], errors="coerce")
                part["pra"] = part[["pts","reb","ast"]].sum(axis=1, skipna=True)
                actuals = part
            except Exception as _e:
                console.print(f"Failed to derive actuals from player_logs: {_e}", style="red"); return
    if actuals is not None and not actuals.empty:
        actuals["date"] = pd.to_datetime(actuals["date"]).dt.date
    dates = pd.date_range(start_d, end_d, freq="D").date
    rows = []
    for d in dates:
        try:
            feats = build_features_for_date(d)
            if slate_only:
                try:
                    sb = scoreboardv2.ScoreboardV2(game_date=str(d), day_offset=0, timeout=30)
                    nd = sb.get_normalized_dict(); gh = pd.DataFrame(nd.get("GameHeader", [])); ls = pd.DataFrame(nd.get("LineScore", []))
                    if not gh.empty and not ls.empty:
                        ls_cols = {c.upper(): c for c in ls.columns}
                        team_map = {}
                        if {"TEAM_ID","TEAM_ABBREVIATION"}.issubset(ls_cols.keys()):
                            for _, r in ls.iterrows():
                                try:
                                    team_map[int(r[ls_cols["TEAM_ID"]])] = str(r[ls_cols["TEAM_ABBREVIATION"]]).upper()
                                except Exception:
                                    pass
                        gh_cols = {c.upper(): c for c in gh.columns}
                        games = []
                        if {"HOME_TEAM_ID","VISITOR_TEAM_ID"}.issubset(gh_cols.keys()):
                            for _, g in gh.iterrows():
                                try:
                                    hid = int(g[gh_cols["HOME_TEAM_ID"]]); vid = int(g[gh_cols["VISITOR_TEAM_ID"]])
                                    h = team_map.get(hid); v = team_map.get(vid)
                                    if h and v:
                                        games.append({"team": h}); games.append({"team": v})
                                except Exception:
                                    pass
                        slate = pd.DataFrame(games)
                        if not slate.empty and "team" in feats.columns:
                            feats["team"] = feats["team"].astype(str).str.upper()
                            feats = feats.merge(slate, on="team", how="inner")
                except Exception:
                    pass
            from .props_train import predict_props  # Import here to avoid sklearn dependency
            preds = predict_props(feats)
            preds["date"] = d
            # join to actuals by (date, player_id)
            part_act = actuals[actuals["date"] == d].copy()
            merged = preds.merge(part_act, on=["date","player_id"], how="inner", suffixes=("","_act"))
            if not merged.empty:
                rows.append(merged)
        except Exception:
            continue
    if not rows:
        console.print("No overlapping predictions and actuals in the range.", style="yellow"); return
    df = pd.concat(rows, ignore_index=True)
    # Compute metrics
    metrics = []
    for target, pred_col in [("pts","pred_pts"),("reb","pred_reb"),("ast","pred_ast"),("threes","pred_threes"),("pra","pred_pra")]:
        if target in df.columns and pred_col in df.columns:
            y = pd.to_numeric(df[target], errors="coerce")
            p = pd.to_numeric(df[pred_col], errors="coerce")
            mask = y.notna() & p.notna()
            if mask.any():
                rmse = float(np.sqrt(mean_squared_error(y[mask], p[mask])))
                mae = float(mean_absolute_error(y[mask], p[mask]))
                r2 = float(r2_score(y[mask], p[mask]))
                metrics.append({"stat": target, "rmse": rmse, "mae": mae, "r2": r2, "n": int(mask.sum())})
    out = pd.DataFrame(metrics)
    console.print(out)
    out_path = paths.data_processed / f"props_eval_{start}_{end}.csv"
    out.to_csv(out_path, index=False)
    console.print(f"Saved props eval metrics to {out_path}")


@cli.command("train-props-npu")
@click.option("--alpha", type=float, default=1.0, show_default=True, help="Ridge regularization strength")
def train_props_npu_cmd(alpha: float):
    """Train props regression models and convert to ONNX for NPU acceleration."""
    console.rule("Train Props Models (NPU)")
    try:
        from .props_npu import train_props_models_npu
        train_props_models_npu(alpha=alpha)
        console.print("Saved props models, ONNX models, and feature columns for NPU.")
    except ImportError as e:
        console.print(f"NPU dependencies not available: {e}", style="red")
    except FileNotFoundError as e:
        console.print(str(e), style="red")
    except Exception as e:
        console.print(f"Failed to train NPU props models: {e}", style="red")


@cli.command("predict-props-npu")
@click.option("--date", "date_str", type=str, required=True, help="Prediction date YYYY-MM-DD (features built up to the day before)")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=False, help="Output CSV path (default props_predictions_npu_YYYY-MM-DD.csv)")
@click.option("--slate-only/--no-slate-only", default=True, show_default=True, help="Filter predictions to teams on the scoreboard slate and add opponent/home flags")
@click.option("--calibrate/--no-calibrate", default=True, show_default=True, help="Apply rolling bias calibration from recent recon vs predictions")
@click.option("--calib-window", type=int, default=7, show_default=True, help="Lookback days for calibration window (excludes today)")
def predict_props_npu_cmd(date_str: str, out_path: str | None, slate_only: bool, calibrate: bool, calib_window: int):
    """Predict player props using NPU-accelerated ONNX models for ultra-fast inference."""
    console.rule("Predict Props (NPU)")
    try:
        feats = build_features_for_date(date_str)
    except Exception as e:
        console.print(f"Failed to build features for {date_str}: {e}", style="red"); return
    
    # Optional slate filter using ScoreboardV2
    if slate_only:
        try:
            sb = scoreboardv2.ScoreboardV2(game_date=date_str, day_offset=0, timeout=30)
            nd = sb.get_normalized_dict()
            gh = pd.DataFrame(nd.get("GameHeader", []))
            ls = pd.DataFrame(nd.get("LineScore", []))
            if not gh.empty and not ls.empty:
                ls_cols = {c.upper(): c for c in ls.columns}
                if {"TEAM_ID","TEAM_ABBREVIATION"}.issubset(ls_cols.keys()):
                    team_map = {}
                    for _, r in ls.iterrows():
                        try:
                            team_map[int(r[ls_cols["TEAM_ID"]])] = str(r[ls_cols["TEAM_ABBREVIATION"]]).upper()
                        except Exception:
                            continue
                    gh_cols = {c.upper(): c for c in gh.columns}
                    if {"HOME_TEAM_ID","VISITOR_TEAM_ID"}.issubset(gh_cols.keys()):
                        games = []
                        for _, g in gh.iterrows():
                            try:
                                hid = int(g[gh_cols["HOME_TEAM_ID"]]); vid = int(g[gh_cols["VISITOR_TEAM_ID"]])
                                h = team_map.get(hid); v = team_map.get(vid)
                                if h and v:
                                    games.append({"team": h, "opponent": v, "home": True})
                                    games.append({"team": v, "opponent": h, "home": False})
                            except Exception:
                                continue
                        slate = pd.DataFrame(games)
                        if not slate.empty and "team" in feats.columns:
                            feats["team"] = feats["team"].astype(str).str.upper()
                            feats = feats.merge(slate, on="team", how="inner")
        except Exception:
            # If scoreboard fails, proceed without filtering
            pass
    
    try:
        from .props_npu import predict_props_npu
        preds = predict_props_npu(feats)
    except ImportError as e:
        console.print(f"NPU dependencies not available: {e}", style="red"); return
    except FileNotFoundError:
        console.print("NPU props models not found. Run train-props-npu first.", style="red"); return
    except Exception as e:
        console.print(f"Failed to predict props with NPU: {e}", style="red"); return
    
    # Optional light calibration (rolling intercept per stat)
    if calibrate:
        try:
            from .props_calibration import compute_biases, apply_biases, save_calibration
            biases = compute_biases(anchor_date=date_str, window_days=int(calib_window))
            preds = apply_biases(preds, biases)
            save_calibration(biases, anchor_date=date_str, window_days=int(calib_window))
            console.print({"calibration": biases})
        except Exception as _e:
            console.print(f"Calibration skipped due to error: {_e}", style="yellow")
    
    if not out_path:
        out_path = str(paths.data_processed / f"props_predictions_npu_{date_str}.csv")
    preds.to_csv(out_path, index=False)
    console.print(f"Saved NPU props predictions to {out_path} (rows={len(preds)}; calibrated={calibrate})")


@cli.command("evaluate-props-calibration-compare")
@click.option("--start", type=str, required=True, help="Start date YYYY-MM-DD")
@click.option("--end", type=str, required=True, help="End date YYYY-MM-DD")
@click.option("--slate-only/--no-slate-only", default=True, show_default=True, help="Restrict predictions to scoreboard slate")
@click.option("--calib-window", type=int, default=7, show_default=True, help="Global calibration window days")
@click.option("--player-calib-window", type=int, default=30, show_default=True, help="Per-player calibration window days")
@click.option("--player-min-pairs", type=int, default=6, show_default=True, help="Minimum pairs per player/stat")
@click.option("--player-shrink-k", type=int, default=8, show_default=True, help="Shrinkage K for per-player")
def evaluate_props_calibration_compare_cmd(start: str, end: str, slate_only: bool, calib_window: int,
                                           player_calib_window: int, player_min_pairs: int, player_shrink_k: float):
    """Compare metrics with global-only calibration vs global + per-player calibration over a date range.

    For each date, runs predict-props twice (saving to temp files), then joins to actuals and computes MAE/RMSE per stat.
    Outputs a summary CSV under data/processed.
    """
    console.rule("Props Calibration: Global vs Per-Player Compare")
    import sys as _sys, subprocess as _sp, datetime as _dt
    # Load actuals store: try parquet first; on failure, fall back to daily CSVs
    act_p = paths.data_processed / "props_actuals.parquet"
    actuals = None
    if act_p.exists():
        try:
            actuals = pd.read_parquet(act_p)
        except Exception:
            actuals = None
    if actuals is None or (hasattr(actuals, "empty") and actuals.empty):
        try:
            daily = sorted(paths.data_processed.glob("props_actuals_*.csv"))
            frames = []
            for pp in daily:
                try:
                    dfp = pd.read_csv(pp)
                    if dfp is not None and not dfp.empty:
                        frames.append(dfp)
                except Exception:
                    continue
            if frames:
                actuals = pd.concat(frames, ignore_index=True)
        except Exception:
            actuals = None
    if actuals is None or actuals.empty:
        console.print("No actuals available; run fetch-prop-actuals for the range first.", style="red"); return
    actuals["date"] = pd.to_datetime(actuals["date"]).dt.strftime("%Y-%m-%d")

    # Iterate dates
    try:
        start_d = _dt.datetime.strptime(start, "%Y-%m-%d").date()
        end_d = _dt.datetime.strptime(end, "%Y-%m-%d").date()
    except Exception:
        console.print("Invalid --start/--end date. Use YYYY-MM-DD.", style="red"); return
    stats = [
        ("pts","pred_pts"),
        ("reb","pred_reb"),
        ("ast","pred_ast"),
        ("threes","pred_threes"),
        ("pra","pred_pra"),
    ]
    rows = []
    for d in pd.date_range(start=start_d, end=end_d, freq="D").date:
        ds = str(d)
        try:
            # Temporary outputs for both variants
            out_g = paths.data_processed / f"_props_predictions_global_{ds}.csv"
            out_p = paths.data_processed / f"_props_predictions_player_{ds}.csv"
            # Build base CLI args
            base = [
                _sys.executable, "-m", "nba_betting.cli", "predict-props",
                "--date", ds,
                "--calibrate", "--calib-window", str(int(calib_window)),
                "--use-pure-onnx",
            ]
            if slate_only:
                base.append("--slate-only")
            # Run global-only
            args_g = base + ["--out", str(out_g), "--no-calibrate-player"]
            _sp.run(args_g, check=False)
            # Run global + per-player
            # Use tuned config by default: do not pass per-player K/min overrides here
            args_p = base + [
                "--out", str(out_p),
                "--calibrate-player", "--player-calib-window", str(int(player_calib_window)),
            ]
            _sp.run(args_p, check=False)
            if (not out_g.exists()) or (not out_p.exists()):
                continue
            g = pd.read_csv(out_g); p = pd.read_csv(out_p)
            # Filter to players with actuals that day
            act_d = actuals[actuals["date"] == ds].copy()
            if act_d.empty:
                continue
            # Join on player_id if present, else on name
            key_cols = ["player_id"] if ("player_id" in g.columns and "player_id" in act_d.columns) else ["player_name"]
            # Normalize key types to avoid dtype mismatches (e.g., object vs int64)
            if key_cols == ["player_id"]:
                try:
                    g["player_id"] = g["player_id"].astype(str)
                    p["player_id"] = p["player_id"].astype(str)
                    act_d["player_id"] = act_d["player_id"].astype(str)
                except Exception:
                    pass
            else:
                # Name-based fallback: normalize case/whitespace
                for _df in (g, p, act_d):
                    if "player_name" in _df.columns:
                        _df["player_name"] = _df["player_name"].astype(str).str.strip()
            mg = g.merge(act_d, on=key_cols, how="inner", suffixes=("","_act"))
            mp = p.merge(act_d, on=key_cols, how="inner", suffixes=("","_act"))
            # Lightweight diagnostics per date to aid debugging if needed
            try:
                console.print({
                    "date": ds,
                    "pred_global": int(len(g)),
                    "pred_player": int(len(p)),
                    "actuals": int(len(act_d)),
                    "joined_global": int(len(mg)),
                    "joined_player": int(len(mp)),
                })
            except Exception:
                pass
            if mg.empty or mp.empty:
                continue
            # Compute per-stat metrics
            for target, pred_col in stats:
                if (target in mg.columns) and (pred_col in mg.columns) and (pred_col in mp.columns):
                    y = pd.to_numeric(mg[target], errors="coerce"); pg = pd.to_numeric(mg[pred_col], errors="coerce"); pp = pd.to_numeric(mp[pred_col], errors="coerce")
                    mask = y.notna() & pg.notna() & pp.notna()
                    if mask.any():
                        import numpy as _np
                        yy = y[mask].to_numpy(dtype=float)
                        pgv = pg[mask].to_numpy(dtype=float)
                        ppv = pp[mask].to_numpy(dtype=float)
                        rmse_g = float(_np.sqrt(_np.mean((yy - pgv) ** 2)))
                        rmse_p = float(_np.sqrt(_np.mean((yy - ppv) ** 2)))
                        mae_g = float(_np.mean(_np.abs(yy - pgv)))
                        mae_p = float(_np.mean(_np.abs(yy - ppv)))
                        rows.append({
                            "date": ds, "stat": target, "n": int(mask.sum()),
                            "rmse_global": rmse_g, "rmse_player": rmse_p, "delta_rmse": rmse_p - rmse_g,
                            "mae_global": mae_g, "mae_player": mae_p, "delta_mae": mae_p - mae_g,
                        })
        except Exception:
            continue
    if not rows:
        console.print("No comparable rows produced; check actuals availability and predictions.", style="yellow"); return
    df = pd.DataFrame(rows)
    # Aggregate by stat across range
    agg = df.groupby("stat").agg(
        n=("n","sum"),
        rmse_global=("rmse_global","mean"), rmse_player=("rmse_player","mean"), delta_rmse=("delta_rmse","mean"),
        mae_global=("mae_global","mean"), mae_player=("mae_player","mean"), delta_mae=("delta_mae","mean"),
    ).reset_index()
    console.print("Per-stat averages across range:")
    console.print(agg)
    out_detail = paths.data_processed / f"props_eval_compare_daily_{start}_{end}.csv"
    out_summary = paths.data_processed / f"props_eval_compare_summary_{start}_{end}.csv"
    df.to_csv(out_detail, index=False)
    agg.to_csv(out_summary, index=False)
    console.print({"saved_detail": str(out_detail), "saved_summary": str(out_summary)})


@cli.command("tune-props-player-calibration")
@click.option("--start", type=str, required=False, help="Start date YYYY-MM-DD (range start)")
@click.option("--end", type=str, required=False, help="End date YYYY-MM-DD (range end)")
@click.option("--days", type=int, required=False, help="If set, tunes over [today-days, yesterday]")
@click.option("--slate-only/--no-slate-only", default=True, show_default=True, help="Limit to slate players")
@click.option("--k-grid", type=str, default="6,8,10,12", show_default=True, help="Comma list of K candidates")
@click.option("--min-grid", type=str, default="6,8,10", show_default=True, help="Comma list of min-pairs candidates")
@click.option("--criterion", type=click.Choice(["mae","rmse"], case_sensitive=False), default="mae", show_default=True, help="Selection criterion")
def tune_props_player_calibration_cmd(start: str | None, end: str | None, days: int | None, slate_only: bool, k_grid: str, min_grid: str, criterion: str):
    """Grid-search per-stat per-player shrinkage (K) and min_pairs over a trailing range.

    Writes data/processed/props_player_calibration_config.json with chosen params for predict-props to consume.
    """
    console.rule("Tune Per-Player Calibration (per stat)")
    import datetime as _dt, sys as _sys, subprocess as _sp, json as _json
    # Resolve date range
    if days and (not start and not end):
        today = _dt.date.today()
        end_d = today - _dt.timedelta(days=1)
        start_d = end_d - _dt.timedelta(days=int(days))
    else:
        if not start or not end:
            console.print("Provide --start and --end or --days", style="red"); return
        try:
            start_d = _dt.datetime.strptime(start, "%Y-%m-%d").date()
            end_d = _dt.datetime.strptime(end, "%Y-%m-%d").date()
        except Exception:
            console.print("Invalid --start/--end format (YYYY-MM-DD)", style="red"); return
    if end_d < start_d:
        console.print("--end must be >= --start", style="red"); return

    # Load actuals store: parquet preferred, CSV snapshots fallback
    act_p = paths.data_processed / "props_actuals.parquet"
    actuals = None
    if act_p.exists():
        try:
            actuals = pd.read_parquet(act_p)
        except Exception:
            actuals = None
    if actuals is None or (hasattr(actuals, "empty") and actuals.empty):
        try:
            daily = sorted(paths.data_processed.glob("props_actuals_*.csv"))
            frames = []
            for pp in daily:
                try:
                    dfp = pd.read_csv(pp)
                    if dfp is not None and not dfp.empty:
                        frames.append(dfp)
                except Exception:
                    continue
            if frames:
                actuals = pd.concat(frames, ignore_index=True)
        except Exception:
            actuals = None
    if actuals is None or actuals.empty:
        console.print("No actuals available; run fetch-prop-actuals first.", style="red"); return
    actuals["date"] = pd.to_datetime(actuals["date"]).dt.strftime("%Y-%m-%d")

    # Parse candidate grids
    try:
        K_vals = [float(x.strip()) for x in k_grid.split(',') if x.strip()]
        N_vals = [int(float(x.strip())) for x in min_grid.split(',') if x.strip()]
    except Exception:
        console.print("Invalid --k-grid or --min-grid", style="red"); return

    stats = [("pts","pred_pts"),("reb","pred_reb"),("ast","pred_ast"),("threes","pred_threes"),("pra","pred_pra")]
    best: dict[str, tuple[float,int,float,int]] = {}  # stat -> (K, min_pairs, score, n_days)

    # For each stat, grid-search across dates, aggregate average score
    for stat_name, pred_col in stats:
        per_day_scores: list[dict] = []
        d = start_d
        while d <= end_d:
            ds = d.strftime("%Y-%m-%d")
            # Ensure we have actuals that day; skip otherwise
            act_d = actuals[actuals["date"] == ds]
            if act_d is None or act_d.empty:
                d += _dt.timedelta(days=1); continue
            # Run predictions for each grid point (target stat override only)
            for K in K_vals:
                for N in N_vals:
                    out_p = paths.data_processed / f"_tune_{stat_name}_K{int(K)}_N{int(N)}_{ds}.csv"
                    # Predict with global calibration and per-player calibration enabled; override only target stat
                    args = [
                        _sys.executable, "-m", "nba_betting.cli", "predict-props",
                        "--date", ds,
                        "--out", str(out_p),
                        "--calibrate", "--calib-window", "7",
                        "--calibrate-player", "--player-calib-window", "30",
                        "--player-min-pairs", str(int(N)), "--player-shrink-k", str(int(K)),
                        "--player-shrink-k-by-stat", f"{stat_name}:{int(K)}",
                        "--player-min-pairs-by-stat", f"{stat_name}:{int(N)}",
                        "--use-pure-onnx",
                    ]
                    if slate_only:
                        args.append("--slate-only")
                    else:
                        args.append("--no-slate-only")
                    try:
                        _sp.run(args, check=False, capture_output=True)
                    except Exception:
                        continue
            # Evaluate each grid for this date
            for K in K_vals:
                for N in N_vals:
                    path = paths.data_processed / f"_tune_{stat_name}_K{int(K)}_N{int(N)}_{ds}.csv"
                    if not path.exists():
                        continue
                    try:
                        pr = pd.read_csv(path)
                        # Normalize join keys
                        key = "player_id" if ("player_id" in pr.columns and "player_id" in act_d.columns) else "player_name"
                        pr_j = pr.copy(); act_j = act_d.copy()
                        if key == "player_id":
                            pr_j[key] = pr_j[key].astype(str); act_j[key] = act_j[key].astype(str)
                        else:
                            pr_j[key] = pr_j[key].astype(str).str.strip(); act_j[key] = act_j[key].astype(str).str.strip()
                        m = pr_j.merge(act_j, on=[key], how="inner", suffixes=("","_act"))
                        if m.empty or (stat_name not in m.columns) or (pred_col not in m.columns):
                            continue
                        y = pd.to_numeric(m[stat_name], errors="coerce"); p = pd.to_numeric(m[pred_col], errors="coerce")
                        mask = y.notna() & p.notna()
                        if not mask.any():
                            continue
                        yy = y[mask].to_numpy(dtype=float); pp = p[mask].to_numpy(dtype=float)
                        if criterion.lower() == "mae":
                            score = float(np.mean(np.abs(yy - pp)))
                        else:
                            score = float(np.sqrt(np.mean((yy - pp) ** 2)))
                        per_day_scores.append({"date": ds, "K": float(K), "min_pairs": int(N), "score": score})
                    except Exception:
                        continue
            d += _dt.timedelta(days=1)
        if not per_day_scores:
            console.print({"stat": stat_name, "status": "no_scores"}, style="yellow")
            continue
        df = pd.DataFrame(per_day_scores)
        agg = df.groupby(["K","min_pairs"]).agg(n_days=("score","count"), score=("score","mean")).reset_index().sort_values("score")
        if agg.empty:
            console.print({"stat": stat_name, "status": "no_agg"}, style="yellow")
            continue
        K_best = float(agg.iloc[0]["K"]); N_best = int(agg.iloc[0]["min_pairs"]); S_best = float(agg.iloc[0]["score"]); n_days = int(agg.iloc[0]["n_days"])
        best[stat_name] = (K_best, N_best, S_best, n_days)
        console.print({"stat": stat_name, "best": {"K": K_best, "min_pairs": N_best, "score": S_best, "n_days": n_days}})

    if not best:
        console.print("Tuning produced no results", style="red"); return
    # Write config JSON
    cfg = {
        "updated_at": _dt.date.today().strftime("%Y-%m-%d"),
        "window_days": int((end_d - start_d).days) + 1,
        "criterion": criterion.lower(),
        "per_stat": {s: {"K": v[0], "min_pairs": v[1]} for s, v in best.items()},
    }
    out_cfg = paths.data_processed / "props_player_calibration_config.json"
    try:
        with open(out_cfg, "w", encoding="utf-8") as f:
            _json.dump(cfg, f, indent=2)
        console.print({"saved": str(out_cfg), "per_stat": cfg.get("per_stat")})
        # Append to history CSV for auditing/evolution tracking
        try:
            hist_rows = []
            for s, v in best.items():
                Kb, Nb, Sb, n_days = v
                hist_rows.append({
                    "updated_at": cfg.get("updated_at"),
                    "window_days": cfg.get("window_days"),
                    "criterion": cfg.get("criterion"),
                    "stat": s,
                    "K": Kb,
                    "min_pairs": Nb,
                    "score": Sb,
                    "n_days": n_days,
                })
            if hist_rows:
                out_hist = paths.data_processed / "props_player_calibration_history.csv"
                try:
                    ex = pd.read_csv(out_hist) if out_hist.exists() else pd.DataFrame()
                except Exception:
                    ex = pd.DataFrame()
                new_df = pd.DataFrame(hist_rows)
                ex = pd.concat([ex, new_df], ignore_index=True)
                ex.to_csv(out_hist, index=False)
                console.print({"history_appended": len(hist_rows), "path": str(out_hist)})
        except Exception as _e:
            console.print(f"Failed to append history: {_e}", style="yellow")
    except Exception as e:
        console.print(f"Failed to write config JSON: {e}", style="red")


@cli.command("benchmark-npu")
@click.option("--runs", type=int, default=100, show_default=True, help="Number of benchmark runs")
@click.option("--players", type=int, default=500, show_default=True, help="Number of players to simulate")
def benchmark_npu_cmd(runs: int, players: int):
    """Benchmark NPU vs CPU performance for props prediction."""
    console.rule("NPU Benchmark")
    try:
        from .props_npu import benchmark_npu_performance
        results = benchmark_npu_performance(num_runs=runs, num_players=players)
        console.print(results)
    except ImportError as e:
        console.print(f"NPU dependencies not available: {e}", style="red")
    except Exception as e:
        console.print(f"Benchmark failed: {e}", style="red")


@cli.command("train-games-npu")
@click.option("--retrain/--no-retrain", default=True, show_default=True, help="Retrain models with latest data before converting to ONNX")
def train_games_npu_cmd(retrain: bool):
    """Train game models (win probability, spread, totals) and convert to ONNX for NPU acceleration."""
    console.rule("Train Game Models (NPU)")
    try:
        from .games_npu import train_game_models_npu
        train_game_models_npu(retrain=retrain)
        console.print("Saved game models and ONNX models for NPU acceleration.")
    except ImportError as e:
        console.print(f"NPU dependencies not available: {e}", style="red")
    except FileNotFoundError as e:
        console.print(str(e), style="red")
    except Exception as e:
        console.print(f"Failed to train NPU game models: {e}", style="red")


@cli.command("train-games-enhanced-onnx")
def train_games_enhanced_onnx_cmd():
    """Train enhanced (45-feature) game + period models on all available data, then convert to ONNX.

    Outputs enhanced sklearn models (*.joblib with _enhanced suffix) and ONNX models (*.onnx with _enhanced suffix) under models/.
    """
    console.rule("Train Enhanced Models + Convert to ONNX")
    try:
        games_file = paths.data_raw / "games_nba_api.csv"
        if not games_file.exists():
            console.print(f"Games file not found: {games_file}. Run 'nba-betting fetch' first.", style="red"); return
        console.print(f"Loading games from {games_file}...")
        games = pd.read_csv(games_file)
        # Normalize common column names from NBA API output
        rename_map = {}
        if 'home_team_tri' in games.columns:
            rename_map.update({
                'home_team_tri': 'home_team',
                'visitor_team_tri': 'visitor_team',
            })
        if 'date_est' in games.columns:
            rename_map['date_est'] = 'date'
        if 'home_score' in games.columns:
            rename_map['home_score'] = 'home_pts'
        if 'visitor_score' in games.columns:
            rename_map['visitor_score'] = 'visitor_pts'
        if rename_map:
            games = games.rename(columns=rename_map)

        from .features_enhanced import build_features_enhanced
        console.print("Building enhanced features (45 features incl. injuries)...", style="cyan")
        df = build_features_enhanced(games, include_advanced_stats=True, include_injuries=True)

        from .train_enhanced import train_models_enhanced
        console.print("Training enhanced models (win/spread/total + halves/quarters)...", style="cyan")
        metrics = train_models_enhanced(df, use_enhanced_features=True)
        console.print({k: (None if v is None else (round(v,4) if isinstance(v, float) else v)) for k, v in metrics.items()})

        # Convert to ONNX
        console.print("Converting enhanced models to ONNX...", style="cyan")
        try:
            from convert_enhanced_to_onnx import convert_enhanced_models
            convert_enhanced_models()
        except ImportError as e:
            console.print(f"skl2onnx not installed; install to convert to ONNX: {e}", style="yellow")
            console.print("You can still use sklearn CPU models; NPU/ONNX requires conversion.", style="yellow")
        console.print("Enhanced training + ONNX conversion complete.")
    except Exception as e:
        console.print(f"Failed to train/convert enhanced models: {e}", style="red")

@cli.command("predict-games-npu")
@click.option("--date", "date_str", type=str, required=True, help="Prediction date YYYY-MM-DD")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=False, help="Output CSV path (default games_predictions_npu_YYYY-MM-DD.csv)")
@click.option("--periods/--no-periods", default=True, show_default=True, help="Include halves and quarters predictions")
@click.option("--calibrate-periods/--no-calibrate-periods", default=True, show_default=True, help="Calibrate halves/quarters to team shares and enforce sum constraints")
def predict_games_npu_cmd(date_str: str, out_path: str | None, periods: bool, calibrate_periods: bool):
    """Predict game outcomes using NPU-accelerated models for ultra-fast inference."""
    console.rule("Predict Games (NPU)")
    try:
        def _slate_from_frame(frame: pd.DataFrame | None, home_candidates: tuple[str, ...], away_candidates: tuple[str, ...]) -> pd.DataFrame | None:
            if frame is None or frame.empty:
                return None
            home_col = next((col for col in home_candidates if col in frame.columns), None)
            away_col = next((col for col in away_candidates if col in frame.columns), None)
            if not home_col or not away_col:
                return None
            slate_df = pd.DataFrame(
                {
                    "date": pd.to_datetime(date_str),
                    "home_team": frame[home_col].astype(str),
                    "visitor_team": frame[away_col].astype(str),
                    "home_pts": np.nan,
                    "visitor_pts": np.nan,
                }
            )
            slate_df = slate_df.dropna(subset=["home_team", "visitor_team"]).copy()
            if slate_df.empty:
                return None
            slate_df["home_team"] = slate_df["home_team"].astype(str).str.strip()
            slate_df["visitor_team"] = slate_df["visitor_team"].astype(str).str.strip()
            slate_df = slate_df[
                (slate_df["home_team"] != "")
                & (slate_df["visitor_team"] != "")
                & (slate_df["home_team"].str.lower() != "nan")
                & (slate_df["visitor_team"].str.lower() != "nan")
            ].drop_duplicates(subset=["home_team", "visitor_team"])
            if slate_df.empty:
                return None
            return slate_df.reset_index(drop=True)

        target_date = pd.to_datetime(date_str).date()
        predictions_candidates = [
            paths.data_processed / f"predictions_{date_str}.csv",
            paths.root / f"predictions_{date_str}.csv",
        ]
        debug_predictions_fallback = str(os.environ.get("NBA_DEBUG_PREDICT_GAMES_NPU_FALLBACK") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        def _load_predictions_output_fallback() -> tuple[pd.DataFrame | None, Path | None]:
            for pred_candidate in predictions_candidates:
                if debug_predictions_fallback:
                    console.print(
                        f"predict-games-npu debug: candidate={pred_candidate} exists={pred_candidate.exists()}",
                        style="blue",
                    )
                if not pred_candidate.exists():
                    continue
                try:
                    pred_frame = pd.read_csv(pred_candidate)
                except Exception:
                    if debug_predictions_fallback:
                        console.print(
                            f"predict-games-npu debug: failed to read {pred_candidate}",
                            style="blue",
                        )
                    continue
                if pred_frame is None or pred_frame.empty:
                    if debug_predictions_fallback:
                        console.print(
                            f"predict-games-npu debug: {pred_candidate} empty",
                            style="blue",
                        )
                    continue
                pred_frame = pred_frame.copy()
                if "date" in pred_frame.columns:
                    pred_frame["date"] = pd.to_datetime(pred_frame["date"], errors="coerce").dt.date
                    pred_frame = pred_frame[pred_frame["date"] == target_date]
                if pred_frame.empty:
                    if debug_predictions_fallback:
                        console.print(
                            f"predict-games-npu debug: {pred_candidate} had no rows for {target_date}",
                            style="blue",
                        )
                    continue
                if not {"home_team", "visitor_team"}.issubset(pred_frame.columns):
                    if debug_predictions_fallback:
                        console.print(
                            f"predict-games-npu debug: {pred_candidate} missing team columns {sorted(pred_frame.columns)}",
                            style="blue",
                        )
                    continue
                if "home_win_prob" not in pred_frame.columns and "win_prob" not in pred_frame.columns:
                    if debug_predictions_fallback:
                        console.print(
                            f"predict-games-npu debug: {pred_candidate} missing probability column {sorted(pred_frame.columns)}",
                            style="blue",
                        )
                    continue
                if debug_predictions_fallback:
                    console.print(
                        f"predict-games-npu debug: using {pred_candidate} rows={len(pred_frame)}",
                        style="blue",
                    )
                return pred_frame.reset_index(drop=True), pred_candidate
            return None, None

        preds = None
        feature_load_error = None
        features_df = pd.DataFrame()

        # Load features for the date (prefer parquet; fallback to CSV to avoid parquet engine dependency)
        features_path = paths.data_processed / "features.parquet"
        csv_fallback = paths.data_processed / "features.csv"
        if features_path.exists() or csv_fallback.exists():
            try:
                if features_path.exists():
                    features_df = pd.read_parquet(features_path)
                else:
                    raise FileNotFoundError
            except Exception:
                # Parquet engine likely missing; try CSV fallback
                if csv_fallback.exists():
                    features_df = pd.read_csv(csv_fallback)
                else:
                    feature_load_error = (
                        "Failed to read features.parquet (pyarrow/fastparquet missing) and CSV fallback not found."
                    )
        else:
            feature_load_error = "Features not found. Run build-features first."


        # Filter to the specific date if provided
        if 'date' in features_df.columns:
            features_df['date'] = pd.to_datetime(features_df['date']).dt.date
            features_df = features_df[features_df['date'] == target_date]
        
        if features_df.empty:
            preds, pred_source = _load_predictions_output_fallback()
            if preds is not None and pred_source is not None:
                console.print(
                    f"Using standard predictions fallback from {pred_source}",
                    style="yellow",
                )

        if features_df.empty and preds is None:
            # Fallbacks to build enhanced slate features even without parquet engine or prewritten odds:
            # 1) If standardized game odds CSV exists, use it to derive slate
            # 2) Else, try odds events CSV aliases (clean CI may have events without consensus lines)
            # 3) Else, fall back to processed season schedule to derive slate (no market lines, but teams/pairs)
            try:
                import json

                odds_path = paths.data_processed / f"game_odds_{date_str}.csv"
                odds_events_path = paths.data_processed / f"odds_events_{date_str}.csv"
                oddsapi_events_path = paths.data_processed / f"oddsapi_events_{date_str}.csv"
                raw_games_csv = paths.data_raw / "games_nba_api.csv"
                raw_games_parq = paths.data_raw / "games_nba_api.parquet"
                slate_df = None

                if odds_path.exists():
                    slate_df = _slate_from_frame(
                        pd.read_csv(odds_path),
                        ("home_team", "home"),
                        ("visitor_team", "away", "away_team"),
                    )

                if slate_df is None and odds_events_path.exists():
                    slate_df = _slate_from_frame(
                        pd.read_csv(odds_events_path),
                        ("home_team", "home"),
                        ("away_team", "visitor_team", "away"),
                    )

                if slate_df is None and oddsapi_events_path.exists():
                    slate_df = _slate_from_frame(
                        pd.read_csv(oddsapi_events_path),
                        ("home_team", "home"),
                        ("away_team", "visitor_team", "away"),
                    )

                if slate_df is None:
                    for pred_candidate in predictions_candidates:
                        if not pred_candidate.exists():
                            continue
                        slate_df = _slate_from_frame(
                            pd.read_csv(pred_candidate),
                            ("home_team", "home"),
                            ("visitor_team", "away_team", "away"),
                        )
                        if slate_df is not None:
                            break

                # Fallback to live schedule data, then local schedule artifacts, if odds are unavailable
                if slate_df is None:
                    try:
                        day = _load_schedule_day(date_str)
                        if not day.empty:
                            slate_df = _slate_from_frame(
                                day,
                                ("home_tricode", "home_team", "home_team_id"),
                                ("away_tricode", "visitor_team", "away_team", "away_team_id"),
                            )
                    except Exception:
                        pass

                raw_games = None
                if raw_games_parq.exists():
                    try:
                        raw_games = pd.read_parquet(raw_games_parq)
                    except Exception:
                        raw_games = None
                if raw_games is None and raw_games_csv.exists():
                    raw_games = pd.read_csv(raw_games_csv)

                if slate_df is not None and not slate_df.empty and raw_games is not None and not raw_games.empty:
                    # Ensure columns exist
                    for col in ["date","home_team","visitor_team","home_pts","visitor_pts"]:
                        if col not in raw_games.columns:
                            raw_games[col] = np.nan
                    # Normalize date type
                    raw_games["date"] = pd.to_datetime(raw_games["date"], errors="coerce")
                    games = pd.concat([raw_games, slate_df], ignore_index=True, sort=False)
                    # Build enhanced features (adds pace/ratings + injuries)
                    from .features_enhanced import build_features_enhanced
                    feats2 = build_features_enhanced(games, include_advanced_stats=True, include_injuries=True, season=2025)
                    # Filter to target date
                    feats2["date"] = pd.to_datetime(feats2["date"]).dt.date
                    target_date = pd.to_datetime(date_str).date()
                    features_df = feats2[feats2["date"] == target_date]

                if features_df.empty:
                    if feature_load_error:
                        raise click.ClickException(feature_load_error)
                    else:
                        raise click.ClickException(f"No games found for {date_str}")
            except click.ClickException:
                raise
            except Exception as e:
                raise click.ClickException(f"No games found for {date_str}: {e}")

        if preds is None:
            _ensure_game_models_available()

            from .games_npu import predict_games_npu
            preds = predict_games_npu(features_df, include_periods=periods, calibrate_periods=calibrate_periods)
    except ImportError as e:
        raise click.ClickException(f"NPU dependencies not available: {e}")
    except FileNotFoundError:
        raise click.ClickException("NPU game models not found. Run train-games-npu first.")
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Failed to predict games with NPU: {e}")

    if preds is None or preds.empty:
        raise click.ClickException(f"predict-games-npu produced no rows for {date_str}")
    
    if not out_path:
        out_path = str(paths.data_processed / f"games_predictions_npu_{date_str}.csv")
    preds.to_csv(out_path, index=False)
    console.print(f"Saved NPU game predictions to {out_path} (rows={len(preds)}; periods={periods}; calibrated={calibrate_periods})")


@cli.command("benchmark-games-npu")
@click.option("--runs", type=int, default=100, show_default=True, help="Number of benchmark runs")
@click.option("--games", type=int, default=100, show_default=True, help="Number of games to simulate")
def benchmark_games_npu_cmd(runs: int, games: int):
    """Benchmark NPU vs CPU performance for game predictions."""
    console.rule("Game NPU Benchmark")
    try:
        from .games_npu import benchmark_game_npu_performance
        results = benchmark_game_npu_performance(num_runs=runs, num_games=games)
        console.print(results)
    except ImportError as e:
        console.print(f"NPU dependencies not available: {e}", style="red")
    except Exception as e:
        console.print(f"Game benchmark failed: {e}", style="red")


@cli.command("props-edges")
@click.option("--date", "date_str", type=str, required=True, help="Slate date YYYY-MM-DD")
@click.option("--use-saved/--no-use-saved", default=True, show_default=True, help="Prefer odds in data/raw if present before fetching")
@click.option("--mode", type=click.Choice(["auto","historical","current"]), default="auto", show_default=True, help="If fetching via OddsAPI, whether to use historical snapshots, current event odds, or auto")
@click.option("--source", type=click.Choice(["auto","oddsapi","bovada"]), default="auto", show_default=True, help="Odds source for player props: oddsapi, bovada, or auto")
@click.option("--api-key", envvar="ODDS_API_KEY", type=str, required=False, help="OddsAPI key (or set env ODDS_API_KEY)")
@click.option("--sigma-pts", type=float, default=7.5, show_default=True)
@click.option("--sigma-reb", type=float, default=3.0, show_default=True)
@click.option("--sigma-ast", type=float, default=2.5, show_default=True)
@click.option("--sigma-threes", type=float, default=1.3, show_default=True)
@click.option("--sigma-pra", type=float, default=9.0, show_default=True)
@click.option("--slate-only/--no-slate-only", default=True, show_default=True, help="Filter to teams on the scoreboard slate")
# NOTE: keep default min-edge low so regular props (often small edges) aren't accidentally filtered out.
@click.option("--min-edge", type=float, default=0.0, show_default=True, help="Minimum model edge (probability diff)")
@click.option("--min-ev", type=float, default=0.0, show_default=True, help="Minimum EV per 1u")
@click.option("--top", type=int, default=1000, show_default=False, help="Limit to top N edges after filtering")
@click.option("--bookmakers", type=str, default=None, help="Comma-separated bookmaker keys to include (e.g., draftkings,fanduel,pinnacle)")
@click.option("--calibrate-sigma/--no-calibrate-sigma", default=False, show_default=True, help="Estimate sigma per stat from recent residuals")
@click.option("--calibrate-prob/--no-calibrate-prob", default=True, show_default=True, help="Apply saved probability calibration curve to model prop probabilities when available")
@click.option("--odds-path", type=click.Path(exists=False, dir_okay=False), required=False, help="Use a specific odds snapshot CSV/parquet path instead of loading or fetching odds")
@click.option("--attach-opening-snapshot/--no-attach-opening-snapshot", default=True, show_default=True, help="Attach opening-line snapshot data when available")
@click.option("--resolve-roster/--no-resolve-roster", default=True, show_default=True, help="Use processed roster files to backfill unmatched player ids")
@click.option("--predictions-csv", type=click.Path(exists=False, dir_okay=False), required=False, help="Use precomputed props_predictions_YYYY-MM-DD.csv from this path; defaults to data/processed")
@click.option("--file-only/--no-file-only", default=False, show_default=True, help="Do not run props models; require predictions CSV to exist")
def props_edges_cmd(date_str: str, use_saved: bool, mode: str, source: str, api_key: str | None, sigma_pts: float, sigma_reb: float, sigma_ast: float, sigma_threes: float, sigma_pra: float, slate_only: bool, min_edge: float, min_ev: float, top: int, bookmakers: str | None, calibrate_sigma: bool, calibrate_prob: bool, odds_path: str | None, attach_opening_snapshot: bool, resolve_roster: bool, predictions_csv: str | None, file_only: bool):
    """Compute player props edges (EV) by merging model predictions with OddsAPI lines for a date.

    Writes data/processed/props_edges_YYYY-MM-DD.csv
    """
    console.rule("Props Edges")
    try:
        pd.to_datetime(date_str)  # validate
    except Exception:
        raise click.ClickException("Invalid --date (YYYY-MM-DD)")
    if not api_key:
        api_key = _load_dotenv_key("ODDS_API_KEY")
    sigma = SigmaConfig(pts=sigma_pts, reb=sigma_reb, ast=sigma_ast, threes=sigma_threes, pra=sigma_pra)
    if calibrate_sigma:
        try:
            sigma = calibrate_sigma_for_date(date_str, window_days=60, min_rows=200, defaults=sigma)
            console.print({"sigma": sigma.__dict__})
        except Exception:
            pass
    try:
        edges = compute_props_edges(
            date=date_str,
            sigma=sigma,
            use_saved=use_saved,
            mode=mode,
            api_key=api_key,
            source=source,
            predictions_path=predictions_csv,
            from_file_only=file_only,
            calibrate_prob=calibrate_prob,
            odds_path=odds_path,
            attach_opening_snapshot=attach_opening_snapshot,
            resolve_roster=resolve_roster,
        )
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(f"Failed to compute edges: {e}")
    if edges is None or edges.empty:
        if file_only:
            raise click.ClickException(f"No edges computed for {date_str} (missing odds or predictions).")
        console.print("No edges computed (missing odds or predictions).", style="yellow"); return
    # Optional slate filter
    if slate_only:
        try:
            sb = scoreboardv2.ScoreboardV2(game_date=date_str, day_offset=0, timeout=30)
            nd = sb.get_normalized_dict(); ls = pd.DataFrame(nd.get("LineScore", []))
            teams = []
            if not ls.empty:
                c = {x.upper(): x for x in ls.columns}
                if "TEAM_ABBREVIATION" in c:
                    teams = list(ls[c["TEAM_ABBREVIATION"]].astype(str).str.upper().unique())
            if teams:
                edges["team"] = edges["team"].astype(str).str.upper()
                edges = edges[edges["team"].isin(teams)].copy()
        except Exception:
            pass
    # By default, keep all books returned for the configured region unless --bookmakers narrows it.
    bookmaker_keys = resolve_player_prop_bookmakers(bookmakers)
    edges = filter_player_prop_bookmakers_df(edges, bookmakers)
    # Thresholds and top-N
    edges = edges[(edges["edge"] >= min_edge) & (edges["ev"] >= min_ev)].copy()
    # Prefer sorting by EV within stat for downstream top-N selection
    if "ev" in edges.columns:
        edges.sort_values(["stat", "ev"], ascending=[True, False], inplace=True)
    else:
        edges.sort_values(["stat", "edge"], ascending=[True, False], inplace=True)
    if top and len(edges) > top:
        edges = edges.groupby("stat", group_keys=False).head(max(1, top // max(1, edges["stat"].nunique())))
    out = paths.data_processed / f"props_edges_{date_str}.csv"
    edges.to_csv(out, index=False)
    console.print({
        "rows": int(len(edges)),
        "output": str(out),
        "bookmakers": (list(bookmaker_keys) if bookmaker_keys else "all"),
    })


@cli.command("export-recommendations")
@click.option("--date", "date_str", type=str, required=True, help="Slate date YYYY-MM-DD")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=False, help="Output CSV path; defaults to data/processed/recommendations_YYYY-MM-DD.csv")
@click.option("--min-ml-ev", "min_ml_ev", type=float, default=0.01, show_default=True, help="Minimum ML EV (per $1 stake) required to emit an ML recommendation")
@click.option("--min-ml-edge", "min_ml_edge", type=float, default=0.015, show_default=True, help="Minimum ML probability edge vs market no-vig implied probability")
@click.option("--ml-blend", "ml_blend", type=float, default=0.25, show_default=True, help="Blend for ML win prob: w*model + (1-w)*market_no_vig")
@click.option("--max-abs-ml-odds", "max_abs_ml_odds", type=float, default=200.0, show_default=True, help="Skip ML recs when |odds| exceeds this (set <=0 to disable)")
@click.option(
    "--max-plus-odds",
    "max_plus_odds",
    type=float,
    default=125.0,
    show_default=True,
    help="Skip any recommendation priced above this positive American odds threshold (e.g. 125). Set <=0 to disable.",
)
def export_recommendations_cmd(
    date_str: str,
    out_path: str | None,
    min_ml_ev: float,
    min_ml_edge: float,
    ml_blend: float,
    max_abs_ml_odds: float,
    max_plus_odds: float,
):
    """Export game recommendations (ML/ATS/TOTAL) to CSV from predictions + odds."""
    import pandas as pd
    import math
    from .config import paths
    from .teams import to_tricode as _tri
    try:
        d = pd.to_datetime(date_str).date()
    except Exception:
        console.print("Invalid --date (YYYY-MM-DD)", style="red"); return
    # Prefer the calibrated daily predictions artifact. The daily update applies
    # totals calibration to predictions_<date>.csv, while games_predictions_npu_<date>.csv
    # may remain unadjusted. Falling back to the raw NPU file here can reintroduce
    # systematic total bias into the recommendation export.
    pred_npu = paths.data_processed / f"games_predictions_npu_{date_str}.csv"
    pred = paths.data_processed / f"predictions_{date_str}.csv"
    use_path = pred if pred.exists() else pred_npu
    if not use_path.exists():
        console.print(f"Predictions not found: {use_path}", style="red"); return
    df = pd.read_csv(use_path)
    # Optional: try merge standardized game_odds CSV
    odds_csv = paths.data_processed / f"game_odds_{date_str}.csv"
    if odds_csv.exists():
        try:
            o = pd.read_csv(odds_csv)
            if "date" in o.columns:
                o["date"] = pd.to_datetime(o["date"], errors="coerce").dt.date
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
            on = ["date","home_team","visitor_team"]
            if all(c in df.columns for c in on) and all(c in o.columns for c in on):
                df = df.merge(o, on=on, how="left", suffixes=("","_odds"))
        except Exception:
            pass
    # Build recs
    recs: list[dict] = []
    def _num(x):
        try:
            return float(x)
        except Exception:
            return None
    def _ev(prob, american):
        try:
            p = float(prob)
        except Exception:
            return None
        try:
            a = float(american)
        except Exception:
            return None
        if a > 0:
            return p * (a/100.0) - (1-p) * 1.0
        else:
            return p * (100.0/(-a)) - (1-p) * 1.0
    def _implied(american):
        try:
            a = float(american)
        except Exception:
            return None
        if a == 0:
            return None
        if a > 0:
            return 100.0 / (a + 100.0)
        return (-a) / ((-a) + 100.0)

    def _no_vig_probs(home_ml: float | None, away_ml: float | None) -> tuple[float | None, float | None]:
        try:
            hm = _num(home_ml)
            am = _num(away_ml)
            if hm is None or am is None:
                return None, None
            ph = _implied(hm)
            pa = _implied(am)
            if ph is None or pa is None:
                return None, None
            s = float(ph) + float(pa)
            if s <= 0:
                return None, None
            return float(ph) / s, float(pa) / s
        except Exception:
            return None, None

    def _val_or(x, default: float) -> float:
        try:
            v = float(x)
            if pd.isna(v):
                return float(default)
            return float(v)
        except Exception:
            return float(default)

    def _phi(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _p_home_cover_from_margin(pred_margin: float, home_spread: float) -> float | None:
        try:
            # Use the model margin directly as the mean of a normal margin distribution.
            # This keeps ATS picks directionally consistent with the underlying spread/margin
            # and avoids over-calibration producing sign-flips.
            mu_ats = float(pred_margin)
            # home covers if margin > -home_spread
            threshold = -float(home_spread)
            # Typical NBA margin stdev is closer to ~11-13; use 12 as a stable default.
            sd = 12.0
            z = (threshold - mu_ats) / max(1e-6, float(sd))
            p = 1.0 - _phi(z)
            return float(min(1.0 - 1e-6, max(1e-6, p)))
        except Exception:
            return None

    def _p_total_over_from_pred_total(pred_total: float, market_total: float) -> float | None:
        try:
            mu = float(pred_total)
            threshold = float(market_total)
            # Totals have higher variance than margins; use a stable default.
            sd = 16.0
            z = (threshold - mu) / max(1e-6, float(sd))
            p = 1.0 - _phi(z)
            return float(min(1.0 - 1e-6, max(1e-6, p)))
        except Exception:
            return None
    def _tier(market: str, ev: float | None, edge: float | None) -> str:
        try:
            m = (market or '').upper()
            if m == 'ML' and ev is not None:
                if ev >= 0.04:
                    return 'High'
                if ev >= 0.02:
                    return 'Medium'
                return 'Low'
            if m == 'ATS' and edge is not None:
                v = abs(edge)
                if v >= 3.0:
                    return 'High'
                if v >= 1.5:
                    return 'Medium'
                return 'Low'
            if m == 'TOTAL' and edge is not None:
                v = abs(edge)
                if v >= 4.0:
                    return 'High'
                if v >= 2.0:
                    return 'Medium'
                return 'Low'
        except Exception:
            pass
        return 'Low'
    def _pick_num(row: "pd.Series", keys: list[str]) -> float | None:
        for k in keys:
            if k in row.index:
                v = _num(row.get(k))
                if v is None:
                    continue
                # treat NaN as missing
                try:
                    if pd.isna(v):
                        continue
                except Exception:
                    pass
                return v
        return None

    for _, r in df.iterrows():
        try:
            home = r.get("home_team"); away = r.get("visitor_team")
            # ML
            # Prefer explicit home_win_prob; fall back to NPU names
            p_home = _num(r.get("home_win_prob"))
            if p_home is None:
                p_home = _num(r.get("win_prob"))
            if p_home is None:
                p_home = _num(r.get("win_prob_from_spread"))
            # Prefer merged odds columns when present (game_odds merge uses suffix "_odds")
            home_ml = _pick_num(r, ["home_ml_odds", "home_ml"])
            away_ml = _pick_num(r, ["away_ml_odds", "away_ml"])
            # Risk limiter: require both sides odds so we can compute market no-vig and blend
            if p_home is not None and (home_ml is not None) and (away_ml is not None):
                # Optional cap on extreme odds (short favorites / longshots)
                try:
                    cap = float(max_abs_ml_odds)
                except Exception:
                    cap = 200.0

                ph_nv, pa_nv = _no_vig_probs(home_ml, away_ml)
                if ph_nv is not None and pa_nv is not None:
                    # Blend toward market no-vig to reduce overconfident model-driven EV spikes
                    w = float(max(0.0, min(1.0, float(ml_blend))))
                    p_home_blend = (w * float(p_home)) + ((1.0 - w) * float(ph_nv))
                    p_home_blend = float(min(1.0 - 1e-6, max(1e-6, p_home_blend)))
                    p_away_blend = 1.0 - p_home_blend

                    ev_h = _ev(p_home_blend, home_ml)
                    ev_a = _ev(p_away_blend, away_ml)

                    # Choose side by EV (handle 0.0 and NaN correctly)
                    evh_v = _val_or(ev_h, -1e9)
                    eva_v = _val_or(ev_a, -1e9)
                    side_ml = home if evh_v >= eva_v else away
                    ev_ml = ev_h if side_ml == home else ev_a
                    price = home_ml if side_ml == home else away_ml
                    mkt_nv = ph_nv if side_ml == home else pa_nv

                    # Additional conservative thresholds
                    try:
                        ev_thr = float(min_ml_ev)
                    except Exception:
                        ev_thr = 0.01
                    try:
                        edge_thr = float(min_ml_edge)
                    except Exception:
                        edge_thr = 0.015
                    prob_pick = p_home_blend if side_ml == home else p_away_blend
                    prob_edge = float(prob_pick) - float(mkt_nv)

                    # Skip if odds are extreme on the picked side
                    if cap > 0:
                        try:
                            if abs(float(price)) > cap:
                                price = None
                        except Exception:
                            price = None

                    # Optional: skip longshot prices above +X (e.g. +125)
                    try:
                        mpo = float(max_plus_odds)
                    except Exception:
                        mpo = 0.0
                    if mpo > 0 and price is not None:
                        try:
                            if float(price) > mpo:
                                price = None
                        except Exception:
                            price = None

                    if (price is not None) and (ev_ml is not None) and (not pd.isna(ev_ml)):
                        if (float(ev_ml) >= ev_thr) and (prob_edge >= edge_thr):
                            recs.append({
                                "market": "ML",
                                "side": side_ml,
                                "home": home,
                                "away": away,
                                "date": str(d),
                                "ev": float(ev_ml),
                                "price": float(price) if price is not None else None,
                                # Prefer no-vig implied prob (more stable than raw implied)
                                "implied_prob": float(mkt_nv) if mkt_nv is not None else None,
                                "tier": _tier('ML', float(ev_ml), None),
                            })
            # ATS
            # Use model margin from baseline or NPU column
            pm = _num(r.get("pred_margin"))
            if pm is None:
                pm = _num(r.get("spread_margin"))
            hs = _pick_num(r, ["home_spread_odds", "home_spread"])
            if pm is not None and hs is not None:
                # Keep point-edge for display/tiers
                edge_spread = pm - (-hs)

                # Prefer merged odds prices; fall back to -110 if missing
                home_spread_price = _pick_num(r, ["home_spread_price_odds", "home_spread_price"]) or -110.0
                away_spread_price = _pick_num(r, ["away_spread_price_odds", "away_spread_price"]) or -110.0

                p_home_cover = _p_home_cover_from_margin(pm, hs)
                ev_home = _ev(p_home_cover, home_spread_price) if p_home_cover is not None else None
                ev_away = _ev((1 - p_home_cover) if p_home_cover is not None else None, away_spread_price)

                # Always emit a closeout ATS pick when we have a market spread.
                # If prices exist, choose side by EV; otherwise choose by point-edge sign.
                if (ev_home is not None) or (ev_away is not None):
                    side_ats = home if (ev_home or -1) >= (ev_away or -1) else away
                    ev_ats = ev_home if side_ats == home else ev_away
                    price = home_spread_price if side_ats == home else away_spread_price
                    implied = _implied(price)
                else:
                    side_ats = home if edge_spread > 0 else away
                    ev_ats = None
                    price = home_spread_price if side_ats == home else away_spread_price
                    implied = _implied(price)

                line = hs if side_ats == home else (-hs if hs is not None else None)
                # Optional odds guard: skip longshot prices above +X (rare for spreads)
                try:
                    mpo = float(max_plus_odds)
                except Exception:
                    mpo = 0.0
                if (mpo > 0) and (price is not None):
                    try:
                        if float(price) > mpo:
                            continue
                    except Exception:
                        continue

                recs.append({
                    "market": "ATS",
                    "side": side_ats,
                    "home": home,
                    "away": away,
                    "date": str(d),
                    "ev": (float(ev_ats) if ev_ats is not None else None),
                    "price": float(price) if price is not None else None,
                    "implied_prob": implied,
                    "edge": float(edge_spread),
                    "line": line,
                    "pred_margin": pm,
                    "market_home_margin": -hs,
                    "tier": _tier("ATS", None, float(edge_spread)),
                })
            # TOTAL
            # Use model total from baseline or NPU column
            pt = _num(r.get("pred_total"))
            if pt is None:
                pt = _num(r.get("totals"))
            tot = _pick_num(r, ["total_odds", "total"])
            if pt is not None and tot is not None:
                edge_total = pt - tot
                # Prefer merged odds prices; fall back to -110 if missing
                over_price = _pick_num(r, ["total_over_price_odds", "total_over_price"]) or -110.0
                under_price = _pick_num(r, ["total_under_price_odds", "total_under_price"]) or -110.0

                p_over = _p_total_over_from_pred_total(pt, tot)
                ev_over = _ev(p_over, over_price) if p_over is not None else None
                ev_under = _ev((1 - p_over) if p_over is not None else None, under_price)

                if (ev_over is not None) or (ev_under is not None):
                    side_tot = "Over" if (ev_over or -1) >= (ev_under or -1) else "Under"
                    ev_tot = ev_over if side_tot == "Over" else ev_under
                    price = over_price if side_tot == "Over" else under_price
                    implied = _implied(price)
                else:
                    side_tot = "Over" if edge_total > 0 else "Under"
                    ev_tot = None
                    price = over_price if side_tot == "Over" else under_price
                    implied = _implied(price)

                # Always emit a closeout TOTAL pick when we have a market total.
                # Optional odds guard: skip longshot prices above +X (rare for totals)
                try:
                    mpo = float(max_plus_odds)
                except Exception:
                    mpo = 0.0
                if (mpo > 0) and (price is not None):
                    try:
                        if float(price) > mpo:
                            continue
                    except Exception:
                        continue

                recs.append({
                    "market": "TOTAL",
                    "side": side_tot,
                    "home": home,
                    "away": away,
                    "date": str(d),
                    "ev": (float(ev_tot) if ev_tot is not None else None),
                    "price": float(price) if price is not None else None,
                    "implied_prob": implied,
                    "edge": float(edge_total),
                    "line": tot,
                    "pred_total": pt,
                    "tier": _tier("TOTAL", None, float(edge_total)),
                })
        except Exception:
            continue
    out = paths.data_processed / f"recommendations_{date_str}.csv" if not out_path else Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Enforce canonical column order
    cols = [
        "market","side","home","away","date",
        "ev","price","implied_prob",
        "edge","line","pred_margin","market_home_margin","pred_total",
        "tier",
    ]
    df_out = pd.DataFrame(recs)
    for c in cols:
        if c not in df_out.columns:
            df_out[c] = None
    df_out = df_out[cols]
    df_out.to_csv(out, index=False)
    console.print({"rows": int(len(recs)), "output": str(out)})


@cli.command("export-props-recommendations")
@click.option("--date", "date_str", type=str, required=True, help="Slate date YYYY-MM-DD")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=False, help="Output CSV path; defaults to data/processed/props_recommendations_YYYY-MM-DD.csv")
@click.option(
    "--max-plus-odds",
    "max_plus_odds",
    type=float,
    default=125.0,
    show_default=True,
    help="Max allowed positive American odds for props cards (set <=0 to disable).",
)
def _export_props_recommendations_cards(date_str: str, out_path: str | None, max_plus_odds: float) -> tuple[int, "Path"]:
    """Internal helper: write props recommendation cards to CSV.

    This is used both by the CLI command and by daily-update so that
    data/processed/props_recommendations_<date>.csv has a stable schema.
    """
    import pandas as pd
    from .config import paths
    try:
        _ = pd.to_datetime(date_str).date()
    except Exception:
        raise ValueError("Invalid date_str")

    edges_p = paths.data_processed / f"props_edges_{date_str}.csv"
    preds_p = paths.data_processed / f"props_predictions_{date_str}.csv"

    df = pd.read_csv(edges_p) if edges_p.exists() else pd.DataFrame()
    pp = pd.read_csv(preds_p) if preds_p.exists() else pd.DataFrame()
    try:
        from .prop_ladders import build_card_sim_ladders, load_smart_sim_prop_ladder_lookup

        sim_ladder_lookup = load_smart_sim_prop_ladder_lookup(paths.data_processed, date_str)
    except Exception:
        sim_ladder_lookup = {}

        def build_card_sim_ladders(player_name, team_name, plays, lookup):
            return []

    model_lookup: dict[tuple[str, str], dict[str, float]] = {}
    if not pp.empty:
        for (player, team), grp in pp.groupby(["player_name", "team"], dropna=False):
            model: dict[str, float] = {}
            for col, key in [
                ("pred_pts", "pts"),
                ("pred_reb", "reb"),
                ("pred_ast", "ast"),
                ("pred_threes", "threes"),
                ("pred_stl", "stl"),
                ("pred_blk", "blk"),
                ("pred_tov", "tov"),
                ("pred_pra", "pra"),
            ]:
                if col in grp.columns:
                    try:
                        v = pd.to_numeric(grp[col], errors="coerce").dropna()
                        if not v.empty:
                            model[key] = float(v.iloc[0])
                    except Exception:
                        pass
            model_lookup[(str(player), str(team).strip().upper())] = model

    cards: list[dict] = []
    if df is None or df.empty:
        # Model-only cards
        if not pp.empty:
            for (player, team), grp in pp.groupby(["player_name", "team"], dropna=False):
                model = {}
                for col, key in [
                    ("pred_pts", "pts"),
                    ("pred_reb", "reb"),
                    ("pred_ast", "ast"),
                    ("pred_threes", "threes"),
                    ("pred_stl", "stl"),
                    ("pred_blk", "blk"),
                    ("pred_tov", "tov"),
                    ("pred_pra", "pra"),
                ]:
                    if col in grp.columns:
                        try:
                            v = pd.to_numeric(grp[col], errors="coerce").dropna()
                            if not v.empty:
                                model[key] = float(v.iloc[0])
                        except Exception:
                            pass
                sim_ladders = build_card_sim_ladders(player, team, [], sim_ladder_lookup)
                cards.append({"player": player, "team": team, "plays": [], "ladders": [], "sim_ladders": sim_ladders, "model": model})
    else:
        # Build plays per player/team
        def _num(x):
            try:
                return float(x)
            except Exception:
                return None

        # Keep cards focused on regular, comparable markets by default.
        # This avoids longshot/special markets (DD/TD, YES/NO) and extreme prices dominating cards.
        def _is_regular_play(row) -> bool:
            try:
                stat = str(row.get("stat") or row.get("market") or "").lower()
                side = str(row.get("side") or "").upper()
                if stat in {"dd", "td"}:
                    return False
                if side not in {"OVER", "UNDER"}:
                    return False
                price = pd.to_numeric(row.get("price"), errors="coerce")
                if not pd.notna(price):
                    return False
                # Regular pricing window (production filter)
                try:
                    mpo = float(max_plus_odds)
                except Exception:
                    mpo = 150.0
                if mpo <= 0:
                    mpo = 1e9
                if price < -150 or price > mpo:
                    return False
                edge = pd.to_numeric(row.get("edge"), errors="coerce")
                ev = pd.to_numeric(row.get("ev"), errors="coerce")
                if not pd.notna(edge) or float(edge) <= 0.0:
                    return False
                if not pd.notna(ev) or float(ev) <= 0.0:
                    return False
                # PTS/PRA have recently underperformed; only show them on cards
                # when the edge is meaningfully strong.
                if stat in {"pts", "pra"}:
                    edge_abs = edge
                    if not pd.notna(edge_abs) or abs(float(edge_abs)) < 0.15:
                        return False
                line = pd.to_numeric(row.get("line"), errors="coerce")
                if not pd.notna(line):
                    return False
                return True
            except Exception:
                return False

        for keys, grp in df.groupby([c for c in ["player_name", "team"] if c in df.columns], dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            player = keys[0] if len(keys) > 0 else None
            team = keys[1] if len(keys) > 1 else None
            g2 = grp.copy()
            try:
                g2 = g2[g2.apply(_is_regular_play, axis=1)].copy()
            except Exception:
                pass
            g2["ev_pct"] = pd.to_numeric(g2.get("ev"), errors="coerce") * 100.0 if "ev" in g2.columns else None
            plays = []
            for _, r in g2.iterrows():
                plays.append(
                    {
                        "market": r.get("stat"),
                        "side": r.get("side"),
                        "line": _num(r.get("line")),
                        "price": _num(r.get("price")),
                        "edge": _num(r.get("edge")),
                        "ev": _num(r.get("ev")),
                        "ev_pct": _num(r.get("ev")) * 100.0 if _num(r.get("ev")) is not None else None,
                        "book": r.get("bookmaker"),
                    }
                )
            sim_ladders = build_card_sim_ladders(player, team, plays, sim_ladder_lookup)
            model = model_lookup.get((str(player), str(team).strip().upper()), {})
            if plays or sim_ladders or model:
                cards.append({"player": player, "team": team, "plays": plays, "ladders": [], "sim_ladders": sim_ladders, "model": model})

    out = paths.data_processed / f"props_recommendations_{date_str}.csv" if not out_path else Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cards, columns=["player", "team", "plays", "ladders", "sim_ladders", "model"]).to_csv(out, index=False)
    console.print({"rows": int(len(cards)), "output": str(out)})
    return int(len(cards)), out


def export_props_recommendations_cmd(date_str: str, out_path: str | None):
    """Export props recommendation cards to CSV from edges (or model-only if edges missing)."""
    try:
        rows, out = _export_props_recommendations_cards(date_str, out_path)
    except Exception:
        console.print("Invalid --date (YYYY-MM-DD)", style="red")
        return
    console.print({"rows": int(rows), "output": str(out)})


def _export_best_edges_snapshot(
    date_str: str,
    max_games: int = 10,
    max_props: int = 25,
    out_games_path: str | None = None,
    out_props_path: str | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write authoritative daily best-edges snapshots (games + props) to CSV.

    Outputs (defaults):
      - data/processed/best_edges_games_YYYY-MM-DD.csv
      - data/processed/best_edges_props_YYYY-MM-DD.csv

        Selection rules intentionally match the current best-ROI portfolio defaults:
      - Games: de-dupe to 1 pick per game, rank by EV (ML) else abs(edge) (ATS/TOTAL)
            - Props: de-dupe to 1 pick per player, rank by EV, filter to pts/threes, min EV >= 1.0%
                            and exclude known low-ROI books by default.
    """
    import ast as _ast
    import json as _json
    import pandas as pd
    from .config import paths
    from .scoring import (
        score_game_pick_0_100,
        score_prop_pick_0_100,
        dump_components_json,
    )

    try:
        _ = pd.to_datetime(date_str).date()
    except Exception:
        raise ValueError("Invalid date_str")

    proc = paths.data_processed
    games_in = proc / f"recommendations_{date_str}.csv"
    props_in = proc / f"props_recommendations_{date_str}.csv"

    out_games = proc / f"best_edges_games_{date_str}.csv" if not out_games_path else Path(out_games_path)
    out_props = proc / f"best_edges_props_{date_str}.csv" if not out_props_path else Path(out_props_path)
    out_games.parent.mkdir(parents=True, exist_ok=True)
    out_props.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        if out_games.exists() and out_props.exists():
            return out_games, out_props

    def _num(x):
        try:
            v = float(x)
            return v if pd.notna(v) else None
        except Exception:
            return None

    def _implied_prob(american: object) -> float | None:
        try:
            a = float(american)
        except Exception:
            return None
        if a == 0:
            return None
        if a > 0:
            return float(100.0 / (a + 100.0))
        return float((-a) / ((-a) + 100.0))

    def _rank_value_from_score(r: dict) -> float:
        try:
            s = _num(r.get("score"))
            if s is not None:
                return float(s)
        except Exception:
            pass
        # Fallback to legacy best-edge value
        try:
            m = str(r.get("market") or "").upper()
            ev = _num(r.get("ev"))
            edge = _num(r.get("edge"))
            if m == "ML":
                return float(ev or 0.0)
            if m in {"ATS", "TOTAL"}:
                return float(abs(edge or 0.0))
            if edge is not None and abs(edge) > 0:
                return float(abs(edge))
            if ev is not None and ev != 0:
                return float(ev)
        except Exception:
            pass
        return 0.0

    # ----- Games -----
    games_rows: list[dict] = []

    # Best-effort odds backfill map (used when recommendations/snapshots are missing price fields)
    odds_map: dict[str, dict] = {}
    try:
        go = proc / f"game_odds_{date_str}.csv"
        if go.exists():
            odf = pd.read_csv(go)
            if isinstance(odf, pd.DataFrame) and not odf.empty:
                for _, rr in odf.iterrows():
                    h = str(rr.get("home_team") or "").strip()
                    a = str(rr.get("visitor_team") or rr.get("away_team") or "").strip()
                    if not h or not a:
                        continue
                    key = f"{a}@@{h}".lower()
                    odds_map[key] = rr.to_dict()
                    # Also store swapped for resilience
                    odds_map[f"{h}@@{a}".lower()] = rr.to_dict()
    except Exception:
        odds_map = {}

    if games_in.exists():
        try:
            gdf = pd.read_csv(games_in)
        except Exception:
            gdf = pd.DataFrame()

        if gdf is not None and not gdf.empty:
            cols = {c.lower(): c for c in gdf.columns}

            def _col(*names: str) -> str | None:
                for n in names:
                    c = cols.get(n.lower())
                    if c:
                        return c
                return None

            market_col = _col("market")
            side_col = _col("side", "pick")
            home_col = _col("home", "home_team")
            away_col = _col("away", "visitor_team")
            ev_col = _col("ev")
            edge_col = _col("edge")
            line_col = _col("line")
            price_col = _col("price", "odds")
            tier_col = _col("tier")
            gid_col = _col("game_id")

            # Best-effort game_id mapping via schedule
            schedule_map: dict[tuple[str, str], str] = {}
            try:
                sched_p = proc / "schedule_2025_26.csv"
                if sched_p.exists():
                    sdf = pd.read_csv(sched_p)
                    scols = {c.lower(): c for c in sdf.columns}
                    if {"date_utc", "home_tricode", "away_tricode", "game_id"}.issubset(set(scols.keys())):
                        day = sdf[pd.to_datetime(sdf[scols["date_utc"]], errors="coerce").dt.strftime("%Y-%m-%d") == date_str].copy()
                        for _, rr in day.iterrows():
                            ht = str(rr.get(scols["home_tricode"]) or "").strip().upper()
                            at = str(rr.get(scols["away_tricode"]) or "").strip().upper()
                            gid = str(rr.get(scols["game_id"]) or "").strip()
                            if ht and at and gid:
                                schedule_map[(at, ht)] = gid
            except Exception:
                schedule_map = {}

            def _norm_team(x: object) -> str:
                return str(x or "").strip()

            # De-dupe: 1 per game (use game_id if available else away@@home)
            best_by_game: dict[str, dict] = {}
            for _, rr in gdf.iterrows():
                market = str(rr.get(market_col) if market_col else rr.get("market") or "").upper()
                home = _norm_team(rr.get(home_col) if home_col else rr.get("home"))
                away = _norm_team(rr.get(away_col) if away_col else rr.get("away"))
                side = _norm_team(rr.get(side_col) if side_col else rr.get("side"))
                row = {
                    "date": date_str,
                    "market": market,
                    "side": side,
                    "home": home,
                    "away": away,
                    "ev": rr.get(ev_col) if ev_col else rr.get("ev"),
                    "edge": rr.get(edge_col) if edge_col else rr.get("edge"),
                    "line": rr.get(line_col) if line_col else rr.get("line"),
                    "price": rr.get(price_col) if price_col else rr.get("price"),
                    "implied_prob": rr.get(cols.get("implied_prob")) if cols.get("implied_prob") else rr.get("implied_prob"),
                    "pred_margin": rr.get(cols.get("pred_margin")) if cols.get("pred_margin") else rr.get("pred_margin"),
                    "market_home_margin": rr.get(cols.get("market_home_margin")) if cols.get("market_home_margin") else rr.get("market_home_margin"),
                    "pred_total": rr.get(cols.get("pred_total")) if cols.get("pred_total") else rr.get("pred_total"),
                    "tier": rr.get(tier_col) if tier_col else rr.get("tier"),
                }

                # Odds backfill for ATS/TOTAL when price is missing
                try:
                    mkt = str(row.get("market") or "").upper()
                    price_v = _num(row.get("price"))
                    line_v = _num(row.get("line"))
                    if (mkt in {"ATS", "TOTAL"}) and (price_v is None):
                        orec = odds_map.get(f"{away}@@{home}".lower())
                        if isinstance(orec, dict):
                            if mkt == "ATS":
                                hp = _num(orec.get("home_spread_price"))
                                ap = _num(orec.get("away_spread_price"))
                                # default -110 if not present
                                hp = hp if hp is not None else -110.0
                                ap = ap if ap is not None else -110.0
                                price_v = hp if str(side) == str(home) else ap
                                # optional: fill line if missing
                                if line_v is None:
                                    hs = _num(orec.get("home_spread"))
                                    if hs is not None:
                                        line_v = hs if str(side) == str(home) else -hs
                            else:
                                op = _num(orec.get("total_over_price"))
                                up = _num(orec.get("total_under_price"))
                                op = op if op is not None else -110.0
                                up = up if up is not None else -110.0
                                is_over = str(side or "").strip().lower().startswith("o")
                                price_v = op if is_over else up
                                if line_v is None:
                                    tt = _num(orec.get("total"))
                                    if tt is not None:
                                        line_v = tt

                            row["price"] = float(price_v) if price_v is not None else None
                            row["line"] = float(line_v) if line_v is not None else row.get("line")
                            if row.get("implied_prob") is None and price_v is not None:
                                row["implied_prob"] = _implied_prob(price_v)
                except Exception:
                    pass

                # Score (0-100) + explain/breakdown
                try:
                    s, comps, sexpl = score_game_pick_0_100(
                        market=row.get("market"),
                        ev=row.get("ev"),
                        edge=row.get("edge"),
                        price=row.get("price"),
                    )
                    row["score"] = s
                    row["score_explain"] = sexpl
                    row["score_components"] = dump_components_json(comps)
                except Exception:
                    row["score"] = None
                    row["score_explain"] = None
                    row["score_components"] = None

                # Why (simple, deterministic, data-driven)
                try:
                    mkt = str(row.get("market") or "").upper()
                    if mkt == "ML":
                        evv = _num(row.get("ev"))
                        pr = _num(row.get("price"))
                        imp = _num(row.get("implied_prob"))
                        parts = []
                        if evv is not None:
                            parts.append(f"EV {evv:.3f}")
                        if pr is not None:
                            ptxt = f"+{int(round(pr))}" if pr > 0 else f"{int(round(pr))}"
                            parts.append(f"odds {ptxt}")
                        if imp is not None:
                            parts.append(f"implied {imp:.3f}")
                        row["why_explain"] = "; ".join(parts) if parts else None
                    elif mkt == "ATS":
                        pm = _num(row.get("pred_margin"))
                        mline = _num(row.get("market_home_margin"))
                        ed = _num(row.get("edge"))
                        parts = []
                        if pm is not None:
                            parts.append(f"model margin {pm:.1f}")
                        if mline is not None:
                            parts.append(f"market home {mline:+.1f}")
                        if ed is not None:
                            parts.append(f"edge {ed:+.2f}")
                        row["why_explain"] = "; ".join(parts) if parts else None
                    elif mkt == "TOTAL":
                        pt = _num(row.get("pred_total"))
                        ln = _num(row.get("line"))
                        ed = _num(row.get("edge"))
                        parts = []
                        if pt is not None:
                            parts.append(f"model total {pt:.1f}")
                        if ln is not None:
                            parts.append(f"line {ln:.1f}")
                        if ed is not None:
                            parts.append(f"edge {ed:+.2f}")
                        row["why_explain"] = "; ".join(parts) if parts else None
                    else:
                        # Fallback
                        ed = _num(row.get("edge"))
                        evv = _num(row.get("ev"))
                        parts = []
                        if evv is not None:
                            parts.append(f"EV {evv:.3f}")
                        if ed is not None:
                            parts.append(f"edge {ed:+.2f}")
                        row["why_explain"] = "; ".join(parts) if parts else None
                except Exception:
                    row["why_explain"] = None

                gid = None
                try:
                    if gid_col:
                        gid = str(rr.get(gid_col) or "").strip() or None
                except Exception:
                    gid = None

                # If no gid, attempt schedule mapping using tricodes when possible
                if not gid:
                    try:
                        from .teams import to_tricode as _to_tri, normalize_team as _norm
                        at = str(_to_tri(_norm(away)) or "").strip().upper()
                        ht = str(_to_tri(_norm(home)) or "").strip().upper()
                        gid = schedule_map.get((at, ht))
                    except Exception:
                        gid = None
                if gid:
                    row["game_id"] = gid

                game_key = (gid or f"{away}@@{home}").lower()
                prev = best_by_game.get(game_key)
                v = _rank_value_from_score(row)
                pv = _rank_value_from_score(prev) if isinstance(prev, dict) else -1.0
                if prev is None or v > pv:
                    row["best_edge_value"] = _rank_value_from_score(row)
                    best_by_game[game_key] = row

            arr = list(best_by_game.values())
            arr.sort(key=lambda r: float(r.get("score") or r.get("best_edge_value") or 0.0), reverse=True)
            games_rows = arr[: max(0, int(max_games))]

    # ----- Props -----
    props_rows: list[dict] = []
    # Prefer props_edges as the authoritative per-book line/EV source for snapshots
    props_edges_in = proc / f"props_edges_{date_str}.csv"

    # Load model baselines for why_explain (optional)
    preds_map: dict[tuple[str, str], dict] = {}
    try:
        preds_p = proc / f"props_predictions_{date_str}.csv"
        if preds_p.exists():
            pp = pd.read_csv(preds_p)
            if pp is not None and not pp.empty:
                # key: (player_id, TEAM) preferred; also fallback to (player_name, TEAM)
                for _, rr in pp.iterrows():
                    try:
                        pid = str(rr.get("player_id") or "").strip()
                        pname = str(rr.get("player_name") or "").strip().lower()
                        team = str(rr.get("team") or "").strip().upper()
                        rec = rr.to_dict()
                        if pid and team:
                            preds_map[(pid, team)] = rec
                        if pname and team:
                            preds_map[(pname, team)] = rec
                    except Exception:
                        continue
    except Exception:
        preds_map = {}

    if props_edges_in.exists():
        try:
            pdf = pd.read_csv(props_edges_in)
        except Exception:
            pdf = pd.DataFrame()

        if pdf is not None and not pdf.empty:
            # Default exclusions based on recent backtest breakdowns.
            default_exclude_books = {"fanduel", "draftkings", "williamhill_us"}
            def _is_regular_edge_play(rr: pd.Series) -> bool:
                try:
                    stat = str(rr.get("stat") or "").strip().lower()
                    side = str(rr.get("side") or "").strip().upper()
                    # Best ROI default: focus on pts + threes.
                    if stat not in {"pts", "threes"}:
                        return False
                    if stat in {"dd", "td"}:
                        return False
                    if side not in {"OVER", "UNDER"}:
                        return False
                    # Exclude poor-performing books by default.
                    try:
                        bk = str(rr.get("bookmaker") or "").strip().lower()
                        if bk and bk in default_exclude_books:
                            return False
                    except Exception:
                        pass
                    price = pd.to_numeric(rr.get("price"), errors="coerce")
                    if not pd.notna(price):
                        return False
                    line = pd.to_numeric(rr.get("line"), errors="coerce")
                    if not pd.notna(line):
                        return False
                    # Minimum EV filter: >= 1.0% (ev >= 0.01).
                    ev = pd.to_numeric(rr.get("ev"), errors="coerce")
                    if not pd.notna(ev):
                        return False
                    # Normalize percent-encoded EV if needed.
                    evf = float(ev)
                    if abs(evf) > 1.5:
                        evf = evf / 100.0
                    if evf < 0.01:
                        return False
                    return True
                except Exception:
                    return False

            best_by_player: dict[str, dict] = {}
            for _, rr in pdf.iterrows():
                try:
                    if not _is_regular_edge_play(rr):
                        continue
                    player = str(rr.get("player_name") or "").strip()
                    pid = str(rr.get("player_id") or "").strip()
                    team = str(rr.get("team") or "").strip().upper()
                    stat = str(rr.get("stat") or "").strip().lower()
                    side = str(rr.get("side") or "").strip().upper()
                    line = rr.get("line")
                    price = rr.get("price")
                    ev = rr.get("ev")
                    edge = rr.get("edge")
                    imp = rr.get("implied_prob")
                    mp = rr.get("model_prob")
                    book = rr.get("bookmaker") or rr.get("bookmaker_title")

                    # Normalize EV: some feeds may encode as percent units (e.g., 30 == 30%)
                    try:
                        evn = _num(ev)
                        if evn is not None and abs(float(evn)) > 1.5:
                            ev = float(evn) / 100.0
                    except Exception:
                        pass

                    row = {
                        "date": date_str,
                        "player": player,
                        "team": team,
                        "market": stat,
                        "side": side,
                        "line": line,
                        "price": price,
                        "ev": ev,
                        "edge": edge,
                        "implied_prob": imp,
                        "model_prob": mp,
                        "book": book,
                        "score": None,
                        "score_explain": None,
                        "score_components": None,
                        "tier": None,
                    }

                    # Score (0-100) + explain/breakdown
                    try:
                        s, comps, sexpl = score_prop_pick_0_100(
                            ev=row.get("ev"),
                            edge=row.get("edge"),
                            model_prob=row.get("model_prob"),
                            implied_prob=row.get("implied_prob"),
                            price=row.get("price"),
                        )
                        row["score"] = s
                        row["score_explain"] = sexpl
                        row["score_components"] = dump_components_json(comps)
                    except Exception:
                        row["score"] = None
                        row["score_explain"] = None
                        row["score_components"] = None

                    # Join baseline prediction for why
                    pred_row = None
                    if pid and team:
                        pred_row = preds_map.get((pid, team))
                    if pred_row is None and player and team:
                        pred_row = preds_map.get((player.strip().lower(), team))

                    pred_val = None
                    try:
                        if isinstance(pred_row, dict) and stat:
                            key_map = {
                                "pts": "pred_pts",
                                "reb": "pred_reb",
                                "ast": "pred_ast",
                                "threes": "pred_threes",
                                "3pt": "pred_threes",
                                "pra": "pred_pra",
                            }
                            col = key_map.get(stat)
                            if col:
                                pred_val = pred_row.get(col)
                    except Exception:
                        pred_val = None

                    # Why explain
                    try:
                        parts = []
                        pv = _num(pred_val)
                        ln = _num(line)
                        if pv is not None:
                            parts.append(f"model {stat} {pv:.1f}")
                        if ln is not None:
                            parts.append(f"line {ln:.1f}")
                        if side:
                            parts.append(side)
                        edv = _num(edge)
                        if edv is not None:
                            parts.append(f"edge {edv:+.3f}")
                        evv = _num(ev)
                        if evv is not None:
                            parts.append(f"EV {evv:.3f}")
                        mpv = _num(mp)
                        if mpv is not None:
                            parts.append(f"model_p {mpv:.3f}")
                        ipv = _num(imp)
                        if ipv is not None:
                            parts.append(f"implied {ipv:.3f}")
                        if book:
                            parts.append(str(book))
                        row["why_explain"] = "; ".join([p for p in parts if p]) if parts else None
                    except Exception:
                        row["why_explain"] = None

                    key = f"{player.strip().lower()}@@{team.strip().upper()}"
                    prev = best_by_player.get(key)
                    v = _rank_value_from_score(row)
                    pv = _rank_value_from_score(prev) if isinstance(prev, dict) else -1.0
                    if prev is None or v > pv:
                        row["best_edge_value"] = _rank_value_from_score(row)
                        best_by_player[key] = row
                except Exception:
                    continue

            arrp = list(best_by_player.values())
            arrp.sort(key=lambda r: float(r.get("score") or r.get("best_edge_value") or 0.0), reverse=True)
            props_rows = arrp[: max(0, int(max_props))]

    elif props_in.exists():
        # Fallback to cards file if edges missing
        try:
            pdf = pd.read_csv(props_in)
        except Exception:
            pdf = pd.DataFrame()

        def _parse_obj(val):
            if isinstance(val, (list, dict)):
                return val
            s = str(val or "")
            if s.strip() in {"", "None", "nan"}:
                return None
            try:
                return _json.loads(s)
            except Exception:
                try:
                    return _ast.literal_eval(s)
                except Exception:
                    return None

        def _top_play_from_row(row: pd.Series) -> dict | None:
            try:
                plays = _parse_obj(row.get("plays"))
                if not (isinstance(plays, list) and plays):
                    return None

                def _evp(p: dict) -> float:
                    try:
                        v = p.get("ev_pct")
                        if v is not None and not pd.isna(v):
                            return float(v)
                        ve = p.get("ev")
                        return (float(ve) * 100.0) if (ve is not None and not pd.isna(ve)) else -1e9
                    except Exception:
                        return -1e9

                cand = list(plays)
                cand.sort(key=lambda p: (_evp(p), abs(p.get("edge") or 0.0)), reverse=True)
                tp = cand[0]
                return tp if isinstance(tp, dict) else None
            except Exception:
                return None

        if pdf is not None and not pdf.empty:
            best_by_player: dict[str, dict] = {}
            for _, rr in pdf.iterrows():
                tp = _top_play_from_row(rr)
                if not isinstance(tp, dict) or not tp:
                    continue
                player = str(rr.get("player") or rr.get("player_name") or "").strip()
                team = str(rr.get("team") or "").strip().upper()
                row = {
                    "date": date_str,
                    "player": player,
                    "team": team,
                    "market": str(tp.get("market") or "").lower(),
                    "side": str(tp.get("side") or "").upper(),
                    "line": tp.get("line"),
                    "price": tp.get("price"),
                    "ev": tp.get("ev"),
                    "edge": tp.get("edge"),
                    "tier": rr.get("tier"),
                    "score": rr.get("score"),
                    "why_explain": None,
                }
                # Minimal why for fallback
                try:
                    parts = []
                    if row.get("market"):
                        parts.append(str(row.get("market")).upper())
                    if row.get("side"):
                        parts.append(str(row.get("side")).upper())
                    ln = _num(row.get("line"))
                    if ln is not None:
                        parts.append(f"line {ln:.1f}")
                    edv = _num(row.get("edge"))
                    if edv is not None:
                        parts.append(f"edge {edv:+.3f}")
                    evv = _num(row.get("ev"))
                    if evv is not None:
                        parts.append(f"EV {evv:.3f}")
                    row["why_explain"] = "; ".join(parts) if parts else None
                except Exception:
                    row["why_explain"] = None
                key = f"{player.strip().lower()}@@{team.strip().upper()}"
                prev = best_by_player.get(key)
                v = _best_edge_value_prop(row)
                pv = _best_edge_value_prop(prev) if isinstance(prev, dict) else -1.0
                if prev is None or v > pv:
                    row["best_edge_value"] = v
                    best_by_player[key] = row

            arrp = list(best_by_player.values())
            arrp.sort(key=lambda r: float(r.get("best_edge_value") or 0.0), reverse=True)
            props_rows = arrp[: max(0, int(max_props))]

    # Write outputs (even if empty; keeps downstream deterministic)
    games_df_out = pd.DataFrame(games_rows)
    props_df_out = pd.DataFrame(props_rows)

    # Canonical column orders
    games_cols = [
        "date",
        "game_id",
        "market",
        "side",
        "home",
        "away",
        "line",
        "price",
        "ev",
        "edge",
        "tier",
        "score",
        "score_explain",
        "score_components",
        "best_edge_value",
        "why_explain",
    ]
    props_cols = [
        "date",
        "player",
        "team",
        "market",
        "side",
        "line",
        "price",
        "ev",
        "edge",
        "implied_prob",
        "model_prob",
        "model_prob_raw",
        "tier",
        "score",
        "score_explain",
        "score_components",
        "best_edge_value",
        "why_explain",
    ]
    for c in games_cols:
        if c not in games_df_out.columns:
            games_df_out[c] = None
    for c in props_cols:
        if c not in props_df_out.columns:
            props_df_out[c] = None
    games_df_out = games_df_out[games_cols]
    props_df_out = props_df_out[props_cols]

    games_df_out.to_csv(out_games, index=False)
    props_df_out.to_csv(out_props, index=False)
    return out_games, out_props


@cli.command("export-best-edges")
@click.option("--date", "date_str", type=str, required=True, help="Slate date YYYY-MM-DD")
@click.option("--max-games", type=int, default=10, show_default=True, help="Max game picks (1 per game)")
@click.option("--max-props", type=int, default=25, show_default=True, help="Max prop picks (1 per player)")
@click.option("--overwrite", is_flag=True, default=False, show_default=True, help="Overwrite existing snapshot files")
@click.option("--out-games", "out_games_path", type=click.Path(dir_okay=False), required=False, help="Output CSV path for games snapshot")
@click.option("--out-props", "out_props_path", type=click.Path(dir_okay=False), required=False, help="Output CSV path for props snapshot")
def export_best_edges_cmd(date_str: str, max_games: int, max_props: int, overwrite: bool, out_games_path: str | None, out_props_path: str | None):
    """Export authoritative daily best-edges snapshots (games + props) for tracking."""
    console.rule("Export Best Edges")
    try:
        outg, outp = _export_best_edges_snapshot(
            date_str=date_str,
            max_games=int(max_games),
            max_props=int(max_props),
            out_games_path=out_games_path,
            out_props_path=out_props_path,
            overwrite=bool(overwrite),
        )
    except Exception as e:
        console.print(f"Failed to export best edges: {e}", style="red")
        return
    console.print({"games": str(outg), "props": str(outp)})


@cli.command("odds-refresh")
@click.option("--date", "date_str", type=str, required=False, help="Target date YYYY-MM-DD; defaults to the current US local slate date")
@click.option("--api-key", envvar="ODDS_API_KEY", type=str, required=False, help="OddsAPI key (or set env ODDS_API_KEY)")
@click.option("--min-prop-edge", type=float, default=0.02, show_default=True, help="Minimum edge filter for props edges")
@click.option("--min-prop-ev", type=float, default=0.0, show_default=True, help="Minimum EV filter for props edges")
def odds_refresh_cmd(date_str: str | None, api_key: str | None, min_prop_edge: float, min_prop_ev: float):
    """Refresh odds and recompute odds-related edges only.

    - Fetch current game odds (OddsAPI via ODDS_API_KEY) and write game_odds_<date>.csv
    - Compute props edges for the date using current odds (source=auto), saving props_edges_<date>.csv
    - Do not retrain or rebuild predictions beyond what's needed for edges
    """
    console.rule("Odds Refresh (odds + props-edges)")
    try:
        target_date = _resolve_target_date(date_str)
    except Exception:
        console.print("Invalid --date (YYYY-MM-DD)", style="red"); return

    # Load API key from env or .env
    if not api_key:
        api_key = _load_dotenv_key("ODDS_API_KEY")

    # 1) Fetch current game odds and save standardized CSV for frontend/merges
    try:
        if api_key:
            console.print("Fetching current game odds (OddsAPI)...", style="cyan")
            cfg = OddsApiConfig(api_key=api_key)
            go = fetch_game_odds_current(cfg, pd.to_datetime(target_date))
            if go is not None and not go.empty:
                out_csv = paths.data_processed / f"game_odds_{target_date}.csv"
                out_csv.parent.mkdir(parents=True, exist_ok=True)
                go.to_csv(out_csv, index=False)
                console.print({"game_odds_rows": int(len(go)), "output": str(out_csv)})
            else:
                console.print("No game odds returned (OddsAPI)", style="yellow")
        else:
            console.print("ODDS_API_KEY not set; skipping game odds fetch", style="yellow")
    except Exception as e:
        console.print(f"Game odds fetch failed: {e}", style="yellow")

    # 2) Compute props edges using current odds (source auto, mode current)
    try:
        console.print("Computing props edges from current odds...", style="cyan")
        # Calibrate sigma best-effort; fall back to defaults if it fails
        try:
            sigma = calibrate_sigma_for_date(str(target_date), window_days=30, min_rows=200, defaults=SigmaConfig())
        except Exception:
            sigma = SigmaConfig()
        edges = compute_props_edges(
            date=str(target_date),
            sigma=sigma,
            use_saved=False,
            mode="current",
            api_key=api_key,
            source="auto",
            predictions_path=None,
            from_file_only=False,
        )
        if edges is None or edges.empty:
            console.print("No props edges computed (missing odds or predictions)", style="yellow"); return
        # Filter and save
        edges = edges[(edges["edge"] >= float(min_prop_edge)) & (edges["ev"] >= float(min_prop_ev))].copy()
        edges.sort_values(["stat", "edge"], ascending=[True, False], inplace=True)
        out = paths.data_processed / f"props_edges_{target_date}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        edges.to_csv(out, index=False)
        console.print({"props_edges_rows": int(len(edges)), "output": str(out)})
    except Exception as e:
        console.print(f"Props edges computation failed: {e}", style="yellow")


@cli.command("odds-snapshots")
@click.option("--date", "date_str", type=str, required=False, help="Target date YYYY-MM-DD; defaults to the current US local slate date")
@click.option("--api-key", envvar="ODDS_API_KEY", type=str, required=False, help="OddsAPI key (or set env ODDS_API_KEY)")
def odds_snapshots_cmd(date_str: str | None, api_key: str | None):
    """Write standardized OddsAPI snapshots for a given date.

    Produces CSVs under data/processed:
    - oddsapi_events_<date>.csv: events on the US/Eastern calendar day with has_odds flag
    - game_odds_<date>.csv: consensus moneyline/spread/total from current event odds

    Notes:
    - Requires ODDS_API_KEY
    - CSV-first; no parquet dependency
    """
    console.rule("Odds Snapshots (events + consensus lines)")
    import requests as _requests
    from .odds_api import ODDS_HOST, NBA_SPORT_KEY, OddsApiConfig, fetch_game_odds_current, consensus_lines_at_close

    try:
        target_date = _resolve_target_date(date_str)
    except Exception:
        console.print("Invalid --date (YYYY-MM-DD)", style="red"); return

    if not api_key:
        api_key = _load_dotenv_key("ODDS_API_KEY")
    if not api_key:
        console.print("Provide --api-key, set ODDS_API_KEY env, or add to .env at repo root.", style="red"); return

    # 1) Fetch events list and filter to US/Eastern day; write oddsapi_events_<date>.csv
    events_out = paths.data_processed / f"oddsapi_events_{target_date}.csv"
    # Also write an allowlisted alias for the site (tracked by !data/processed/odds_*.csv)
    events_out_alias = paths.data_processed / f"odds_events_{target_date}.csv"
    try:
        ev_resp = _requests.get(f"{ODDS_HOST}/v4/sports/{NBA_SPORT_KEY}/events", params={"apiKey": api_key}, headers={"Accept":"application/json","User-Agent":"nba-betting/1.0"}, timeout=45)
        ev_resp.raise_for_status()
        events = ev_resp.json() or []
    except Exception as e:
        console.print(f"Events fetch failed: {e}", style="yellow")
        events = []

    def _et_date(iso_str: str):
        try:
            ct_raw = pd.to_datetime(iso_str, utc=True)
        except Exception:
            return None
        for tzname in ("America/New_York", "US/Eastern"):
            try:
                return ct_raw.tz_convert(tzname).date()
            except Exception:
                continue
        try:
            month = int(ct_raw.month)
            offset_hours = 4 if 3 <= month <= 11 else 5
            return (ct_raw - pd.Timedelta(hours=offset_hours)).date()
        except Exception:
            return ct_raw.date()

    ev_rows: list[dict] = []
    # If we also fetch game odds later, capture event_ids with odds
    have_ids: set[str] = set()

    # 2) Fetch current game odds and write consensus game_odds_<date>.csv
    game_odds_out = paths.data_processed / f"game_odds_{target_date}.csv"
    try:
        cfg = OddsApiConfig(api_key=api_key)
        long_df = fetch_game_odds_current(cfg, pd.to_datetime(target_date))
        if long_df is not None and not long_df.empty:
            try:
                have_ids = set([str(x) for x in long_df["event_id"].dropna().astype(str).unique()])
            except Exception:
                have_ids = set()
            wide = consensus_lines_at_close(long_df)
            if wide is not None and not wide.empty:
                tmp = wide.copy()
                tmp["date"] = pd.to_datetime(tmp["commence_time"], utc=True).dt.tz_convert("US/Eastern").dt.strftime("%Y-%m-%d")
                tmp = tmp.rename(columns={"away_team":"visitor_team"})
                # Map consensus point columns to canonical names
                if "spread_point" in tmp.columns:
                    tmp["home_spread"] = tmp["spread_point"]
                    tmp["away_spread"] = tmp["home_spread"].apply(lambda x: -x if pd.notna(x) else pd.NA)
                if "total_point" in tmp.columns:
                    tmp["total"] = tmp["total_point"]
                # Build output with price columns included (+ liquidity counts when available)
                cols = [c for c in [
                    "date","commence_time","home_team","visitor_team",
                    "home_ml","away_ml",
                    "home_spread","away_spread","home_spread_price","away_spread_price",
                    "total","total_over_price","total_under_price",
                    "books_count","books_h2h","books_spreads","books_totals"
                ] if c in tmp.columns]
                out_df = tmp[cols].copy()
                out_df["bookmaker"] = "oddsapi_consensus"
                # Persist Bovada odds and fill any missing fields (prefer OddsAPI values)
                try:
                    bov = fetch_bovada_odds_current(str(target_date))
                    if isinstance(bov, pd.DataFrame) and not bov.empty:
                        bov = bov.rename(columns={"away_team":"visitor_team"}).copy()
                        bov["_key"] = bov.apply(lambda r: f"{str(r.get('home_team') or '').strip()}@@{str(r.get('visitor_team') or '').strip()}", axis=1)
                        smap = bov.set_index("_key").to_dict(orient="index")
                        def _fill_from_bov(row):
                            k = f"{str(row.get('home_team') or '').strip()}@@{str(row.get('visitor_team') or '').strip()}"
                            rec = smap.get(k)
                            if not rec:
                                return row
                            def _fill(col):
                                if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
                                    return row[col]
                                return rec.get(col)
                            # Fill fields only if missing
                            for c in [
                                "home_ml","away_ml",
                                "home_spread","away_spread","home_spread_price","away_spread_price",
                                "total","total_over_price","total_under_price"
                            ]:
                                val = _fill(c)
                                row[c] = val
                            return row
                        out_df = out_df.apply(_fill_from_bov, axis=1)
                except Exception as ex:
                    console.print({"warning":"Bovada fill failed","error":str(ex)}, style="yellow")
                out_df.to_csv(game_odds_out, index=False)
                console.print({"game_odds_rows": int(len(out_df)), "output": str(game_odds_out)})
            else:
                console.print("No consensus rows from current game odds", style="yellow")
        else:
            console.print("No game odds returned (OddsAPI)", style="yellow")
    except Exception as e:
        console.print(f"Game odds fetch failed: {e}", style="yellow")

    # 3) Per-period market lines from Bovada (quarters/halves) if available
    try:
        from .odds_bovada import fetch_bovada_period_lines_current
        pl = fetch_bovada_period_lines_current(str(target_date))
        if isinstance(pl, pd.DataFrame) and not pl.empty:
            out_pl = paths.data_processed / f"period_lines_{target_date}.csv"
            out_pl.parent.mkdir(parents=True, exist_ok=True)
            pl.to_csv(out_pl, index=False)
            console.print({"period_lines_rows": int(len(pl)), "period_lines_output": str(out_pl)})
        else:
            # Don't overwrite an existing file with empty output
            pass
    except Exception as e:
        console.print(f"Period lines fetch failed: {e}", style="yellow")

    try:
        tgt = pd.to_datetime(target_date).date()
        for ev in events:
            try:
                ct_et = _et_date(ev.get("commence_time"))
            except Exception:
                ct_et = None
            if ct_et == tgt:
                eid = ev.get("id")
                ev_rows.append({
                    "event_id": eid,
                    "commence_time": ev.get("commence_time"),
                    "home_team": ev.get("home_team"),
                    "away_team": ev.get("away_team"),
                    "has_odds": (str(eid) in have_ids) if eid else False,
                })
        if ev_rows:
            df_ev = pd.DataFrame(ev_rows)
            df_ev.to_csv(events_out, index=False)
            # Write alias for site allowlist
            try:
                df_ev.to_csv(events_out_alias, index=False)
            except Exception:
                pass
            console.print({"events_rows": int(len(ev_rows)), "output": str(events_out), "alias": str(events_out_alias)})
        else:
            console.print("No events found for target date (ET)", style="yellow")
    except Exception as e:
        console.print(f"Events snapshot write failed: {e}", style="yellow")


@cli.command("odds-snapshots-props")
@click.option("--date", "date_str", type=str, required=False, help="Target date YYYY-MM-DD; defaults to the current US local slate date")
@click.option("--api-key", envvar="ODDS_API_KEY", type=str, required=False, help="OddsAPI key (or set env ODDS_API_KEY)")
@click.option("--regions", type=str, required=False, default="us", show_default=True, help="OddsAPI regions (e.g. us, us2)")
@click.option("--bookmakers", type=str, required=False, default=None, help="Comma-separated bookmaker keys to include; defaults to FanDuel, DraftKings, BetMGM, bet365")
@click.option(
    "--markets",
    type=str,
    required=False,
    default="",
    show_default=False,
    help="Comma-separated OddsAPI market keys. Defaults to our full supported player_* set.",
)
def odds_snapshots_props_cmd(date_str: str | None, api_key: str | None, regions: str, bookmakers: str | None, markets: str):
    """Write a per-date OddsAPI player props snapshot under data/raw.

    Output (used by props-edges when --use-saved):
    - data/raw/odds_nba_player_props_<date>.csv

        Also writes (best-effort):
        - data/raw/odds_nba_player_props_opening_<date>.parquet (or .csv fallback)
            Captures the first non-empty snapshot of the day for opening-line tracking.
        - data/raw/odds_nba_player_props_history_<date>.csv
            Append-only raw snapshot history (no in-memory rewrite).
    """
    console.rule("Odds Snapshots (player props)")
    from .odds_api import OddsApiConfig, fetch_player_props_current

    try:
        target_date = _resolve_target_date(date_str)
    except Exception:
        console.print("Invalid --date (YYYY-MM-DD)", style="red")
        return

    if not api_key:
        api_key = _load_dotenv_key("ODDS_API_KEY")
    if not api_key:
        console.print("Provide --api-key, set ODDS_API_KEY env, or add to .env at repo root.", style="red")
        return

    mkts_list = [m.strip() for m in str(markets or "").split(",") if m.strip()]
    bookmaker_keys = resolve_player_prop_bookmakers(bookmakers)
    cfg = OddsApiConfig(
        api_key=api_key,
        regions=(regions or "us").strip() or "us",
        bookmakers=player_prop_bookmakers_csv(bookmakers),
    )

    try:
        df = fetch_player_props_current(cfg, pd.to_datetime(str(target_date)), markets=(mkts_list or None), verbose=False)
    except Exception as e:
        console.print(f"Player props snapshot failed: {e}", style="yellow")
        return

    if df is not None and not df.empty:
        df = filter_player_prop_bookmakers_df(df, bookmakers)

    out = paths.data_raw / f"odds_nba_player_props_{target_date}.csv"
    existing_snapshot_has_rows = False
    if out.exists():
        try:
            existing_preview = pd.read_csv(out, nrows=1)
            existing_snapshot_has_rows = bool(isinstance(existing_preview, pd.DataFrame) and not existing_preview.empty)
        except Exception:
            existing_snapshot_has_rows = False
    if df is None or df.empty:
        if existing_snapshot_has_rows:
            console.print(
                {
                    "warning": "player_props_snapshot_empty",
                    "preserved_existing_snapshot": str(out),
                },
                style="yellow",
            )
            return
        raise click.ClickException(
            f"No player props fetched for {target_date} and no non-empty same-day snapshot exists to preserve."
        )
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        (df if df is not None else pd.DataFrame()).to_csv(out, index=False)
    except Exception as e:
        console.print(f"Failed writing snapshot: {e}", style="yellow")
        return

    # Persist an "opening" snapshot per prop outcome.
    # IMPORTANT: markets can appear throughout the day; keep this file up-to-date by
    # adding any newly-seen (event, book, market, player, side) keys while preserving
    # the earliest snapshot for keys already present.
    try:
        cur = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        if cur is not None and not cur.empty:
            import re
            import unicodedata

            open_pq = paths.data_raw / f"odds_nba_player_props_opening_{target_date}.parquet"
            open_csv = paths.data_raw / f"odds_nba_player_props_opening_{target_date}.csv"

            open_cols = [
                "snapshot_ts",
                "event_id",
                "commence_time",
                "bookmaker",
                "market",
                "outcome_name",
                "player_name",
                "point",
                "price",
            ]

            def _norm_name_key(s: str) -> str:
                s = (s or "").strip().upper()
                try:
                    s = unicodedata.normalize("NFKD", s)
                    s = "".join(ch for ch in s if not unicodedata.combining(ch))
                    s = s.encode("ascii", "ignore").decode("ascii")
                except Exception:
                    pass
                s = s.replace("-", " ")
                if "(" in s:
                    s = s.split("(", 1)[0]
                s = re.sub(r"[^A-Z0-9\s]", "", s)
                s = re.sub(r"\s+", " ", s).strip()
                for suf in (" JR", " SR", " II", " III", " IV", " V"):
                    if s.endswith(suf):
                        s = s[: -len(suf)].strip()
                return s

            def _map_side(x: str) -> str | None:
                u = str(x).upper()
                if "OVER" in u:
                    return "OVER"
                if "UNDER" in u:
                    return "UNDER"
                if u in ("YES", "Y"):
                    return "YES"
                if u in ("NO", "N"):
                    return "NO"
                return None

            def _select_canonical_open(df_in: pd.DataFrame) -> pd.DataFrame:
                try:
                    tmp = df_in[[c for c in open_cols if c in df_in.columns]].copy()
                    if tmp.empty:
                        return tmp
                    tmp["_name_key"] = tmp.get("player_name").astype(str).map(_norm_name_key)
                    tmp["_side"] = tmp.get("outcome_name").astype(str).map(_map_side)
                    tmp["_price_num"] = pd.to_numeric(tmp.get("price"), errors="coerce")
                    # Implied probability from American odds
                    ip = pd.Series(np.nan, index=tmp.index, dtype="float64")
                    m_pos = tmp["_price_num"] > 0
                    if m_pos.any():
                        ip.loc[m_pos] = 100.0 / (tmp.loc[m_pos, "_price_num"] + 100.0)
                    m_neg = tmp["_price_num"] < 0
                    if m_neg.any():
                        ip.loc[m_neg] = (-tmp.loc[m_neg, "_price_num"]) / ((-tmp.loc[m_neg, "_price_num"]) + 100.0)
                    target_ip = 110.0 / (110.0 + 100.0)  # -110 break-even
                    tmp["_metric"] = (ip - float(target_ip)).abs().fillna(1e9)
                    keys = ["event_id", "bookmaker", "market", "_name_key", "_side"]
                    tmp = tmp[
                        tmp.get("event_id").notna()
                        & tmp.get("bookmaker").notna()
                        & tmp.get("market").notna()
                        & tmp.get("_name_key").notna()
                        & tmp.get("_side").notna()
                    ].copy()
                    if tmp.empty:
                        return tmp[[c for c in open_cols if c in tmp.columns]].copy()
                    try:
                        idx = tmp.groupby(keys)["_metric"].idxmin()
                        out = tmp.loc[idx, [c for c in open_cols if c in tmp.columns]].copy()
                    except Exception:
                        out = tmp[[c for c in open_cols if c in tmp.columns]].copy()
                    return out
                except Exception:
                    return pd.DataFrame()

            # Canonical open candidates from *this* snapshot
            cur_open = _select_canonical_open(cur)

            # Load existing opening snapshot (if any) and canonicalize it as well.
            # Be defensive: if parquet exists but is unreadable (missing engine, partial write, etc.),
            # fall back to CSV if available so we don't accidentally reset opening state.
            existing_open = pd.DataFrame()
            if open_pq.exists():
                try:
                    existing_open = pd.read_parquet(open_pq, columns=[c for c in open_cols])
                except Exception:
                    existing_open = pd.DataFrame()
            if (existing_open is None) or (not isinstance(existing_open, pd.DataFrame)) or existing_open.empty:
                if open_csv.exists():
                    try:
                        existing_open = pd.read_csv(open_csv, usecols=[c for c in open_cols])
                    except Exception:
                        existing_open = pd.DataFrame()
            existing_open = _select_canonical_open(existing_open) if isinstance(existing_open, pd.DataFrame) and (not existing_open.empty) else pd.DataFrame(columns=open_cols)

            # Append only newly-seen keys
            def _key_series(df_k: pd.DataFrame) -> pd.Series:
                try:
                    if df_k is None or (not isinstance(df_k, pd.DataFrame)) or df_k.empty:
                        return pd.Series([], dtype=str)
                    idx = df_k.index
                    eid = df_k["event_id"].astype(str) if "event_id" in df_k.columns else pd.Series("", index=idx, dtype=str)
                    bk = df_k["bookmaker"].astype(str) if "bookmaker" in df_k.columns else pd.Series("", index=idx, dtype=str)
                    mk = df_k["market"].astype(str) if "market" in df_k.columns else pd.Series("", index=idx, dtype=str)
                    pn = df_k["player_name"].astype(str) if "player_name" in df_k.columns else pd.Series("", index=idx, dtype=str)
                    on = df_k["outcome_name"].astype(str) if "outcome_name" in df_k.columns else pd.Series("", index=idx, dtype=str)
                    nm = pn.map(_norm_name_key)
                    sd = on.map(_map_side)
                    return eid + "|" + bk + "|" + mk + "|" + nm.astype(str) + "|" + sd.astype(str)
                except Exception:
                    return pd.Series([], dtype=str)

            ex_keys = set(_key_series(existing_open).dropna().astype(str).tolist()) if not existing_open.empty else set()
            if not cur_open.empty:
                cur_keys = _key_series(cur_open)
                add_mask = ~cur_keys.astype(str).isin(ex_keys)
                add_rows = cur_open.loc[add_mask].copy() if bool(add_mask.any()) else pd.DataFrame(columns=open_cols)
            else:
                add_rows = pd.DataFrame(columns=open_cols)

            combined = pd.concat([existing_open, add_rows], ignore_index=True, sort=False)
            combined = combined[[c for c in open_cols if c in combined.columns]].copy()
            if not combined.empty:
                wrote_open = None
                wrote_pq = False
                try:
                    combined.to_parquet(open_pq, index=False)
                    wrote_open = open_pq
                    wrote_pq = True
                except Exception:
                    combined.to_csv(open_csv, index=False)
                    wrote_open = open_csv
                    # Ensure we don't leave a stale parquet file that would be preferred at read time
                    try:
                        if open_pq.exists():
                            open_pq.unlink()
                    except Exception:
                        pass
                # Optional: keep CSV around even when parquet works (useful for quick inspection)
                if wrote_pq:
                    try:
                        combined.to_csv(open_csv, index=False)
                    except Exception:
                        pass
                if wrote_open is not None:
                    console.print({"opening_rows": int(len(combined)), "opening_output": str(wrote_open)})
    except Exception as e:
        console.print(f"Opening snapshot write skipped: {e}", style="yellow")

    # Append to a per-date history file (append-only) so we can audit movement over time.
    # IMPORTANT: do not read/concat/rewrite the full file here (keeps memory bounded on 512MB deploys).
    try:
        cur = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        if cur is not None and not cur.empty:
            hist_csv = paths.data_raw / f"odds_nba_player_props_history_{target_date}.csv"
            hist_csv.parent.mkdir(parents=True, exist_ok=True)
            hist_cols = [
                "snapshot_ts",
                "event_id",
                "commence_time",
                "bookmaker",
                "market",
                "outcome_name",
                "player_name",
                "point",
                "price",
            ]
            cur_hist = cur[[c for c in hist_cols if c in cur.columns]].copy()
            if hist_csv.exists():
                cur_hist.to_csv(hist_csv, mode="a", header=False, index=False)
            else:
                cur_hist.to_csv(hist_csv, index=False)
            console.print({"history_appended_rows": int(len(cur_hist)), "history_output": str(hist_csv)})
    except Exception as e:
        console.print(f"History append skipped: {e}", style="yellow")

    rows = 0 if df is None else int(len(df))
    markets_n = 0
    try:
        markets_n = int(df["market"].nunique()) if (df is not None and ("market" in df.columns)) else 0
    except Exception:
        markets_n = 0
    books_n = 0
    try:
        books_n = int(df["bookmaker"].nunique()) if (df is not None and ("bookmaker" in df.columns)) else 0
    except Exception:
        books_n = 0
    console.print({
        "rows": rows,
        "markets": markets_n,
        "books": books_n,
        "bookmakers": (list(bookmaker_keys) if bookmaker_keys else "all"),
        "output": str(out),
    })

    console.print({
        "date": str(target_date),
        "rows": rows,
        "markets": markets_n,
        "bookmakers": books_n,
        "output": str(out),
    })

@cli.command("train-pbp-markets")
@click.option("--start", type=str, required=False, help="Optional start date YYYY-MM-DD to fetch PBP if needed")
@click.option("--end", type=str, required=False, help="Optional end date YYYY-MM-DD to fetch PBP if needed")
def train_pbp_markets_cmd(start: str | None, end: str | None):
    """Train ONNX-exportable models for:
    - Tip winner (home_won_tip probability)
    - First basket scorer (per-player scoring model; normalized at inference)
    - Total 3s made in first 3 minutes (regression + Poisson approx)

    Uses existing processed PBP/boxscores when available. If --start/--end are provided,
    will attempt to backfill missing PBP for that range before training.
    """
    console.rule("Train PBP-derived Markets")
    # Best-effort backfill if requested
    if start and end:
        try:
            from .pbp import backfill_pbp
            _ = backfill_pbp(start, end, only_final=True, rate_delay=0.35)
        except Exception as e:
            console.print(f"PBP backfill failed: {e}", style="yellow")
    try:
        arts = train_all_pbp_markets()
        out = {
            k: {"model": str(v.model_path), "onnx": (str(v.onnx_path) if v.onnx_path else None)}
            for k, v in arts.items()
        }
        console.print(out)
    except Exception as e:
        console.print(f"Training failed: {e}", style="red")

@cli.command("predict-pbp-markets")
@click.option("--date", "date_str", type=str, required=True, help="Target date YYYY-MM-DD (games on this day)")
def predict_pbp_markets_cmd(date_str: str):
    """Generate predictions for PBP-derived markets for a given date.

    Outputs under data/processed:
    - tip_winner_probs_<date>.csv
    - first_basket_probs_<date>.csv
    - early_threes_<date>.csv
    """
    console.rule("Predict PBP-derived Markets")
    try:
        tip = predict_tip_for_date(date_str)
        fb = predict_first_basket_for_date(date_str)
        thr = predict_early_threes_for_date(date_str)
        console.print({
            "tip_rows": int(len(tip) if tip is not None else 0),
            "first_basket_rows": int(len(fb) if fb is not None else 0),
            "early_threes_rows": int(len(thr) if thr is not None else 0),
        })
    except Exception as e:
        console.print(f"Prediction failed: {e}", style="red")


@cli.command("backfill-pbp-markets")
@click.option("--start", "start_date", type=str, required=False, help="Start date YYYY-MM-DD; default = season start (Oct 1) of current season")
@click.option("--end", "end_date", type=str, required=False, help="End date YYYY-MM-DD; default = today (local)")
@click.option("--with-pbp", "with_pbp", is_flag=True, default=False, help="Also backfill PBP logs for the range (finals-only)")
def backfill_pbp_markets_cmd(start_date: str | None, end_date: str | None, with_pbp: bool):
    """Backfill PBP-derived market predictions over a date range.

    Writes per-day CSVs under data/processed for:
    - tip_winner_probs_<date>.csv
    - first_basket_probs_<date>.csv
    - early_threes_<date>.csv
    """
    console.rule("Backfill PBP-derived Markets")
    import datetime as _dt
    # Defaults
    today = _dt.date.today()
    if end_date is None:
        e = today
    else:
        try:
            e = pd.to_datetime(end_date).date()
        except Exception:
            console.print("Invalid --end (YYYY-MM-DD)", style="red"); return
    if start_date is None:
        # Compute season start: Oct 1 of season year (season year = year if >= Jul else year-1)
        yr = today.year
        if today.month < 7:
            yr -= 1
        s = _dt.date(yr, 10, 1)
    else:
        try:
            s = pd.to_datetime(start_date).date()
        except Exception:
            console.print("Invalid --start (YYYY-MM-DD)", style="red"); return
    if e < s:
        console.print("--end must be >= --start", style="red"); return
    # Ensure boxscores exist for the date range to build candidates
    try:
        _ = backfill_boxscores(str(s), str(e), only_final=True, rate_delay=0.35)
    except Exception as ex:
        console.print({"warning": f"boxscores backfill failed: {ex}"}, style="yellow")
    if with_pbp:
        try:
            _ = backfill_pbp(str(s), str(e), only_final=True, rate_delay=0.35)
        except Exception as ex:
            console.print({"warning": f"pbp backfill failed: {ex}"}, style="yellow")
    dates = list(pd.date_range(s, e, freq="D").date)
    total = {"tip": 0, "first_basket": 0, "early_threes": 0}
    for d in track(dates, description="Backfilling PBP markets"):
        ds = str(d)
        try:
            tip = predict_tip_for_date(ds)
            fb = predict_first_basket_for_date(ds)
            thr = predict_early_threes_for_date(ds)
            total["tip"] += 0 if tip is None else len(tip)
            total["first_basket"] += 0 if fb is None else len(fb)
            total["early_threes"] += 0 if thr is None else len(thr)
        except Exception as ex:
            console.print({"date": ds, "error": str(ex)}, style="yellow")
    console.print({
        "start": str(s),
        "end": str(e),
        "days": len(dates),
        "rows_tip": int(total["tip"]),
        "rows_first_basket": int(total["first_basket"]),
        "rows_early_threes": int(total["early_threes"]),
    })


@cli.command("export-game-cards")
@click.option("--date", "date_str", type=str, required=True, help="Target date YYYY-MM-DD")
def export_game_cards_cmd(date_str: str):
    """Export per-game cards for a date.

    Writes data/processed/game_cards_<date>.csv with columns:
    - date, game_id, home_team, visitor_team, commence_time
    - prob_home_tip
    - early_threes_expected, early_threes_prob_ge_1
    - first_basket_top5 ("TEAM: Player (p%)"; semicolon-separated)

    If reconciliation files are present for the date, the export also includes a full recap:
    - final scores + derived margin/total
    - ATS / OU result vs consensus lines (when odds available)
    - PBP markets reconciliation (tip/first-basket/early-threes)
    - a compact props actuals recap (team sums + top scorer)
    """
    console.rule("Export Game Cards")
    from .teams import to_tricode as _to_tri
    out_path = paths.data_processed / f"game_cards_{date_str}.csv"

    # Load odds for matchup framing (home/visitor and time) + market lines
    odds_path = paths.data_processed / f"game_odds_{date_str}.csv"
    if odds_path.exists():
        go = pd.read_csv(odds_path)
    else:
        go = pd.DataFrame(columns=["home_team","visitor_team","commence_time"])  # fallback blank

    # Build gameId mapping from boxscores_<date>
    gid_map = {}
    box_path = paths.data_processed / f"boxscores_{date_str}.csv"
    if box_path.exists():
        bs = pd.read_csv(box_path)
        if not bs.empty:
            # For each gameId, collect team tricodes/abbreviations
            gid_col = "gameId" if "gameId" in bs.columns else ("game_id" if "game_id" in bs.columns else None)
            team_col = None
            for c in ("teamTricode", "team_tricode", "TEAM_ABBREVIATION", "team_abbr", "team"):
                if c in bs.columns:
                    team_col = c
                    break
            if gid_col is not None and team_col is not None:
                grp = bs.groupby(gid_col)[team_col].unique().reset_index()
            else:
                grp = pd.DataFrame(columns=["gameId", "teamTricode"])  # fallback empty
            for _, r in grp.iterrows():
                game_id = str(r.get(gid_col or "gameId")) if pd.notna(r.get(gid_col or "gameId")) else None
                arr = r.get(team_col or "teamTricode")
                vals = arr.tolist() if hasattr(arr, "tolist") else (list(arr) if isinstance(arr, (list, tuple)) else [arr])
                teams = set(str(x) for x in vals if isinstance(x, str) and x)
                if game_id and len(teams) == 2:
                    key = tuple(sorted(list(teams)))
                    gid_map[key] = game_id

    # Fallback (pregame): derive (HOME,AWAY)->game_id map from NBA CDN (public)
    if not gid_map:
        try:
            import requests  # type: ignore
            from datetime import datetime as _dt, date as _date
            target = _dt.strptime(date_str, "%Y-%m-%d").date()
            today = _date.today()
            # a) Today's live scoreboard
            if target == today:
                u = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
                r = requests.get(u, headers={"Accept":"application/json","User-Agent":"nba-betting/1.0"}, timeout=10)
                if r.ok:
                    j = r.json() or {}
                    games = (j.get("scoreboard") or {}).get("games") or []
                    for g in games:
                        gid = str(g.get("gameId") or "").strip()
                        home = str((g.get("homeTeam") or {}).get("teamTricode") or "").upper()
                        away = str((g.get("awayTeam") or {}).get("teamTricode") or "").upper()
                        if gid and home and away:
                            key = tuple(sorted([home, away]))
                            gid_map[key] = gid
            # b) Season schedule (broad scan; avoid strict string-date equals in case of UTC/ET drift)
            if not gid_map:
                u = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"
                r = requests.get(u, headers={"Accept":"application/json","User-Agent":"nba-betting/1.0"}, timeout=15)
                if r.ok:
                    j = r.json() or {}
                    league = j.get("leagueSchedule") or {}
                    game_dates = league.get("gameDates") or []
                    for gd in game_dates:
                        for g in (gd.get("games") or []):
                            gid = str(g.get("gameId") or "").strip()
                            home = str(((g.get("homeTeam") or {}).get("teamTricode")) or "").upper()
                            away = str(((g.get("awayTeam") or {}).get("teamTricode")) or "").upper()
                            if not (gid and home and away):
                                continue
                            # Filter by ET calendar day using startTimeUTC if present; else fall back to gd.gameDate
                            try:
                                from datetime import datetime as _dt
                                from zoneinfo import ZoneInfo as _ZI
                                et = _ZI("US/Eastern")
                            except Exception:
                                et = None
                            ok_day = False
                            st = g.get("gameDateTimeUTC") or g.get("startTimeUTC") or g.get("startTimeUTCFormatted")
                            if st:
                                try:
                                    dt = pd.to_datetime(st, utc=True)
                                    ds = dt.tz_convert("US/Eastern").strftime("%Y-%m-%d") if et else dt.strftime("%Y-%m-%d")
                                    ok_day = (ds == date_str)
                                except Exception:
                                    ok_day = False
                            if not ok_day:
                                gd_str = str(gd.get("gameDate") or "").split("T")[0]
                                ok_day = (gd_str == date_str)
                            if ok_day:
                                key = tuple(sorted([home, away]))
                                gid_map[key] = gid
            # c) Legacy data.nba.net scoreboard for the specific date (robust for historical + today)
            if not gid_map:
                try:
                    ymd = date_str.replace("-", "")
                    u2 = f"https://data.nba.net/prod/v1/{ymd}/scoreboard.json"
                    r2 = requests.get(u2, headers={"Accept":"application/json","User-Agent":"nba-betting/1.0"}, timeout=15)
                    if r2.ok:
                        js = r2.json() or {}
                        games = (js.get("games") or [])
                        for g in games:
                            gid = str(g.get("gameId") or "").strip()
                            home = str(((g.get("hTeam") or {}).get("triCode")) or "").upper()
                            away = str(((g.get("vTeam") or {}).get("triCode")) or "").upper()
                            if gid and home and away:
                                key = tuple(sorted([home, away]))
                                gid_map[key] = gid
                except Exception:
                    pass
            # d) Local schedule helper as a final fallback (stable schema + ET dates)
            if not gid_map:
                try:
                    from .schedule import fetch_schedule_2025_26 as _fetch_sched
                    df = _fetch_sched()
                    if df is not None and not df.empty:
                        # date_est is a date; compare to date_str
                        ds = pd.to_datetime(date_str).date()
                        day = df[pd.to_datetime(df.get("date_est"), errors="coerce").dt.date == ds].copy()
                        for _, r in day.iterrows():
                            gid = str(r.get("game_id") or "").strip()
                            home = str(r.get("home_tricode") or "").upper()
                            away = str(r.get("away_tricode") or "").upper()
                            if gid and home and away:
                                key = tuple(sorted([home, away]))
                                gid_map[key] = gid
                except Exception:
                    pass
        except Exception:
            pass

    # Load PBP market outputs
    tip = None; fb = None; thr = None
    tip_path = paths.data_processed / f"tip_winner_probs_{date_str}.csv"
    if tip_path.exists():
        try:
            tip = pd.read_csv(tip_path)
            # Normalize game_id to zero-padded 10-char strings
            if tip is not None and "game_id" in tip.columns:
                tip["game_id"] = tip["game_id"].astype(str).str.replace(".0$", "", regex=True).str.replace("^nan$","", regex=True).str.replace("^None$","", regex=True).str.replace("^\s+$","", regex=True).str.zfill(10)
            if tip is not None and tip.empty:
                tip = None
        except Exception:
            tip = None
    fb_path = paths.data_processed / f"first_basket_probs_{date_str}.csv"
    if fb_path.exists():
        try:
            fb = pd.read_csv(fb_path)
            if fb is not None and "game_id" in fb.columns:
                fb["game_id"] = fb["game_id"].astype(str).str.replace(".0$", "", regex=True).str.replace("^nan$","", regex=True).str.replace("^None$","", regex=True).str.replace("^\s+$","", regex=True).str.zfill(10)
            if fb is not None and fb.empty:
                fb = None
        except Exception:
            fb = None
    thr_path = paths.data_processed / f"early_threes_{date_str}.csv"
    if thr_path.exists():
        try:
            thr = pd.read_csv(thr_path)
            if thr is not None and "game_id" in thr.columns:
                thr["game_id"] = thr["game_id"].astype(str).str.replace(".0$", "", regex=True).str.replace("^nan$","", regex=True).str.replace("^None$","", regex=True).str.replace("^\s+$","", regex=True).str.zfill(10)
            if thr is not None and thr.empty:
                thr = None
        except Exception:
            thr = None

    rows: list[dict] = []
    # If odds available, iterate matchups; else iterate game IDs from tip/thr/fb
    def _build_row(game_id: Optional[str], home: Optional[str], away: Optional[str], ctime: Optional[str]) -> dict:
        # Normalize game_id to 10-digit zero-padded string for consistent frontend merges
        gid_norm = None
        if game_id is not None and str(game_id).strip():
            gid_norm = str(game_id).strip()
            # Some sources may already be zero-padded; enforce 10-digit
            gid_norm = gid_norm.replace('.0', '') if gid_norm.endswith('.0') else gid_norm
            if gid_norm.isdigit():
                gid_norm = gid_norm.zfill(10)
        row = {"date": date_str, "game_id": gid_norm, "home_team": home, "visitor_team": away, "commence_time": ctime}
        # Attach tip
        if tip is not None and game_id is not None:
            gid_norm = str(game_id).strip()
            gid_norm = gid_norm.zfill(10)
            trow = tip[tip["game_id"].astype(str).str.zfill(10) == gid_norm]
            if not trow.empty and "prob_home_tip" in trow.columns:
                row["prob_home_tip"] = float(trow.iloc[0]["prob_home_tip"])
        # Attach early threes
        if thr is not None and game_id is not None:
            gid_norm = str(game_id).strip().zfill(10)
            h = thr[thr["game_id"].astype(str).str.zfill(10) == gid_norm]
            if not h.empty:
                row["early_threes_expected"] = float(h.iloc[0].get("expected_threes_0_3", h.iloc[0].get("threes_0_3_pred", np.nan)))
                row["early_threes_prob_ge_1"] = float(h.iloc[0].get("prob_ge_1", np.nan))
        # Attach first basket top5
        if fb is not None and game_id is not None:
            gid_norm = str(game_id).strip().zfill(10)
            sub = fb[fb["game_id"].astype(str).str.zfill(10) == gid_norm].copy()
            if not sub.empty:
                sub = sub.sort_values("prob_first_basket", ascending=False).head(5)
                parts = []
                for _, r in sub.iterrows():
                    t = str(r.get("team","")); p = str(r.get("player_name","")); pr = float(r.get("prob_first_basket", 0))*100.0
                    parts.append(f"{t}: {p} ({pr:.1f}%)")
                row["first_basket_top5"] = "; ".join(parts)
        return row

    # Preload schedule day for a last-resort gid lookup by tricodes
    _sched_day = None
    try:
        from .schedule import fetch_schedule_2025_26 as _fetch_sched
        _df_sched = _fetch_sched()
        if _df_sched is not None and not _df_sched.empty:
            ds = pd.to_datetime(date_str).date()
            _sched_day = _df_sched[pd.to_datetime(_df_sched.get("date_est"), errors="coerce").dt.date == ds].copy()
    except Exception:
        _sched_day = None

    if go is not None and not go.empty and {"home_team","visitor_team"}.issubset(go.columns):
        # Normalize to tricodes for matching
        def _tri(x):
            try:
                return _to_tri(str(x))
            except Exception:
                return str(x)
        go = go.copy()
        go["home_tri"] = go["home_team"].map(_tri)
        go["away_tri"] = go["visitor_team"].map(_tri)
        for _, r in go.iterrows():
            home = r.get("home_team"); away = r.get("visitor_team"); ctime = r.get("commence_time")
            key = tuple(sorted([str(r.get("home_tri")), str(r.get("away_tri"))]))
            gid = gid_map.get(key)
            if (not gid) and (_sched_day is not None and not _sched_day.empty):
                try:
                    tri_home = str(r.get("home_tri") or "").upper()
                    tri_away = str(r.get("away_tri") or "").upper()
                    cand = _sched_day[((_sched_day["home_tricode"].astype(str).str.upper()==tri_home) & (_sched_day["away_tricode"].astype(str).str.upper()==tri_away))
                                      | ((_sched_day["home_tricode"].astype(str).str.upper()==tri_away) & (_sched_day["away_tricode"].astype(str).str.upper()==tri_home))]
                    if not cand.empty:
                        gid = str(cand.iloc[0].get("game_id") or "").strip()
                except Exception:
                    pass
            row = _build_row(gid, home, away, ctime)
            # Carry through odds/lines (when available)
            for c in (
                "home_ml",
                "away_ml",
                "home_spread",
                "away_spread",
                "home_spread_price",
                "away_spread_price",
                "total",
                "total_over_price",
                "total_under_price",
                "books_count",
                "bookmaker",
            ):
                if c in go.columns:
                    try:
                        row[c] = r.get(c)
                    except Exception:
                        pass
            rows.append(row)
    else:
        # Fallback: derive from tip/first basket files (game_id only)
        gid_set = set()
        for df in (tip, thr, fb):
            if df is not None and not df.empty and "game_id" in df.columns:
                gid_set.update(str(x) for x in df["game_id"].dropna().unique().tolist())
        for gid in sorted(gid_set):
            rows.append(_build_row(gid, None, None, None))

    cards = pd.DataFrame(rows)
    # Post-attach PBP fields via merges to ensure no row misses due to per-row attach edge cases
    try:
        if not cards.empty:
            # Normalize home/away tricodes for stable joins
            def _tri_safe(x):
                try:
                    if x is None or pd.isna(x):
                        return None
                    s = str(x).strip()
                    if not s:
                        return None
                    return _to_tri(s)
                except Exception:
                    try:
                        return str(x).strip().upper()
                    except Exception:
                        return None

            cards["home_tri"] = cards.get("home_team").map(_tri_safe)
            cards["away_tri"] = cards.get("visitor_team").map(_tri_safe)

            cards["gid10"] = cards.get("game_id").astype(str).str.replace(".0$","", regex=True).str.replace("^nan$","", regex=True).str.replace("^None$","", regex=True).str.replace("^\\s+$","", regex=True).str.zfill(10)
            # Tip merge
            if tip is not None and not tip.empty and {"game_id","prob_home_tip"}.issubset(set(tip.columns)):
                tp = tip.copy()
                tp["gid10"] = tp["game_id"].astype(str).str.replace(".0$","", regex=True).str.zfill(10)
                cards = cards.merge(tp[["gid10","prob_home_tip"]], on="gid10", how="left")
            # Early threes merge
            if thr is not None and not thr.empty and "game_id" in thr.columns:
                th = thr.copy()
                th["gid10"] = th["game_id"].astype(str).str.replace(".0$","", regex=True).str.zfill(10)
                # Prefer expected_threes_0_3 if present; else threes_0_3_pred
                exp_col = "expected_threes_0_3" if "expected_threes_0_3" in th.columns else ("threes_0_3_pred" if "threes_0_3_pred" in th.columns else None)
                if exp_col is not None:
                    th = th[["gid10", exp_col, "prob_ge_1"]].rename(columns={exp_col:"early_threes_expected", "prob_ge_1":"early_threes_prob_ge_1"})
                    cards = cards.merge(th, on="gid10", how="left")
            # First basket top5 aggregation
            if fb is not None and not fb.empty and {"game_id","team","player_name","prob_first_basket"}.issubset(set(fb.columns)):
                f = fb.copy()
                f["gid10"] = f["game_id"].astype(str).str.replace(".0$","", regex=True).str.zfill(10)
                f = f.sort_values(["gid10","prob_first_basket"], ascending=[True, False])
                top5 = f.groupby("gid10").head(5).copy()
                top5["part"] = top5.apply(lambda r: f"{str(r.get('team',''))}: {str(r.get('player_name',''))} ({float(r.get('prob_first_basket',0))*100:.1f}%)", axis=1)
                agg = top5.groupby("gid10")["part"].apply(lambda s: "; ".join(s.tolist())).reset_index().rename(columns={"part":"first_basket_top5"})
                cards = cards.merge(agg, on="gid10", how="left")
            # Coalesce into clean canonical columns
            for to_col in ("prob_home_tip","early_threes_expected","early_threes_prob_ge_1","first_basket_top5"):
                cols = []
                if f"{to_col}_x" in cards.columns: cols.append(f"{to_col}_x")
                if f"{to_col}_y" in cards.columns: cols.append(f"{to_col}_y")
                if to_col in cards.columns: cols.append(to_col)
                if cols:
                    ser = None
                    for c in cols:
                        s = cards.get(c)
                        if ser is None:
                            ser = s
                        else:
                            try:
                                ser = ser.where(ser.notna(), s)
                            except Exception:
                                pass
                    if ser is not None:
                        cards[to_col] = ser
            # Drop suffix variants after coalescing
            drop_cols = [c for c in cards.columns if c.endswith("_x") or c.endswith("_y")]
            if drop_cols:
                try:
                    cards = cards.drop(columns=drop_cols)
                except Exception:
                    for c in drop_cols:
                        try:
                            cards = cards.drop(columns=[c])
                        except Exception:
                            pass
            # Drop helper key
            cards = cards.drop(columns=["gid10"]) 
    except Exception as ex:
        console.print({"warning": f"post-merge of PBP fields failed: {ex}"}, style="yellow")

    # If reconciliation outputs exist for this date, merge in a full recap
    try:
        if not cards.empty:
            cards["gid10"] = cards.get("game_id").astype(str).str.replace(".0$","", regex=True).str.replace("^nan$","", regex=True).str.replace("^None$","", regex=True).str.replace("^\\s+$","", regex=True).str.zfill(10)
            # 1) Games reconciliation (final scores, errors)
            rg_path = paths.data_processed / f"recon_games_{date_str}.csv"
            if rg_path.exists():
                rg = pd.read_csv(rg_path)
                # Normalize + rename to avoid ambiguity
                rg = rg.copy()
                if "home_tri" in rg.columns:
                    rg["home_tri"] = rg["home_tri"].astype(str).str.upper()
                if "away_tri" in rg.columns:
                    rg["away_tri"] = rg["away_tri"].astype(str).str.upper()
                rg = rg.rename(
                    columns={
                        "home_pts": "final_home_pts",
                        "visitor_pts": "final_visitor_pts",
                        "total_actual": "final_total_pts",
                        "actual_margin": "final_margin",
                        "margin_error": "margin_error",
                        "total_error": "total_error",
                    }
                )
                keep = [
                    c
                    for c in (
                        "date",
                        "home_tri",
                        "away_tri",
                        "final_home_pts",
                        "final_visitor_pts",
                        "final_total_pts",
                        "final_margin",
                        "pred_margin",
                        "pred_total",
                        "margin_error",
                        "total_error",
                    )
                    if c in rg.columns
                ]
                if {"date", "home_tri", "away_tri"}.issubset(set(keep)):
                    cards = cards.merge(rg[keep], on=["date", "home_tri", "away_tri"], how="left")

            # 2) PBP markets reconciliation
            pbpr_path = paths.data_processed / f"pbp_reconcile_{date_str}.csv"
            if pbpr_path.exists():
                pr = pd.read_csv(pbpr_path)
                pr = pr.copy()
                pr["gid10"] = pr["game_id"].astype(str).str.replace(".0$", "", regex=True).str.replace("^nan$", "", regex=True).str.replace("^None$", "", regex=True).str.replace("^\\s+$", "", regex=True).str.zfill(10)
                # Some runs may produce duplicate rows per game; keep the first per gid10
                try:
                    pr = pr.drop_duplicates(subset=["gid10"], keep="first")
                except Exception:
                    pass
                # Only merge non-key fields to avoid clobbering home/visitor names
                drop = {"date", "game_id", "home_team", "visitor_team"}
                cols = [c for c in pr.columns if c not in drop]
                if "gid10" in pr.columns:
                    cols = ["gid10"] + [c for c in cols if c != "gid10"]
                pr_m = pr[cols].copy() if cols else pr[["gid10"]].copy()
                # Prefix to avoid collisions with pregame PBP fields
                try:
                    ren = {c: f"pbp_{c}" for c in pr_m.columns if c != "gid10"}
                    pr_m = pr_m.rename(columns=ren)
                except Exception:
                    pass
                cards = cards.merge(pr_m, on="gid10", how="left")

            # 3) Props reconciliation recap (actuals per game)
            rp_path = paths.data_processed / f"recon_props_{date_str}.csv"
            if rp_path.exists():
                rp = pd.read_csv(rp_path)
                if rp is not None and not rp.empty and "game_id" in rp.columns:
                    rp = rp.copy()
                    rp["gid10"] = rp["game_id"].astype(str).str.replace(".0$", "", regex=True).str.replace("^nan$", "", regex=True).str.replace("^None$", "", regex=True).str.replace("^\\s+$", "", regex=True).str.zfill(10)
                    if "team_abbr" in rp.columns:
                        rp["team_abbr"] = rp["team_abbr"].astype(str).str.upper()
                    for c in ("pts", "reb", "ast", "threes", "pra"):
                        if c in rp.columns:
                            rp[c] = pd.to_numeric(rp[c], errors="coerce")
                    # Totals and top scorer
                    gsum = rp.groupby("gid10", as_index=False).agg(
                        props_actual_rows_n=("player_id", "size"),
                        props_actual_players_n=("player_id", "nunique"),
                    )
                    cards = cards.merge(gsum, on="gid10", how="left")
                    # Team sums (attach to home/away)
                    if {"team_abbr", "gid10"}.issubset(set(rp.columns)):
                        agg_cols = {c: "sum" for c in ("pts", "reb", "ast", "threes", "pra") if c in rp.columns}
                        if agg_cols:
                            team_sum = rp.groupby(["gid10", "team_abbr"], as_index=False).agg(agg_cols)
                            home_sum = team_sum.rename(columns={"team_abbr": "home_tri"})
                            home_sum = home_sum.rename(columns={c: f"props_home_{c}_sum" for c in agg_cols.keys()})
                            cards = cards.merge(home_sum, on=["gid10", "home_tri"], how="left")
                            away_sum = team_sum.rename(columns={"team_abbr": "away_tri"})
                            away_sum = away_sum.rename(columns={c: f"props_away_{c}_sum" for c in agg_cols.keys()})
                            cards = cards.merge(away_sum, on=["gid10", "away_tri"], how="left")
                    # Top scorer by points
                    if {"player_name", "pts"}.issubset(set(rp.columns)):
                        tmp = rp.dropna(subset=["gid10", "pts"]).copy()
                        if not tmp.empty:
                            tmp = tmp.sort_values(["gid10", "pts"], ascending=[True, False])
                            top = tmp.groupby("gid10", as_index=False).head(1)
                            top = top[["gid10", "player_name", "team_abbr", "pts"]].rename(
                                columns={
                                    "player_name": "props_top_pts_player",
                                    "team_abbr": "props_top_pts_team",
                                    "pts": "props_top_pts",
                                }
                            )
                            cards = cards.merge(top, on="gid10", how="left")

            # 4) Derived results vs market lines (ATS/OU)
            for c in ("home_spread", "total", "final_home_pts", "final_visitor_pts", "final_total_pts", "final_margin"):
                if c in cards.columns:
                    cards[c] = pd.to_numeric(cards[c], errors="coerce")
            if {"final_margin", "home_spread"}.issubset(set(cards.columns)):
                v = cards["final_margin"] + cards["home_spread"]
                cards["ats_home_margin"] = v
                cards["ats_home_result"] = np.where(v > 0, "W", np.where(v < 0, "L", "P"))
            if {"final_total_pts", "total"}.issubset(set(cards.columns)):
                dv = cards["final_total_pts"] - cards["total"]
                cards["ou_margin"] = dv
                cards["ou_result"] = np.where(dv > 0, "O", np.where(dv < 0, "U", "P"))

            # Drop helper
            try:
                cards = cards.drop(columns=["gid10"])
            except Exception:
                pass
    except Exception as ex:
        console.print({"warning": f"reconciliation recap merge failed: {ex}"}, style="yellow")

    cards.to_csv(out_path, index=False)
    console.print({"rows": int(len(cards)), "output": str(out_path)})


@cli.command("reconcile-pbp-markets")
@click.option("--date", "date_str", type=str, required=True, help="Target date YYYY-MM-DD to reconcile PBP-derived markets")
def reconcile_pbp_markets_cmd(date_str: str):
    """Reconcile PBP-derived markets (tip, first-basket, early-threes) for a date.

    Writes per-game reconciliation to data/processed/pbp_reconcile_<date>.csv and prints summary metrics.
    """
    console.rule("Reconcile PBP-derived Markets")
    try:
        _ = pd.to_datetime(date_str).date()
    except Exception:
        console.print("Invalid --date (YYYY-MM-DD)", style="red"); return

    # Load predictions (safe reader to handle blank files)
    def _read_csv_safe(path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    tip_path = paths.data_processed / f"tip_winner_probs_{date_str}.csv"
    fb_path = paths.data_processed / f"first_basket_probs_{date_str}.csv"
    thr_path = paths.data_processed / f"early_threes_{date_str}.csv"
    tip = _read_csv_safe(tip_path) if tip_path.exists() else pd.DataFrame()
    fb = _read_csv_safe(fb_path) if fb_path.exists() else pd.DataFrame()
    thr = _read_csv_safe(thr_path) if thr_path.exists() else pd.DataFrame()

    # Build PBP map by game_id using combined file if available; else per-game files by discovered IDs
    pbp_map: dict[str, pd.DataFrame] = {}
    pbp_comb = paths.data_processed / f"pbp_{date_str}.csv"
    if pbp_comb.exists():
        try:
            df = pd.read_csv(pbp_comb)
            if "game_id" in df.columns and not df.empty:
                for gid, grp in df.groupby("game_id"):
                    key_raw = str(gid).strip()
                    # Accept only numeric gameIds to avoid 'NA' and placeholders
                    if not key_raw or not key_raw.isdigit():
                        continue
                    pbp_map[key_raw.zfill(10)] = grp.copy()
        except Exception:
            pbp_map = {}
    if not pbp_map:
        # Discover game IDs from any available predictions or CDN schedule
        gids = set()
        for df in (tip, fb, thr):
            if not df.empty and "game_id" in df.columns:
                gids.update(str(x) for x in df["game_id"].dropna().astype(str).tolist())
        if not gids:
            try:
                from .pbp_markets import _game_ids_for_date as _ids
                gids = set(_ids(date_str))
            except Exception:
                gids = set()
        dpg = paths.data_processed / "pbp"
        for gid in gids:
            # Only consider numeric gids
            gid_s = str(gid).strip()
            if not gid_s.isdigit():
                continue
            gzp = gid_s.zfill(10)
            for name in (f"pbp_{gid_s}.csv", f"pbp_{gzp}.csv"):
                fpath = dpg / name
                if fpath.exists():
                    try:
                        pbp_map[gzp] = pd.read_csv(fpath)
                        break
                    except Exception:
                        continue

    # Home/away map for this date (for tip winner mapping)
    _gid_to_homeaway: dict[str, tuple[str,str]] = {}
    try:
        from .pbp_markets import _gid_team_map_for_date as _map
        m = _map(date_str) or {}
        _gid_to_homeaway = {str(k): (v[0], v[1]) for k, v in m.items()}
    except Exception:
        _gid_to_homeaway = {}

    # Helpers
    def _count_early_threes_q1(gdf: pd.DataFrame) -> int:
        if gdf is None or gdf.empty:
            return 0
        desc_cols = _pbp_desc_cols(gdf)
        c_per = "PERIOD" if "PERIOD" in gdf.columns else ("period" if "period" in gdf.columns else None)
        tmp = gdf.copy()
        if c_per:
            try:
                tmp = tmp[tmp[c_per] == 1]
            except Exception:
                pass
        # Prefer actionNumber ordering; else compute elapsed via shared parser
        if "actionNumber" in gdf.columns:
            try:
                tmp = tmp.sort_values("actionNumber", ascending=True)
            except Exception:
                pass
        # else: leave original order; we'll guard with per-row elapsed checks
        cnt = 0
        for _, r in tmp.iterrows():
            # Compute elapsed using shared helper
            try:
                from .pbp_markets import _to_sec_left as _sec
            except Exception:
                _sec = None
            t = r.get("PCTIMESTRING") or r.get("clock") or r.get("time")
            sec_left = _sec(t) if _sec else None
            if sec_left is None:
                continue
            elapsed = 12*60 - sec_left
            if elapsed is None or elapsed > 180:
                continue
            text = " ".join([str(r.get(c, "")) for c in desc_cols]).lower()
            # Handle both textual and structured formats
            made_three_text = ("3pt" in text) and ("made" in text or "makes" in text or "jump shot" in text)
            made_three_struct = False
            try:
                if (str(r.get("shotResult")).lower() == "made") and int(r.get("shotValue") or 0) == 3:
                    made_three_struct = True
            except Exception:
                made_three_struct = False
            if made_three_text or made_three_struct:
                cnt += 1
        return int(cnt)

    # Build reconciliation rows
    rows: list[dict] = []
    # Build union of game_ids from predictions and pbp_map
    gid_set = set()
    for df in (tip, fb, thr):
        if not df.empty and "game_id" in df.columns:
            gid_set.update(str(x) for x in df["game_id"].dropna().astype(str).tolist())
    gid_set.update(pbp_map.keys())

    for gid in sorted(gid_set):
        gid_key = str(gid)
        gid_norm = gid_key.zfill(10) if gid_key.isdigit() else gid_key
        rec: dict = {"date": date_str, "game_id": gid_norm}
        # Team tricodes if available
        home, away = _gid_to_homeaway.get(gid_key) or _gid_to_homeaway.get(gid_norm) or (None, None)
        # If not available, attempt to derive from the game's PBP (location h/v)
        gdf = None
        for _k in (gid_key, gid_norm):
            if _k in pbp_map:
                gdf = pbp_map[_k]
                break
        if (home is None or away is None) and gdf is not None and not gdf.empty:
            try:
                if {"teamTricode","location"}.issubset(set(gdf.columns)):
                    tri_h = gdf[gdf["location"].astype(str).str.lower()=="h"].get("teamTricode").dropna()
                    tri_v = gdf[gdf["location"].astype(str).str.lower()=="v"].get("teamTricode").dropna()
                    h0 = str(tri_h.iloc[0]).upper() if len(tri_h)>0 else None
                    v0 = str(tri_v.iloc[0]).upper() if len(tri_v)>0 else None
                    if h0 and v0:
                        home, away = h0, v0
            except Exception:
                pass
        rec.update({"home_team": home, "visitor_team": away})

        # Tip reconciliation
        p_home = None
        if not tip.empty:
            m = tip[tip["game_id"].astype(str).str.zfill(10) == gid_norm]
            if not m.empty and "prob_home_tip" in m.columns:
                try:
                    p_home = float(m.iloc[0]["prob_home_tip"])
                except Exception:
                    p_home = None
        rec["tip_prob_home"] = p_home
        outcome = None
        if gdf is not None and not gdf.empty:
            ev = _pbp_jump_ball_event(gdf)
            if ev:
                winner_text = (ev.get("winner_text") or "").strip()
                if winner_text:
                    # Try to infer the winner's team directly from PBP player names
                    try:
                        name_cols = [c for c in ("playerName","PLAYER1_NAME","player1_name") if c in gdf.columns]
                        team_cols = [c for c in ("teamTricode","PLAYER1_TEAM_ABBREVIATION","team_abbr") if c in gdf.columns]
                        t_winner = None
                        if name_cols and team_cols:
                            nl = winner_text.lower()
                            for _, rr in gdf.iterrows():
                                nm = None
                                for c in name_cols:
                                    nv = rr.get(c)
                                    if isinstance(nv, str) and nv.strip():
                                        nm = nv; break
                                if not nm:
                                    continue
                                if nl in str(nm).lower():
                                    tv = None
                                    for tc in team_cols:
                                        tv = rr.get(tc)
                                        if isinstance(tv, str) and tv.strip():
                                            break
                                    if tv:
                                        t_winner = str(tv).upper()
                                        break
                        if t_winner and home and away:
                            if t_winner == str(home).upper():
                                outcome = 1.0
                            elif t_winner == str(away).upper():
                                outcome = 0.0
                    except Exception:
                        pass
        rec["tip_outcome_home"] = outcome
        if p_home is not None and outcome is not None:
            try:
                brier = (float(p_home) - float(outcome))**2
                import math
                logloss = -(float(outcome)*math.log(max(1e-9, float(p_home))) + (1-float(outcome))*math.log(max(1e-9, 1-float(p_home))))
            except Exception:
                brier = None; logloss = None
        else:
            brier = None; logloss = None
        rec["tip_brier"] = brier
        rec["tip_logloss"] = logloss

        # First basket reconciliation
        fb_sub = fb[fb["game_id"].astype(str).str.zfill(10) == gid_norm] if not fb.empty else pd.DataFrame()
        fb_top1_name = None; fb_top1_prob = None; fb_actual_name = None; fb_hit_top1 = None; fb_hit_top5 = None; fb_prob_actual = None
        if gdf is not None and not gdf.empty:
            ev = _pbp_first_fg_event(gdf)
            if ev:
                fb_actual_name = (ev.get("player_name") or "").strip()
        if not fb_sub.empty:
            sub = fb_sub.copy().sort_values("prob_first_basket", ascending=False)
            try:
                fb_top1_name = str(sub.iloc[0].get("player_name"))
                fb_top1_prob = float(sub.iloc[0].get("prob_first_basket"))
            except Exception:
                pass
            if fb_actual_name:
                name_l = fb_actual_name.lower()
                # Top-1
                fb_hit_top1 = name_l in str(fb_top1_name or "").lower()
                # Top-5
                hit5 = False; p_act = None
                for _, r in sub.head(5).iterrows():
                    if name_l in str(r.get("player_name","")) .lower():
                        hit5 = True; p_act = float(r.get("prob_first_basket", np.nan)); break
                fb_hit_top5 = hit5
                if p_act is None:
                    m2 = sub[sub["player_name"].astype(str).str.lower().str.contains(name_l)].head(1)
                    if not m2.empty:
                        try:
                            p_act = float(m2.iloc[0].get("prob_first_basket"))
                        except Exception:
                            p_act = None
                fb_prob_actual = p_act
        rec.update({
            "first_basket_top1_name": fb_top1_name,
            "first_basket_top1_prob": fb_top1_prob,
            "first_basket_actual_name": fb_actual_name,
            "first_basket_hit_top1": fb_hit_top1,
            "first_basket_hit_top5": fb_hit_top5,
            "first_basket_prob_actual": fb_prob_actual,
        })

        # Early threes reconciliation
        yhat = None; p_ge1 = None
        if not thr.empty:
            m = thr[thr["game_id"].astype(str).str.zfill(10) == gid_norm]
            if not m.empty:
                try:
                    yhat = float(m.iloc[0].get("expected_threes_0_3", m.iloc[0].get("threes_0_3_pred", np.nan)))
                except Exception:
                    yhat = None
                try:
                    p_ge1 = float(m.iloc[0].get("prob_ge_1", np.nan))
                except Exception:
                    p_ge1 = None
        actual_thr = None; err_thr = None; brier_ge1 = None
        if gdf is not None and not gdf.empty:
            actual_thr = _count_early_threes_q1(gdf)
            if yhat is not None:
                try:
                    err_thr = float(actual_thr) - float(yhat)
                except Exception:
                    err_thr = None
            if p_ge1 is None and yhat is not None:
                p_ge1 = 1.0 - float(np.exp(-max(0.0, float(yhat))))
            if p_ge1 is not None and actual_thr is not None:
                brier_ge1 = (float(p_ge1) - (1.0 if int(actual_thr) >= 1 else 0.0))**2
        rec.update({
            "early_threes_expected": yhat,
            "early_threes_prob_ge_1": p_ge1,
            "early_threes_actual": actual_thr,
            "early_threes_error": err_thr,
            "early_threes_brier_ge1": brier_ge1,
        })

        rows.append(rec)

    out = paths.data_processed / f"pbp_reconcile_{date_str}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    # Print summary metrics
    df = pd.DataFrame(rows)
    tip_brier = float(pd.to_numeric(df.get("tip_brier"), errors="coerce").dropna().mean()) if not df.empty else float("nan")
    tip_acc = float(np.mean([(float(p)>=0.5)==(float(y)==1.0) for p, y in zip(pd.to_numeric(df.get("tip_prob_home"), errors="coerce").dropna(), pd.to_numeric(df.get("tip_outcome_home"), errors="coerce").dropna())])) if ("tip_prob_home" in df.columns and "tip_outcome_home" in df.columns) else float("nan")
    if {"first_basket_hit_top1","first_basket_hit_top5"}.issubset(set(df.columns)) and not df.empty:
        fb_df = df.dropna(subset=["first_basket_hit_top1","first_basket_hit_top5"], how="all")
        fb_top1_acc = float(pd.to_numeric(fb_df.get("first_basket_hit_top1"), errors="coerce").dropna().mean()) if not fb_df.empty else float("nan")
        fb_top5_cov = float(pd.to_numeric(fb_df.get("first_basket_hit_top5"), errors="coerce").dropna().mean()) if not fb_df.empty else float("nan")
    else:
        fb_top1_acc = float("nan"); fb_top5_cov = float("nan")
    _thr_col = df.get("early_threes_error")
    if isinstance(_thr_col, pd.Series):
        thr_errs = pd.to_numeric(_thr_col, errors="coerce").dropna()
    else:
        thr_errs = pd.Series(dtype=float)
    thr_mae = float(thr_errs.abs().mean()) if len(thr_errs) > 0 else float("nan")
    thr_rmse = float(np.sqrt((thr_errs**2).mean())) if len(thr_errs) > 0 else float("nan")
    thr_brier = float(pd.to_numeric(df.get("early_threes_brier_ge1"), errors="coerce").dropna().mean()) if not df.empty else float("nan")
    console.print({
        "date": date_str,
        "output": str(out),
        "tip": {"brier": tip_brier, "acc@0.5": tip_acc},
        "first_basket": {"top1_acc": fb_top1_acc, "top5_cov": fb_top5_cov},
        "early_threes": {"mae": thr_mae, "rmse": thr_rmse, "brier_ge1": thr_brier},
    })

@cli.command("reconcile-quarters")
@click.option("--date", "date_str", type=str, required=True, help="Target date YYYY-MM-DD to reconcile quarters/halves vs predictions")
def reconcile_quarters_cmd(date_str: str):
    """Reconcile per-quarter and half totals for a date by joining predictions with raw line scores.

    Writes data/processed/recon_quarters_<date>.csv with columns including:
    - date, home_team, visitor_team, home_tri, away_tri
    - actual_q{1..4}_total, actual_h1_total, actual_h2_total, actual_game_total
    - pred_q{1..4}_total, pred_h1_total, pred_h2_total, pred_game_total
    - err_q{1..4}_total, err_h1_total, err_h2_total, err_game_total (actual - pred)
    """
    console.rule("Reconcile Quarters")
    try:
        _ = pd.to_datetime(date_str).date()
    except Exception:
        console.print("Invalid --date (YYYY-MM-DD)", style="red"); return

    from .teams import to_tricode as _to_tri, normalize_team as _norm
    # Locate predictions for the date (prefer NPU periods if available)
    pred_candidates = [
        paths.data_processed / f"games_predictions_npu_{date_str}.csv",
        paths.data_processed / f"predictions_{date_str}.csv",
        paths.root / f"predictions_{date_str}.csv",
    ]
    pred_path = next((p for p in pred_candidates if p.exists()), None)
    if pred_path is None:
        console.print(f"No predictions found for {date_str}", style="red"); return
    try:
        preds = pd.read_csv(pred_path)
    except Exception as e:
        console.print(f"Failed to read predictions: {e}", style="red"); return
    if preds is None or preds.empty:
        console.print("Predictions file is empty", style="red"); return

    # Normalize prediction team names and tricodes
    preds = preds.copy()
    for c in ("home_team","visitor_team"):
        if c in preds.columns:
            preds[c] = preds[c].apply(_norm)
        else:
            preds[c] = pd.NA
    preds["home_tri"] = preds["home_team"].astype(str).map(_to_tri)
    preds["away_tri"] = preds["visitor_team"].astype(str).map(_to_tri)

    # Identify prediction columns
    def _first_col(df: pd.DataFrame, names: list[str]) -> str | None:
        for n in names:
            if n in df.columns:
                return n
        return None
    # Game total and per-period predicted totals columns (model naming from games_npu.predict_game)
    game_pred_col = _first_col(preds, ["totals","pred_total","model_total","total"])
    # Period columns usually: quarters_q{i}_total and halves_h{1,2}_total
    have_quarters = any((f"quarters_q{i}_total" in preds.columns) for i in range(1,5))
    have_halves = any((f"halves_h{i}_total" in preds.columns) for i in range(1,3))

    # Load raw games with line scores and filter to date
    raw_csv = paths.data_raw / "games_nba_api.csv"
    raw_parq = paths.data_raw / "games_nba_api.parquet"
    raw = None
    if raw_parq.exists():
        try:
            raw = pd.read_parquet(raw_parq)
        except Exception:
            raw = None
    if raw is None and raw_csv.exists():
        try:
            raw = pd.read_csv(raw_csv)
        except Exception:
            raw = None
    if raw is None or raw.empty:
        console.print("Raw games with line scores not found. Run fetch or enrich-periods.", style="red"); return
    # Filter to date
    try:
        raw_day = raw.copy()
        raw_day["date"] = pd.to_datetime(raw_day["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        raw_day = raw_day[raw_day["date"] == date_str]
    except Exception:
        raw_day = raw[raw.get("date").astype(str) == date_str].copy()
    if raw_day is None or raw_day.empty:
        # Fallback: derive actual per-period totals from Play-by-Play when raw line scores are missing
        console.print(f"No raw games found on {date_str}; attempting PBP-based fallback", style="yellow")

        # Build PBP frames map for the date
        pbp_map: dict[str, pd.DataFrame] = {}
        pbp_comb = paths.data_processed / f"pbp_{date_str}.csv"
        if pbp_comb.exists():
            try:
                dfc = pd.read_csv(pbp_comb)
                if (dfc is not None) and (not dfc.empty):
                    # If the combined PBP file lacks any usable score columns, treat it as unusable
                    # and fall back to ESPN-based PBP for quarter totals.
                    cols = set(dfc.columns)
                    has_pair = ("scoreHome" in cols and "scoreAway" in cols) or ("home_score" in cols and ("away_score" in cols or "visitor_score" in cols)) or ("homeScore" in cols and "awayScore" in cols)
                    has_combo = ("SCORE" in cols) or ("score" in cols) or ("combinedScore" in cols)
                    if not (has_pair or has_combo):
                        dfc = None
                        pbp_map = {}
                    else:
                        # Support alternative column names
                        gid_col = "game_id" if "game_id" in dfc.columns else ("GAME_ID" if "GAME_ID" in dfc.columns else None)
                        if gid_col is not None:
                            for gid, grp in dfc.groupby(gid_col):
                                key = str(gid).strip()
                                pbp_map[key.zfill(10) if key.isdigit() else key] = grp.copy()
            except Exception:
                pbp_map = {}
        if (not pbp_map) and pbp_comb.exists():
            # Try per-game files under data/processed/pbp
            try:
                dpg = paths.data_processed / "pbp"
                if dpg.exists():
                    for f in dpg.glob("pbp_*.csv"):
                        try:
                            gid = f.stem.replace("pbp_", "").strip()
                            key = gid.zfill(10) if gid.isdigit() else gid
                            pbp_map[key] = pd.read_csv(f)
                        except Exception:
                            continue
            except Exception:
                pbp_map = {}

        # ESPN fallback: if NBA PBP is missing/blocked, use pbp_espn_<date>.csv
        if not pbp_map:
            pbp_espn = paths.data_processed / f"pbp_espn_{date_str}.csv"
            try:
                dfe = None
                if pbp_espn.exists():
                    dfe = pd.read_csv(pbp_espn)

                # If we have a cached ESPN daily file but it covers only a small fraction of the
                # matchups we predicted for this date, refresh from ESPN to avoid partial slates
                # (e.g., a prior interrupted fetch).
                try:
                    expected_pairs = set(
                        zip(
                            preds["home_tri"].astype(str).str.upper(),
                            preds["away_tri"].astype(str).str.upper(),
                        )
                    )
                    expected_pairs = {p for p in expected_pairs if all(x and x != "NAN" for x in p)}
                    have_pairs = set()
                    if (dfe is not None) and (not dfe.empty) and {"home_tri", "away_tri"}.issubset(set(dfe.columns)):
                        have_pairs = set(
                            zip(
                                dfe["home_tri"].astype(str).str.upper(),
                                dfe["away_tri"].astype(str).str.upper(),
                            )
                        )
                        have_pairs = {p for p in have_pairs if all(x and x != "NAN" for x in p)}
                    if expected_pairs and have_pairs:
                        cov = float(len(have_pairs & expected_pairs)) / float(len(expected_pairs))
                        if cov < 0.80:
                            console.print(
                                f"Cached pbp_espn_{date_str}.csv appears incomplete vs predictions (coverage={cov:.0%}); refreshing from ESPN",
                                style="yellow",
                            )
                            from .pbp_espn import fetch_pbp_espn_for_date
                            dfe2, _ = fetch_pbp_espn_for_date(
                                date_str,
                                only_final=True,
                                rate_delay=0.15,
                                force_scoreboard=True,
                            )
                            if (dfe2 is not None) and (not dfe2.empty):
                                dfe = dfe2
                except Exception:
                    pass

                # If missing or schema is too old, fetch fresh from ESPN.
                if (dfe is None) or dfe.empty or ("home_score" not in dfe.columns) or ("away_score" not in dfe.columns):
                    from .pbp_espn import fetch_pbp_espn_for_date
                    dfe, _ = fetch_pbp_espn_for_date(date_str, only_final=True, rate_delay=0.15, force_scoreboard=True)

                if (dfe is not None) and (not dfe.empty):
                    key_col = "game_id" if "game_id" in dfe.columns and dfe["game_id"].notna().any() else ("event_id" if "event_id" in dfe.columns else None)
                    if key_col:
                        for gid, grp in dfe.groupby(key_col):
                            key = str(gid).strip()
                            pbp_map[key] = grp.copy()
            except Exception:
                pbp_map = {}

        if not pbp_map:
            # Try per-game ESPN files under data/processed/pbp_espn
            try:
                dpe = paths.data_processed / "pbp_espn"
                if dpe.exists():
                    for f in dpe.glob("pbp_espn_*.csv"):
                        try:
                            # pbp_espn_<gameId>.csv or pbp_espn_event_<eventId>.csv
                            stem = f.stem
                            if stem.startswith("pbp_espn_event_"):
                                gid = stem.replace("pbp_espn_event_", "").strip()
                            else:
                                gid = stem.replace("pbp_espn_", "").strip()
                            if not gid:
                                continue
                            dfx = pd.read_csv(f)
                            if (dfx is None) or dfx.empty or ("date" not in dfx.columns):
                                continue
                            dfx["date"] = pd.to_datetime(dfx["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                            dfx = dfx[dfx["date"] == date_str]
                            if dfx is None or dfx.empty:
                                continue
                            pbp_map[gid] = dfx
                        except Exception:
                            continue
            except Exception:
                pbp_map = {}
        if not pbp_map:
            console.print("PBP logs not found; cannot build quarter reconciliation for this date", style="red"); return

        # Map game_id -> (home_tri, away_tri)
        # Prefer reusing prior pbp reconciliation if available for this date
        try:
            pbp_recon_path = paths.data_processed / f"pbp_reconcile_{date_str}.csv"
            if pbp_recon_path.exists():
                pr = pd.read_csv(pbp_recon_path)
                if (pr is not None) and (not pr.empty) and {"game_id","home_team","visitor_team"}.issubset(set(pr.columns)):
                    # Normalize keys to string game_id
                    _m = {}
                    for _, r in pr.iterrows():
                        gid = str(r.get("game_id")).strip()
                        h = str(r.get("home_team") or "").upper()
                        a = str(r.get("visitor_team") or "").upper()
                        if gid and h and a:
                            _m[gid] = (h, a)
                    # If we got any, filter pbp_map to just these game IDs
                    if _m:
                        # Normalize pbp_map keys similarly (both raw and zfilled forms)
                        keep_keys = set(_m.keys()) | set(k.zfill(10) for k in _m.keys() if k.isdigit())
                        # Only apply this filter if pbp_map is keyed by the same IDs.
                        # ESPN PBP uses event_id keys, and filtering by NBA game_id would wipe the map.
                        try:
                            has_overlap = any(((k in keep_keys) or (str(k).strip() in keep_keys)) for k in pbp_map.keys())
                        except Exception:
                            has_overlap = False
                        if has_overlap:
                            pbp_map = {k: v for k, v in pbp_map.items() if (k in keep_keys) or (k.strip() in keep_keys)}
                            if pbp_map:
                                gid_team_map = _m
        except Exception:
            pass
        try:
            from .pbp_markets import _gid_team_map_for_date as _map
            gid_team_map = _map(date_str) or {}
            # Normalize keys to strings to avoid int/str mismatches when comparing against
            # pbp_map keys (which are always stringified).
            try:
                gid_team_map = {str(k).strip(): v for k, v in gid_team_map.items() if k is not None}
            except Exception:
                pass
        except Exception:
            gid_team_map = {}

        # If our PBP frames are keyed differently than the mapping (e.g., ESPN event_id vs NBA game_id),
        # force a derived mapping from the frames.
        try:
            if gid_team_map and pbp_map:
                pbp_keys = set(str(k).strip() for k in pbp_map.keys())
                map_keys = set(str(k).strip() for k in gid_team_map.keys())
                if pbp_keys.isdisjoint(map_keys):
                    gid_team_map = {}
        except Exception:
            pass
        if not gid_team_map:
            # Derive mapping directly from PBP frames when structured columns are available
            derived: dict[str, tuple[str,str]] = {}
            # Restrict to matchups we actually predicted for this date
            valid_pairs = set(zip(preds["home_tri"].astype(str).str.upper(), preds["away_tri"].astype(str).str.upper()))
            for gid, gdf in pbp_map.items():
                try:
                    cols = set(gdf.columns)
                    if {"home_tri", "away_tri"}.issubset(cols):
                        h0 = str(gdf["home_tri"].dropna().iloc[0]).upper() if gdf["home_tri"].notna().any() else None
                        v0 = str(gdf["away_tri"].dropna().iloc[0]).upper() if gdf["away_tri"].notna().any() else None
                        if h0 and v0:
                            pair = (h0, v0)
                            if not valid_pairs or pair in valid_pairs:
                                if pair not in derived.values():
                                    derived[str(gid)] = pair
                            continue
                    # Prefer structured liveData schema
                    if {"teamTricode","location"}.issubset(cols):
                        tri_h = gdf[gdf["location"].astype(str).str.lower()=="h"].get("teamTricode").dropna()
                        tri_v = gdf[gdf["location"].astype(str).str.lower()=="v"].get("teamTricode").dropna()
                        h0 = str(tri_h.iloc[0]).upper() if len(tri_h)>0 else None
                        v0 = str(tri_v.iloc[0]).upper() if len(tri_v)>0 else None
                        if h0 and v0:
                            pair = (h0, v0)
                            if not valid_pairs or pair in valid_pairs:
                                # Only keep first mapping per pair to avoid duplicates
                                if pair not in derived.values():
                                    derived[str(gid)] = pair
                            continue
                    # Legacy schema heuristic: look for PLAYER_TEAM abbreviations tagged by home/away if present
                    # As a last resort, take the first two distinct tricodes observed and assume order is (home, away) if 'location' missing
                    tri_col = None
                    for c in ("PLAYER1_TEAM_ABBREVIATION","team_abbr","TEAM_ABBREVIATION","teamTricode","team"):
                        if c in cols:
                            tri_col = c; break
                    if tri_col:
                        tri_vals = [str(x).upper() for x in gdf[tri_col].dropna().astype(str).tolist()]
                        uniq = []
                        for t in tri_vals:
                            if t and t not in uniq:
                                uniq.append(t)
                            if len(uniq) >= 2:
                                break
                        if len(uniq) >= 2:
                            pair = (uniq[0], uniq[1])
                            if not valid_pairs or pair in valid_pairs:
                                if pair not in derived.values():
                                    derived[str(gid)] = pair
                except Exception:
                    continue
            gid_team_map = derived
        if not gid_team_map:
            console.print("Could not map game IDs to teams from PBP; skipping PBP fallback", style="red"); return

        # Helper to extract cumulative home/away score from a PBP row
        def _scores_from_row(r: pd.Series) -> tuple[float|None, float|None]:
            # Prefer explicit numeric columns if present
            for hc, ac in (("scoreHome","scoreAway"), ("home_score","away_score"), ("homeScore","awayScore"), ("home_score","visitor_score"), ("HOME_SCORE","VISITOR_SCORE")):
                if (hc in r.index) and (ac in r.index):
                    try:
                        h = float(pd.to_numeric(r.get(hc), errors="coerce"))
                        a = float(pd.to_numeric(r.get(ac), errors="coerce"))
                        if not (np.isnan(h) or np.isnan(a)):
                            return h, a
                    except Exception:
                        pass
            # Else parse combined score like "102-99" in SCORE/score
            for sc in ("SCORE","score","combinedScore"):
                if sc in r.index:
                    s = str(r.get(sc) or "")
                    if "-" in s:
                        parts = [p.strip() for p in s.split("-")]
                        if len(parts) == 2:
                            try:
                                return float(parts[0]), float(parts[1])
                            except Exception:
                                pass
            return None, None

        # Build rows by iterating PBP games and computing per-period cumulative scores
        rows: list[dict] = []
        for gid, gdf in pbp_map.items():
            gid_key = str(gid).strip()
            gid_key_alt = gid_key.zfill(10) if gid_key.isdigit() else None

            if gid_key in gid_team_map:
                htri, atri = gid_team_map[gid_key]
            elif gid_key_alt and (gid_key_alt in gid_team_map):
                htri, atri = gid_team_map[gid_key_alt]
            else:
                continue
            htri = str(htri or "").upper(); atri = str(atri or "").upper()
            if (not htri) or (not atri):
                continue
            # Ensure period column exists and is numeric
            if "period" not in gdf.columns:
                # Try alternative capitalization
                if "PERIOD" in gdf.columns:
                    gdf = gdf.rename(columns={"PERIOD":"period"})
                else:
                    continue
            tmp = gdf.copy()
            # Determine end-of-period by choosing the smallest remaining clock value (closest to 0:00).
            # ESPN export already provides clock_sec_remaining.
            if "clock_sec_remaining" in tmp.columns:
                tmp["_clock_sec"] = pd.to_numeric(tmp["clock_sec_remaining"], errors="coerce")
            else:
                clk = "clock" if "clock" in tmp.columns else ("PCTIMESTRING" if "PCTIMESTRING" in tmp.columns else None)
                if clk:
                    def _clock_to_sec(v: Any) -> float:
                        try:
                            s = str(v or "")
                            # Examples: PT12M00.00S, PT00M00.00S
                            m = re.match(r"^PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$", s)
                            if not m:
                                return float("nan")
                            mm = float(m.group(1)) if m.group(1) else 0.0
                            ss = float(m.group(2)) if m.group(2) else 0.0
                            return (mm * 60.0) + ss
                        except Exception:
                            return float("nan")
                    tmp["_clock_sec"] = tmp[clk].apply(_clock_to_sec)
                else:
                    tmp["_clock_sec"] = float("nan")

            if "actionNumber" in tmp.columns:
                tmp["_action_num"] = pd.to_numeric(tmp["actionNumber"], errors="coerce")
            elif "sequence" in tmp.columns:
                tmp["_action_num"] = pd.to_numeric(tmp["sequence"], errors="coerce")
            else:
                tmp["_action_num"] = float("nan")

            # Compute end-of-period cumulative scores
            per_end: dict[int, tuple[float,float]] = {}
            for p in [1,2,3,4]:
                sub = tmp[tmp["period"] == p]
                if sub is None or sub.empty:
                    continue
                # Sort within period: clock closest to 0 first; break ties by later actionNumber.
                sub2 = sub.sort_values(["_clock_sec", "_action_num"], ascending=[True, False], na_position="last")
                hs, as_ = None, None
                for _, rr in sub2.iterrows():
                    hs, as_ = _scores_from_row(rr)
                    if (hs is not None) and (as_ is not None):
                        break
                if (hs is None) or (as_ is None):
                    continue
                per_end[p] = (float(hs), float(as_))

            # Derive per-period totals by differences of cumulative scores
            aq = {}
            last_h, last_a = 0.0, 0.0
            for p in [1,2,3,4]:
                if p in per_end:
                    hs, as_ = per_end[p]
                    aq[f"q{p}"] = max(0.0, (hs - last_h) + (as_ - last_a))
                    last_h, last_a = hs, as_
                else:
                    aq[f"q{p}"] = pd.NA
            ah1 = (aq.get("q1") if pd.notna(aq.get("q1")) else 0.0) + (aq.get("q2") if pd.notna(aq.get("q2")) else 0.0)
            ah2 = (aq.get("q3") if pd.notna(aq.get("q3")) else 0.0) + (aq.get("q4") if pd.notna(aq.get("q4")) else 0.0)
            atot = (last_h + last_a) if (last_h is not None and last_a is not None) else pd.NA

            # Match prediction by tri pair
            m = preds[(preds["home_tri"].astype(str).str.upper() == htri) & (preds["away_tri"].astype(str).str.upper() == atri)]
            pred_row = m.iloc[0] if not m.empty else None

            rec: dict = {
                "date": date_str,
                "home_team": pred_row.get("home_team") if (pred_row is not None) else None,
                "visitor_team": pred_row.get("visitor_team") if (pred_row is not None) else None,
                "home_tri": htri,
                "away_tri": atri,
            }
            for i in range(1,5):
                rec[f"actual_q{i}_total"] = aq.get(f"q{i}")
            rec["actual_h1_total"] = ah1 if pd.notna(ah1) else pd.NA
            rec["actual_h2_total"] = ah2 if pd.notna(ah2) else pd.NA
            rec["actual_game_total"] = atot

            # Predictions
            if pred_row is not None:
                for i in range(1,5):
                    pk = f"quarters_q{i}_total"
                    rec[f"pred_q{i}_total"] = float(pred_row.get(pk)) if pk in pred_row.index and pd.notna(pred_row.get(pk)) else pd.NA
                for hi, parts in [(1, (1,2)), (2, (3,4))]:
                    hk = f"halves_h{hi}_total"
                    val = float(pred_row.get(hk)) if hk in pred_row.index and pd.notna(pred_row.get(hk)) else pd.NA
                    if (pd.isna(val)) and all(pd.notna(rec.get(f"pred_q{j}_total")) for j in parts):
                        val = float(rec.get(f"pred_q{parts[0]}_total", 0.0)) + float(rec.get(f"pred_q{parts[1]}_total", 0.0))
                    rec[f"pred_h{hi}_total"] = val
                if game_pred_col is not None:
                    try:
                        rec["pred_game_total"] = float(pred_row.get(game_pred_col))
                    except Exception:
                        rec["pred_game_total"] = pd.NA
                else:
                    if all(pd.notna(rec.get(f"pred_q{i}_total")) for i in range(1,5)):
                        rec["pred_game_total"] = sum(float(rec.get(f"pred_q{i}_total", 0.0)) for i in range(1,5))
                    else:
                        rec["pred_game_total"] = pd.NA
            else:
                for i in range(1,5):
                    rec[f"pred_q{i}_total"] = pd.NA
                rec["pred_h1_total"] = pd.NA
                rec["pred_h2_total"] = pd.NA
                rec["pred_game_total"] = pd.NA

            # Errors (actual - pred)
            for i in range(1,5):
                a = rec.get(f"actual_q{i}_total"); p = rec.get(f"pred_q{i}_total")
                rec[f"err_q{i}_total"] = (float(a) - float(p)) if (pd.notna(a) and pd.notna(p)) else pd.NA
            for hi, ak in [(1, "actual_h1_total"), (2, "actual_h2_total")]:
                a = rec.get(ak); p = rec.get(f"pred_h{hi}_total")
                rec[f"err_h{hi}_total"] = (float(a) - float(p)) if (pd.notna(a) and pd.notna(p)) else pd.NA
            a = rec.get("actual_game_total"); p = rec.get("pred_game_total")
            rec["err_game_total"] = (float(a) - float(p)) if (pd.notna(a) and pd.notna(p)) else pd.NA

            rows.append(rec)

        if not rows:
            console.print("PBP fallback produced no rows; nothing to write", style="yellow")
            try:
                console.print(
                    {
                        "date": date_str,
                        "pbp_games": int(len(pbp_map)),
                        "mapped_games": int(len(gid_team_map)),
                        "pbp_key_sample": list(list(pbp_map.keys())[:5]),
                        "map_key_sample": list(list(gid_team_map.keys())[:5]) if isinstance(gid_team_map, dict) else None,
                    },
                    style="dim",
                )
            except Exception:
                pass
            return
        out = paths.data_processed / f"recon_quarters_{date_str}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        console.print({"date": date_str, "rows": int(len(rows)), "output": str(out), "source": "pbp"})
        return

    # If we found raw rows but Q1 line scores are missing (commonly all zeros/NaN),
    # the downstream quarter backtests will be broken. Prefer PBP-derived actuals in that case.
    try:
        if {"home_q1", "visitor_q1", "home_pts", "visitor_pts"}.issubset(set(raw_day.columns)):
            q1_tot = pd.to_numeric(raw_day["home_q1"], errors="coerce") + pd.to_numeric(raw_day["visitor_q1"], errors="coerce")
            final_tot = pd.to_numeric(raw_day["home_pts"], errors="coerce") + pd.to_numeric(raw_day["visitor_pts"], errors="coerce")

            played = final_tot.fillna(0) > 0
            if int(played.sum()) > 0:
                q1_zero_rate = float((q1_tot[played].fillna(0) <= 0).mean())
                # Additional guard: if other quarters exist and are mostly non-zero, Q1 is likely missing.
                q234_nonzero_rate = None
                if {"home_q2", "visitor_q2", "home_q3", "visitor_q3", "home_q4", "visitor_q4"}.issubset(set(raw_day.columns)):
                    q234_tot = (
                        pd.to_numeric(raw_day["home_q2"], errors="coerce") + pd.to_numeric(raw_day["visitor_q2"], errors="coerce") +
                        pd.to_numeric(raw_day["home_q3"], errors="coerce") + pd.to_numeric(raw_day["visitor_q3"], errors="coerce") +
                        pd.to_numeric(raw_day["home_q4"], errors="coerce") + pd.to_numeric(raw_day["visitor_q4"], errors="coerce")
                    )
                    q234_nonzero_rate = float((q234_tot[played].fillna(0) > 0).mean())

                # Trigger PBP fallback if Q1 is missing for (almost) all played games.
                if (q1_zero_rate >= 0.95) and ((q234_nonzero_rate is None) or (q234_nonzero_rate >= 0.50)):
                    console.print(
                        f"Raw line scores for {date_str} appear to be missing Q1 totals; using PBP-derived quarters instead",
                        style="yellow",
                    )
                    raw_day = raw_day.iloc[0:0].copy()
    except Exception:
        pass

    if raw_day is None or raw_day.empty:
        # Reuse the existing PBP fallback path
        console.print(f"No usable raw line scores on {date_str}; attempting PBP-based fallback", style="yellow")

        # Build PBP frames map for the date
        pbp_map: dict[str, pd.DataFrame] = {}
        pbp_comb = paths.data_processed / f"pbp_{date_str}.csv"
        if pbp_comb.exists():
            try:
                dfc = pd.read_csv(pbp_comb)
                if (dfc is not None) and (not dfc.empty):
                    # Support alternative column names
                    gid_col = "game_id" if "game_id" in dfc.columns else ("GAME_ID" if "GAME_ID" in dfc.columns else None)
                    if gid_col is not None:
                        for gid, grp in dfc.groupby(gid_col):
                            key = str(gid).strip()
                            pbp_map[key.zfill(10) if key.isdigit() else key] = grp.copy()
                    elif "gameId" in dfc.columns:
                        for gid, grp in dfc.groupby("gameId"):
                            key = str(gid).strip()
                            pbp_map[key.zfill(10) if key.isdigit() else key] = grp.copy()
            except Exception:
                pbp_map = {}
        if not pbp_map:
            # Try per-game files under data/processed/pbp
            try:
                dpg = paths.data_processed / "pbp"
                if dpg.exists():
                    for f in dpg.glob("pbp_*.csv"):
                        try:
                            gid = f.stem.replace("pbp_", "").strip()
                            key = gid.zfill(10) if gid.isdigit() else gid
                            pbp_map[key] = pd.read_csv(f)
                        except Exception:
                            continue
            except Exception:
                pbp_map = {}
        if not pbp_map:
            console.print("PBP logs not found; cannot build quarter reconciliation for this date", style="red"); return

        # Map game_id -> (home_tri, away_tri)
        # Prefer reusing prior pbp reconciliation if available for this date
        try:
            pbp_recon_path = paths.data_processed / f"pbp_reconcile_{date_str}.csv"
            if pbp_recon_path.exists():
                pr = pd.read_csv(pbp_recon_path)
                if (pr is not None) and (not pr.empty) and {"game_id","home_team","visitor_team"}.issubset(set(pr.columns)):
                    # Normalize keys to string game_id
                    _m = {}
                    for _, r in pr.iterrows():
                        gid = str(r.get("game_id")).strip()
                        h = str(r.get("home_team") or "").upper()
                        a = str(r.get("visitor_team") or "").upper()
                        if gid and h and a:
                            _m[gid] = (h, a)
                    # If we got any, filter pbp_map to just these game IDs
                    if _m:
                        # Normalize pbp_map keys similarly (both raw and zfilled forms)
                        keep_keys = set(_m.keys()) | set(k.zfill(10) for k in _m.keys() if k.isdigit())
                        # Only apply this filter if pbp_map is keyed by the same IDs.
                        # ESPN PBP uses event_id keys, and filtering by NBA game_id would wipe the map.
                        try:
                            has_overlap = any(((k in keep_keys) or (str(k).strip() in keep_keys)) for k in pbp_map.keys())
                        except Exception:
                            has_overlap = False
                        if has_overlap:
                            pbp_map = {k: v for k, v in pbp_map.items() if (k in keep_keys) or (k.strip() in keep_keys)}
                            if pbp_map:
                                gid_team_map = _m
        except Exception:
            pass
        try:
            from .pbp_markets import _gid_team_map_for_date as _map
            gid_team_map = _map(date_str) or {}
            # Normalize keys to strings to avoid int/str mismatches when comparing against
            # pbp_map keys (which are always stringified).
            try:
                gid_team_map = {str(k).strip(): v for k, v in gid_team_map.items() if k is not None}
            except Exception:
                pass
        except Exception:
            gid_team_map = {}
        if not gid_team_map:
            # Derive mapping directly from PBP frames when structured columns are available
            derived: dict[str, tuple[str,str]] = {}
            # Restrict to matchups we actually predicted for this date
            valid_pairs = set(zip(preds["home_tri"].astype(str).str.upper(), preds["away_tri"].astype(str).str.upper()))
            for gid, gdf in pbp_map.items():
                try:
                    cols = set(gdf.columns)
                    # Prefer structured liveData schema
                    if {"teamTricode","location"}.issubset(cols):
                        tri_h = gdf[gdf["location"].astype(str).str.lower()=="h"].get("teamTricode").dropna()
                        tri_v = gdf[gdf["location"].astype(str).str.lower()=="v"].get("teamTricode").dropna()
                        h0 = str(tri_h.iloc[0]).upper() if len(tri_h)>0 else None
                        v0 = str(tri_v.iloc[0]).upper() if len(tri_v)>0 else None
                        if h0 and v0:
                            pair = (h0, v0)
                            if not valid_pairs or pair in valid_pairs:
                                # Only keep first mapping per pair to avoid duplicates
                                if pair not in derived.values():
                                    derived[str(gid)] = pair
                            continue
                    # Legacy schema heuristic: look for PLAYER_TEAM abbreviations tagged by home/away if present
                    # As a last resort, take the first two distinct tricodes observed and assume order is (home, away) if 'location' missing
                    tri_col = None
                    for c in ("PLAYER1_TEAM_ABBREVIATION","team_abbr","TEAM_ABBREVIATION","teamTricode"):
                        if c in cols:
                            tri_col = c; break
                    if tri_col:
                        tri_vals = [str(x).upper() for x in gdf[tri_col].dropna().astype(str).tolist()]
                        uniq = []
                        for t in tri_vals:
                            if t and t not in uniq:
                                uniq.append(t)
                            if len(uniq) >= 2:
                                break
                        if len(uniq) >= 2:
                            pair = (uniq[0], uniq[1])
                            if not valid_pairs or pair in valid_pairs:
                                if pair not in derived.values():
                                    derived[str(gid)] = pair
                except Exception:
                    continue
            gid_team_map = derived
        if not gid_team_map:
            console.print("Could not map game IDs to teams from PBP; skipping PBP fallback", style="red"); return

        # Helper to extract cumulative home/away score from a PBP row
        def _scores_from_row(r: pd.Series) -> tuple[float|None, float|None]:
            # Prefer explicit numeric columns if present
            for hc, ac in (("scoreHome","scoreAway"), ("home_score","visitor_score"), ("HOME_SCORE","VISITOR_SCORE")):
                if (hc in r.index) and (ac in r.index):
                    try:
                        h = float(pd.to_numeric(r.get(hc), errors="coerce"))
                        a = float(pd.to_numeric(r.get(ac), errors="coerce"))
                        if not (np.isnan(h) or np.isnan(a)):
                            return h, a
                    except Exception:
                        pass
            # Else parse combined score like "102-99" in SCORE/score
            for sc in ("SCORE","score","combinedScore"):
                if sc in r.index:
                    s = str(r.get(sc) or "")
                    if "-" in s:
                        parts = [p.strip() for p in s.split("-")]
                        if len(parts) == 2:
                            try:
                                return float(parts[0]), float(parts[1])
                            except Exception:
                                pass
            return None, None

        # Build rows by iterating PBP games and computing per-period cumulative scores
        rows: list[dict] = []
        for gid, gdf in pbp_map.items():
            gid_key = str(gid).strip()
            gid_key_alt = gid_key.zfill(10) if gid_key.isdigit() else None

            if gid_key in gid_team_map:
                htri, atri = gid_team_map[gid_key]
            elif gid_key_alt and (gid_key_alt in gid_team_map):
                htri, atri = gid_team_map[gid_key_alt]
            else:
                continue
            htri = str(htri or "").upper(); atri = str(atri or "").upper()
            if (not htri) or (not atri):
                continue
            # Ensure period column exists and is numeric
            if "period" not in gdf.columns:
                # Try alternative capitalization
                if "PERIOD" in gdf.columns:
                    gdf = gdf.rename(columns={"PERIOD":"period"})
                else:
                    continue
            tmp = gdf.copy()
            # Determine end-of-period by choosing the smallest remaining clock value (closest to 0:00).
            clk = "clock" if "clock" in tmp.columns else ("PCTIMESTRING" if "PCTIMESTRING" in tmp.columns else None)
            if clk:
                def _clock_to_sec(v: Any) -> float:
                    try:
                        s = str(v or "")
                        # Examples: PT12M00.00S, PT00M00.00S
                        m = re.match(r"^PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$", s)
                        if not m:
                            return float("nan")
                        mm = float(m.group(1)) if m.group(1) else 0.0
                        ss = float(m.group(2)) if m.group(2) else 0.0
                        return (mm * 60.0) + ss
                    except Exception:
                        return float("nan")
                tmp["_clock_sec"] = tmp[clk].apply(_clock_to_sec)
            else:
                tmp["_clock_sec"] = float("nan")

            if "actionNumber" in tmp.columns:
                tmp["_action_num"] = pd.to_numeric(tmp["actionNumber"], errors="coerce")
            else:
                tmp["_action_num"] = float("nan")

            # Compute end-of-period cumulative scores
            per_end: dict[int, tuple[float,float]] = {}
            for p in [1,2,3,4]:
                sub = tmp[tmp["period"] == p]
                if sub is None or sub.empty:
                    continue
                # Sort within period: clock closest to 0 first; break ties by later actionNumber.
                sub2 = sub.sort_values(["_clock_sec", "_action_num"], ascending=[True, False], na_position="last")
                hs, as_ = None, None
                for _, rr in sub2.iterrows():
                    hs, as_ = _scores_from_row(rr)
                    if (hs is not None) and (as_ is not None):
                        break
                if (hs is None) or (as_ is None):
                    continue
                per_end[p] = (float(hs), float(as_))

            # Derive per-period totals by differences of cumulative scores
            aq = {}
            last_h, last_a = 0.0, 0.0
            for p in [1,2,3,4]:
                if p in per_end:
                    hs, as_ = per_end[p]
                    aq[f"q{p}"] = max(0.0, (hs - last_h) + (as_ - last_a))
                    last_h, last_a = hs, as_
                else:
                    aq[f"q{p}"] = pd.NA
            ah1 = (aq.get("q1") if pd.notna(aq.get("q1")) else 0.0) + (aq.get("q2") if pd.notna(aq.get("q2")) else 0.0)
            ah2 = (aq.get("q3") if pd.notna(aq.get("q3")) else 0.0) + (aq.get("q4") if pd.notna(aq.get("q4")) else 0.0)
            atot = (last_h + last_a) if (last_h is not None and last_a is not None) else pd.NA

            # Match prediction by tri pair
            m = preds[(preds["home_tri"].astype(str).str.upper() == htri) & (preds["away_tri"].astype(str).str.upper() == atri)]
            pred_row = m.iloc[0] if not m.empty else None

            rec: dict = {
                "date": date_str,
                "home_team": pred_row.get("home_team") if (pred_row is not None) else None,
                "visitor_team": pred_row.get("visitor_team") if (pred_row is not None) else None,
                "home_tri": htri,
                "away_tri": atri,
            }
            for i in range(1,5):
                rec[f"actual_q{i}_total"] = aq.get(f"q{i}")
            rec["actual_h1_total"] = ah1 if pd.notna(ah1) else pd.NA
            rec["actual_h2_total"] = ah2 if pd.notna(ah2) else pd.NA
            rec["actual_game_total"] = atot

            # Predictions
            if pred_row is not None:
                for i in range(1,5):
                    pk = f"quarters_q{i}_total"
                    rec[f"pred_q{i}_total"] = float(pred_row.get(pk)) if pk in pred_row.index and pd.notna(pred_row.get(pk)) else pd.NA
                for hi, parts in [(1, (1,2)), (2, (3,4))]:
                    hk = f"halves_h{hi}_total"
                    val = float(pred_row.get(hk)) if hk in pred_row.index and pd.notna(pred_row.get(hk)) else pd.NA
                    if (pd.isna(val)) and all(pd.notna(rec.get(f"pred_q{j}_total")) for j in parts):
                        val = float(rec.get(f"pred_q{parts[0]}_total", 0.0)) + float(rec.get(f"pred_q{parts[1]}_total", 0.0))
                    rec[f"pred_h{hi}_total"] = val
                if game_pred_col is not None:
                    try:
                        rec["pred_game_total"] = float(pred_row.get(game_pred_col))
                    except Exception:
                        rec["pred_game_total"] = pd.NA
                else:
                    if all(pd.notna(rec.get(f"pred_q{i}_total")) for i in range(1,5)):
                        rec["pred_game_total"] = sum(float(rec.get(f"pred_q{i}_total", 0.0)) for i in range(1,5))
                    else:
                        rec["pred_game_total"] = pd.NA
            else:
                for i in range(1,5):
                    rec[f"pred_q{i}_total"] = pd.NA
                rec["pred_h1_total"] = pd.NA
                rec["pred_h2_total"] = pd.NA
                rec["pred_game_total"] = pd.NA

            # Errors (actual - pred)
            for i in range(1,5):
                a = rec.get(f"actual_q{i}_total"); p = rec.get(f"pred_q{i}_total")
                rec[f"err_q{i}_total"] = (float(a) - float(p)) if (pd.notna(a) and pd.notna(p)) else pd.NA
            for hi, ak in [(1, "actual_h1_total"), (2, "actual_h2_total")]:
                a = rec.get(ak); p = rec.get(f"pred_h{hi}_total")
                rec[f"err_h{hi}_total"] = (float(a) - float(p)) if (pd.notna(a) and pd.notna(p)) else pd.NA
            a = rec.get("actual_game_total"); p = rec.get("pred_game_total")
            rec["err_game_total"] = (float(a) - float(p)) if (pd.notna(a) and pd.notna(p)) else pd.NA

            rows.append(rec)

        if not rows:
            console.print("PBP fallback produced no rows; nothing to write", style="yellow")
            try:
                console.print(
                    {
                        "date": date_str,
                        "pbp_games": int(len(pbp_map)),
                        "mapped_games": int(len(gid_team_map)),
                        "pbp_key_sample": list(list(pbp_map.keys())[:5]),
                        "map_key_sample": list(list(gid_team_map.keys())[:5]) if isinstance(gid_team_map, dict) else None,
                    },
                    style="dim",
                )
            except Exception:
                pass
            return
        out = paths.data_processed / f"recon_quarters_{date_str}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        console.print({"date": date_str, "rows": int(len(rows)), "output": str(out), "source": "pbp"})
        return
    # Compute tricodes
    raw_day = raw_day.copy()
    raw_day["home_tri"] = raw_day["home_team"].astype(str).map(_to_tri)
    raw_day["away_tri"] = raw_day["visitor_team"].astype(str).map(_to_tri)

    # Build reconciliation rows by matchup key (home_tri,away_tri)
    rows: list[dict] = []
    for _, g in raw_day.iterrows():
        htri = str(g.get("home_tri") or "").upper(); atri = str(g.get("away_tri") or "").upper()
        if not htri or not atri:
            continue
        # Match prediction by tricode pair
        m = preds[(preds["home_tri"].astype(str).str.upper() == htri) & (preds["away_tri"].astype(str).str.upper() == atri)]
        if m.empty:
            # Try reversed (in case of naming mismatch)
            m = preds[(preds["home_tri"].astype(str).str.upper() == htri) & (preds["visitor_team"].astype(str).map(_to_tri).str.upper() == atri)]
        pred_row = m.iloc[0] if not m.empty else None

        rec: dict = {
            "date": date_str,
            "home_team": g.get("home_team"),
            "visitor_team": g.get("visitor_team"),
            "home_tri": htri,
            "away_tri": atri,
        }
        # Actuals from raw line scores
        aq = {}
        for i in range(1,5):
            hq = g.get(f"home_q{i}"); vq = g.get(f"visitor_q{i}")
            try:
                aq[f"q{i}"] = (float(hq) if pd.notna(hq) else 0.0) + (float(vq) if pd.notna(vq) else 0.0)
            except Exception:
                aq[f"q{i}"] = pd.NA
        # Halves
        def _sum2(a,b):
            try:
                return (float(a) if pd.notna(a) else 0.0) + (float(b) if pd.notna(b) else 0.0)
            except Exception:
                return pd.NA
        ah1 = _sum2(g.get("home_q1"), g.get("visitor_q1"))
        ah1 = (ah1 + _sum2(g.get("home_q2"), g.get("visitor_q2"))) if pd.notna(ah1) else pd.NA
        ah2 = _sum2(g.get("home_q3"), g.get("visitor_q3"))
        ah2 = (ah2 + _sum2(g.get("home_q4"), g.get("visitor_q4"))) if pd.notna(ah2) else pd.NA
        atot = None
        try:
            atot = (float(g.get("home_pts")) if pd.notna(g.get("home_pts")) else 0.0) + (float(g.get("visitor_pts")) if pd.notna(g.get("visitor_pts")) else 0.0)
        except Exception:
            atot = pd.NA

        for i in range(1,5):
            rec[f"actual_q{i}_total"] = aq.get(f"q{i}")
        rec["actual_h1_total"] = ah1
        rec["actual_h2_total"] = ah2
        rec["actual_game_total"] = atot

        # Predictions
        if pred_row is not None:
            # Quarter preds
            for i in range(1,5):
                pk = f"quarters_q{i}_total"
                rec[f"pred_q{i}_total"] = float(pred_row.get(pk)) if pk in pred_row.index and pd.notna(pred_row.get(pk)) else pd.NA
            # Halves from columns or derive from quarters if missing
            for hi, parts in [(1, (1,2)), (2, (3,4))]:
                hk = f"halves_h{hi}_total"
                val = float(pred_row.get(hk)) if hk in pred_row.index and pd.notna(pred_row.get(hk)) else pd.NA
                if (pd.isna(val)) and all(pd.notna(rec.get(f"pred_q{j}_total")) for j in parts):
                    val = float(rec.get(f"pred_q{parts[0]}_total", 0.0)) + float(rec.get(f"pred_q{parts[1]}_total", 0.0))
                rec[f"pred_h{hi}_total"] = val
            # Game total prediction
            if game_pred_col is not None:
                try:
                    rec["pred_game_total"] = float(pred_row.get(game_pred_col))
                except Exception:
                    rec["pred_game_total"] = pd.NA
            else:
                # Sum quarters if available
                if all(pd.notna(rec.get(f"pred_q{i}_total")) for i in range(1,5)):
                    rec["pred_game_total"] = sum(float(rec.get(f"pred_q{i}_total", 0.0)) for i in range(1,5))
                else:
                    rec["pred_game_total"] = pd.NA
        else:
            # No prediction available; leave pred_* as NA
            for i in range(1,5):
                rec[f"pred_q{i}_total"] = pd.NA
            rec["pred_h1_total"] = pd.NA
            rec["pred_h2_total"] = pd.NA
            rec["pred_game_total"] = pd.NA

        # Errors (actual - pred)
        for i in range(1,5):
            a = rec.get(f"actual_q{i}_total"); p = rec.get(f"pred_q{i}_total")
            rec[f"err_q{i}_total"] = (float(a) - float(p)) if (pd.notna(a) and pd.notna(p)) else pd.NA
        for hi, ak in [(1, "actual_h1_total"), (2, "actual_h2_total")]:
            a = rec.get(ak); p = rec.get(f"pred_h{hi}_total")
            rec[f"err_h{hi}_total"] = (float(a) - float(p)) if (pd.notna(a) and pd.notna(p)) else pd.NA
        a = rec.get("actual_game_total"); p = rec.get("pred_game_total")
        rec["err_game_total"] = (float(a) - float(p)) if (pd.notna(a) and pd.notna(p)) else pd.NA

        rows.append(rec)

    out = paths.data_processed / f"recon_quarters_{date_str}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    console.print({"date": date_str, "rows": int(len(rows)), "output": str(out)})


@cli.command("reconcile-quarters-range")
@click.option("--start", "start_date", type=str, required=True, help="Start date YYYY-MM-DD")
@click.option("--end", "end_date", type=str, required=True, help="End date YYYY-MM-DD")
def reconcile_quarters_range_cmd(start_date: str, end_date: str):
    """Backfill recon_quarters_<date>.csv for a date range.

    Iterates [start..end] inclusive and invokes reconcile-quarters for each date.
    """
    console.rule("Reconcile Quarters: Range")
    try:
        s = pd.to_datetime(start_date).date(); e = pd.to_datetime(end_date).date()
    except Exception:
        console.print("Invalid --start/--end (YYYY-MM-DD)", style="red"); return
    if e < s:
        console.print("--end must be >= --start", style="red"); return
    dates = list(pd.date_range(s, e, freq="D").date)
    ok = 0; fail = 0
    for d in track(dates, description="Reconciling"):
        ds = str(d)
        try:
            # Avoid Click command invocation (which triggers argument parsing/usage output).
            # Call the underlying function directly.
            cb = getattr(reconcile_quarters_cmd, "callback", None)
            if callable(cb):
                cb(ds)
            else:
                # Extremely defensive fallback; should not happen.
                cli.main(["reconcile-quarters", "--date", ds], standalone_mode=False)  # type: ignore
            ok += 1
        except Exception:
            fail += 1
    console.print({"start": str(s), "end": str(e), "ok": int(ok), "fail": int(fail)})

@cli.command("calibrate-totals")
@click.option("--anchor", "anchor_date", type=str, required=True, help="Anchor date YYYY-MM-DD (typically yesterday)")
@click.option("--window", type=int, default=14, show_default=True, help="Lookback window in days (excludes anchor)")
@click.option("--prior-games", type=float, default=20.0, show_default=True, help="Empirical-Bayes prior strength for per-team shrinkage")
def calibrate_totals_cmd(anchor_date: str, window: int, prior_games: float):
    """Compute rolling bias calibration for game and quarter totals.

    Produces data/processed/calibration_totals_<anchor>.json with keys:
    - global: game_total_bias, q{1..4}_bias
    - team: per-team overrides with shrinkage toward global
    """
    console.rule("Calibrate Totals (rolling)")
    try:
        a = pd.to_datetime(anchor_date).date()
    except Exception:
        console.print("Invalid --anchor (YYYY-MM-DD)", style="red"); return
    if window <= 0:
        console.print("--window must be > 0", style="red"); return
    # Build date range excluding anchor
    dates = list(pd.date_range(a - pd.Timedelta(days=window), a - pd.Timedelta(days=1), freq="D").date)
    if not dates:
        console.print("Empty window; nothing to calibrate", style="yellow"); return

    # Helper to winsorize errors (robustness against outliers)
    def _winsorize(s: pd.Series, lo: float = 0.05, hi: float = 0.95) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return s
        ql, qh = s.quantile(lo), s.quantile(hi)
        return s.clip(lower=ql, upper=qh)

    # Accumulate errors across dates
    all_rows: list[pd.DataFrame] = []
    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        # 1) Prefer recon_quarters if exists (already has pred/actual totals)
        rq = paths.data_processed / f"recon_quarters_{ds}.csv"
        if rq.exists():
            try:
                part = pd.read_csv(rq)
                part["date"] = ds
                all_rows.append(part)
                continue
            except Exception:
                pass
        # 2) Fallback: recon_games for the day (game-level totals)
        rg = paths.data_processed / f"recon_games_{ds}.csv"
        if rg.exists():
            try:
                gdf = pd.read_csv(rg)
                cols = {c.lower(): c for c in gdf.columns}
                # recon_games has actual totals but may lack pred_total; if so, merge with predictions
                have_actual = {"home_tri","away_tri","total_actual"}.issubset(set(cols.keys()))
                if have_actual:
                    gsel = gdf[[cols["home_tri"], cols["away_tri"], cols["total_actual"]]].copy()
                    gsel.columns = ["home_tri","away_tri","actual_game_total"]
                    # Attach predictions for pred_game_total
                    from .teams import to_tricode as _tri, normalize_team as _norm
                    pred_candidates = [
                        paths.data_processed / f"games_predictions_npu_{ds}.csv",
                        paths.data_processed / f"predictions_{ds}.csv",
                        paths.root / f"predictions_{ds}.csv",
                    ]
                    pred_path = next((p for p in pred_candidates if p.exists()), None)
                    if pred_path is None:
                        # If recon has pred_total column populated, use it; else skip
                        if "pred_total" in cols:
                            tmp = gdf[[cols["home_tri"], cols["away_tri"], cols["pred_total"], cols["total_actual"]]].copy()
                            tmp.columns = ["home_tri","away_tri","pred_game_total","actual_game_total"]
                            tmp["date"] = ds
                            all_rows.append(tmp)
                            continue
                    else:
                        pr = pd.read_csv(pred_path)
                        pr = pr.copy(); pr["home_tri"] = pr.get("home_team","").astype(str).map(_norm).map(_tri); pr["away_tri"] = pr.get("visitor_team","").astype(str).map(_norm).map(_tri)
                        def _pick_col(df, names):
                            for n in names:
                                if n in df.columns:
                                    return n
                            return None
                        pcol = _pick_col(pr, ["totals","pred_total","model_total","total"])
                        if pcol is not None:
                            pr2 = pr[["home_tri","away_tri", pcol]].copy().rename(columns={pcol: "pred_game_total"})
                            merged = gsel.merge(pr2, on=["home_tri","away_tri"], how="inner")
                            if not merged.empty:
                                merged["date"] = ds
                                all_rows.append(merged[["date","home_tri","away_tri","pred_game_total","actual_game_total"]])
                                continue
            except Exception:
                pass
        # 3) Fallback: join finals_<date>.csv with predictions for that date
        try:
            from .teams import to_tricode as _tri, normalize_team as _norm
            pred_candidates = [
                paths.data_processed / f"games_predictions_npu_{ds}.csv",
                paths.data_processed / f"predictions_{ds}.csv",
                paths.root / f"predictions_{ds}.csv",
            ]
            pred_path = next((p for p in pred_candidates if p.exists()), None)
            finals_path = paths.data_processed / f"finals_{ds}.csv"
            if pred_path is None or (not finals_path.exists()):
                # 4) Last resort: raw line scores if finals missing
                raw_csv = paths.data_raw / "games_nba_api.csv"
                if pred_path is None or (not raw_csv.exists()):
                    continue
                pr = pd.read_csv(pred_path)
                pr = pr.copy(); pr["home_tri"] = pr.get("home_team","").astype(str).map(_norm).map(_tri); pr["away_tri"] = pr.get("visitor_team","").astype(str).map(_norm).map(_tri)
                rw = pd.read_csv(raw_csv)
                rw = rw.copy(); rw["date"] = pd.to_datetime(rw["date"], errors="coerce").dt.strftime("%Y-%m-%d"); rw = rw[rw["date"] == ds]
                rw["home_tri"] = rw["home_team"].astype(str).map(_tri); rw["away_tri"] = rw["visitor_team"].astype(str).map(_tri)
                if pr.empty or rw.empty:
                    continue
                rows_local = []
                for _, rgrow in rw.iterrows():
                    h = str(rgrow.get("home_tri") or "").upper(); a2 = str(rgrow.get("away_tri") or "").upper()
                    m = pr[(pr["home_tri"].astype(str).str.upper()==h) & (pr["away_tri"].astype(str).str.upper()==a2)]
                    if m.empty:
                        continue
                    r0 = m.iloc[0]
                    def _pick(dfrow, names):
                        for n in names:
                            if n in dfrow.index and pd.notna(dfrow.get(n)):
                                return float(dfrow.get(n))
                        return None
                    pg = _pick(r0, ["totals","pred_total","model_total","total"]) or float("nan")
                    at = float(rgrow.get("home_pts") or 0.0) + float(rgrow.get("visitor_pts") or 0.0)
                    rows_local.append({"date": ds, "home_tri": h, "away_tri": a2, "pred_game_total": pg, "actual_game_total": at})
                if rows_local:
                    all_rows.append(pd.DataFrame(rows_local))
            else:
                pr = pd.read_csv(pred_path)
                pr = pr.copy(); pr["home_tri"] = pr.get("home_team","").astype(str).map(_norm).map(_tri); pr["away_tri"] = pr.get("visitor_team","").astype(str).map(_norm).map(_tri)
                fn = pd.read_csv(finals_path)
                # Normalize finals columns
                fcols = {c.lower(): c for c in fn.columns}
                # We need home_tri, away_tri, home_pts, visitor_pts
                needf = {"home_tri","away_tri","home_pts","visitor_pts"}
                if not needf.issubset(set(fcols.keys())):
                    continue
                fn2 = fn[[fcols["home_tri"], fcols["away_tri"], fcols["home_pts"], fcols["visitor_pts"]]].copy()
                fn2.columns = ["home_tri","away_tri","home_pts","visitor_pts"]
                # Select prediction total column
                def _pick_col(df, names):
                    for n in names:
                        if n in df.columns:
                            return n
                    return None
                pred_col = _pick_col(pr, ["totals","pred_total","model_total","total"])
                if pred_col is None:
                    continue
                pr2 = pr[["home_tri","away_tri", pred_col]].copy().rename(columns={pred_col: "pred_game_total"})
                merged = pr2.merge(fn2, on=["home_tri","away_tri"], how="inner")
                if merged.empty:
                    continue
                merged["actual_game_total"] = pd.to_numeric(merged.get("home_pts"), errors="coerce") + pd.to_numeric(merged.get("visitor_pts"), errors="coerce")
                merged["date"] = ds
                out_part = merged[["date","home_tri","away_tri","pred_game_total","actual_game_total"]].copy()
                out_part = out_part.dropna(subset=["pred_game_total","actual_game_total"], how="any")
                if not out_part.empty:
                    all_rows.append(out_part)
        except Exception:
            continue

    if not all_rows:
        console.print("No data found in window; skipping calibration", style="yellow"); return
    df = pd.concat(all_rows, ignore_index=True)
    # Ensure numeric
    df["pred_game_total"] = pd.to_numeric(df.get("pred_game_total", pd.NA), errors="coerce")
    df["actual_game_total"] = pd.to_numeric(df.get("actual_game_total", pd.NA), errors="coerce")
    df = df.dropna(subset=["pred_game_total","actual_game_total"], how="any")
    if df.empty:
        console.print("No comparable rows after filtering", style="yellow"); return
    df["err_game_total"] = df["actual_game_total"] - df["pred_game_total"]

    # Global bias (winsorized mean)
    g_err = _winsorize(df["err_game_total"], 0.05, 0.95)
    global_bias = float(g_err.mean()) if len(g_err) > 0 else 0.0

    # Optional: smart-sim quarter evaluation biases (if a recent eval CSV exists)
    sim_global_bias: float | None = None
    sim_q_biases: dict[str, float] = {}
    sim_h_biases: dict[str, float] = {}
    try:
        candidates = sorted(paths.data_processed.glob("smart_sim_quarter_eval_*.csv"), key=lambda p: p.stat().st_mtime)
        sim_path = candidates[-1] if candidates else None
        if sim_path is not None and sim_path.exists():
            sdf = pd.read_csv(sim_path)
            if isinstance(sdf, pd.DataFrame) and not sdf.empty and "date" in sdf.columns:
                sdf = sdf.copy()
                sdf["date"] = pd.to_datetime(sdf["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                w_start = str(dates[0])
                w_end = str(dates[-1])
                sdf = sdf[(sdf["date"].astype(str) >= w_start) & (sdf["date"].astype(str) <= w_end)]

                def _sim_bias(act_col: str, pred_col: str) -> float | None:
                    if act_col not in sdf.columns or pred_col not in sdf.columns:
                        return None
                    act = pd.to_numeric(sdf[act_col], errors="coerce")
                    pred = pd.to_numeric(sdf[pred_col], errors="coerce")
                    e = _winsorize(act - pred)
                    if e.empty:
                        return None
                    return float(e.mean())

                # Quarter total signed biases (actual - pred)
                for i in (1, 2, 3, 4):
                    b = _sim_bias(f"q{i}_total_act", f"q{i}_total_pred")
                    if b is not None:
                        sim_q_biases[f"q{i}"] = float(b)

                # Half total signed biases (actual - pred)
                for i in (1, 2):
                    b = _sim_bias(f"h{i}_total_act", f"h{i}_total_pred")
                    if b is not None:
                        sim_h_biases[f"h{i}"] = float(b)

                # Game total signed bias (actual - pred) derived from Q1..Q4 when available
                if all((f"q{i}_total_act" in sdf.columns and f"q{i}_total_pred" in sdf.columns) for i in (1, 2, 3, 4)):
                    act_tot = sum(pd.to_numeric(sdf[f"q{i}_total_act"], errors="coerce") for i in (1, 2, 3, 4))
                    pred_tot = sum(pd.to_numeric(sdf[f"q{i}_total_pred"], errors="coerce") for i in (1, 2, 3, 4))
                    e = _winsorize(act_tot - pred_tot)
                    if not e.empty:
                        sim_global_bias = float(e.mean())
    except Exception:
        sim_global_bias = None
        sim_q_biases = {}
        sim_h_biases = {}

    # Optional: per-quarter and per-half biases when available (from recon_quarters)
    q_biases: dict[str, float] = {}
    h_biases: dict[str, float] = {}
    # If df has per-quarter actual/pred columns, compute err and mean
    have_q = all([(f"actual_q{i}_total" in df.columns) and (f"pred_q{i}_total" in df.columns) for i in range(1,5)])
    if have_q:
        for i in range(1,5):
            aq = pd.to_numeric(df.get(f"actual_q{i}_total"), errors="coerce")
            pq = pd.to_numeric(df.get(f"pred_q{i}_total"), errors="coerce")
            e = _winsorize(aq - pq)
            q_biases[f"q{i}"] = float(e.mean()) if len(e) > 0 else 0.0
    have_h = ("actual_h1_total" in df.columns and "pred_h1_total" in df.columns and
              "actual_h2_total" in df.columns and "pred_h2_total" in df.columns)
    if have_h:
        for i in (1,2):
            ah = pd.to_numeric(df.get(f"actual_h{i}_total"), errors="coerce")
            ph = pd.to_numeric(df.get(f"pred_h{i}_total"), errors="coerce")
            e = _winsorize(ah - ph)
            h_biases[f"h{i}"] = float(e.mean()) if len(e) > 0 else 0.0

    # Per-team bias with shrinkage toward global
    # Approximate team predicted points by splitting total using predicted margin if available
    # home_pred_pts = (total + margin)/2; away_pred_pts = total - home_pred_pts
    # If margin unavailable, split evenly
    # Collect per-team rows
    team_rows = []
    # Try to attach margin from predictions if available (best-effort merge by date+matchup)
    # We won't refetch; simply compute team error from finals and implied split
    for _, r in df.iterrows():
        try:
            h = str(r.get("home_tri") or r.get("home_team") or "").upper()
            a2 = str(r.get("away_tri") or r.get("visitor_team") or "").upper()
        except Exception:
            h = None; a2 = None
        if not h or not a2:
            continue
        # Without per-row margin, assume even split for calibration baseline (robust)
        tot = float(r.get("pred_game_total"))
        hp = tot/2.0; ap = tot/2.0
        # Actual team totals require finals; try to load per-date finals quickly
        ds = str(r.get("date"))
        try:
            fin = pd.read_csv(paths.data_processed / f"finals_{ds}.csv")
        except Exception:
            fin = pd.DataFrame()
        if not fin.empty:
            fin["home_tri"] = fin.get("home_tri").astype(str).str.upper(); fin["away_tri"] = fin.get("away_tri").astype(str).str.upper()
            mm = fin[(fin["home_tri"]==h) & (fin["away_tri"]==a2)]
            if not mm.empty:
                hh = pd.to_numeric(mm.iloc[0].get("home_pts"), errors="coerce")
                vv = pd.to_numeric(mm.iloc[0].get("visitor_pts"), errors="coerce")
                if pd.notna(hh):
                    team_rows.append({"team": h, "err": float(hh) - float(hp)})
                if pd.notna(vv):
                    team_rows.append({"team": a2, "err": float(vv) - float(ap)})
    team_df = pd.DataFrame(team_rows)
    team_bias: dict[str, float] = {}
    if not team_df.empty:
        for team, grp in team_df.groupby("team"):
            errs = _winsorize(pd.to_numeric(grp["err"], errors="coerce"))
            n = len(errs)
            if n == 0:
                continue
            mu = float(errs.mean())
            w = float(n) / (float(n) + float(prior_games))
            team_bias[team] = float(w * mu + (1.0 - w) * global_bias)

    calib = {
        "anchor": str(a),
        "window_days": int(window),
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "global": {
            "game_total_bias": float(global_bias),
            **({"sim_game_total_bias": float(sim_global_bias)} if sim_global_bias is not None else {}),
            **({"quarters": q_biases} if q_biases else {}),
            **({"halves": h_biases} if h_biases else {}),
            **({"sim_quarters": sim_q_biases} if sim_q_biases else {}),
            **({"sim_halves": sim_h_biases} if sim_h_biases else {}),
        },
        "team": team_bias,
    }
    out_json = paths.data_processed / f"calibration_totals_{anchor_date}.json"
    try:
        import json
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(calib, f, indent=2)
        console.print({"output": str(out_json), "global_game_total_bias": float(global_bias), "teams": int(len(team_bias))})
    except Exception as e:
        console.print(f"Failed to write calibration JSON: {e}", style="red")


@cli.command("calibrate-period-probs")
@click.option("--anchor", "anchor_date", type=str, required=True, help="Anchor date YYYY-MM-DD (typically yesterday)")
@click.option("--window", type=int, default=30, show_default=True, help="Lookback window in days (excludes anchor)")
@click.option("--bins", type=int, default=10, show_default=True, help="Number of reliability bins")
@click.option("--alpha", type=float, default=1.0, show_default=True, help="Laplace smoothing strength")
@click.option("--smart-sim-path", type=click.Path(dir_okay=False), required=False, help="Optional explicit smart_sim_quarter_eval CSV path")
def calibrate_period_probs_cmd(anchor_date: str, window: int, bins: int, alpha: float, smart_sim_path: str | None):
    """Calibrate period over probabilities using smart_sim_quarter_eval_*.csv.

    Produces data/processed/calibration_period_probs_<anchor>.json.

    Calibration is simple binning (reliability curve) with Laplace smoothing.
    It is intentionally sklearn-free for ARM64 friendliness.
    """
    console.rule("Calibrate Period Probabilities")
    try:
        a = pd.to_datetime(anchor_date).date()
    except Exception:
        console.print("Invalid --anchor (YYYY-MM-DD)", style="red")
        return
    if window <= 0:
        console.print("--window must be > 0", style="red")
        return
    bins = int(max(2, min(50, int(bins))))
    alpha = float(alpha) if alpha is not None else 1.0
    alpha = float(max(0.0, min(10.0, alpha)))

    # Window includes anchor (typically yesterday) so the freshest completed games
    # are incorporated into calibration. The runtime lookup uses <= (date-1), so
    # an artifact dated yesterday is eligible for today's slate.
    w_start = (a - pd.Timedelta(days=max(1, window) - 1)).strftime("%Y-%m-%d")
    w_end = a.strftime("%Y-%m-%d")

    sim_path = None
    try:
        if smart_sim_path:
            sim_path = Path(str(smart_sim_path))
        else:
            candidates = sorted(paths.data_processed.glob("smart_sim_quarter_eval_*.csv"), key=lambda p: p.stat().st_mtime)
            sim_path = candidates[-1] if candidates else None
    except Exception:
        sim_path = None

    if sim_path is None or (not sim_path.exists()):
        console.print("No smart_sim_quarter_eval_*.csv found; cannot calibrate period probs", style="yellow")
        return

    try:
        sdf = pd.read_csv(sim_path)
    except Exception as e:
        console.print(f"Failed to read {sim_path}: {e}", style="red")
        return

    if sdf is None or sdf.empty or "date" not in sdf.columns:
        console.print("smart_sim_quarter_eval file missing 'date' or is empty", style="yellow")
        return

    sdf = sdf.copy()
    sdf["date"] = pd.to_datetime(sdf["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    sdf = sdf[(sdf["date"].astype(str) >= w_start) & (sdf["date"].astype(str) <= w_end)]
    if sdf.empty:
        console.print({"warning": "no rows in window", "start": w_start, "end": w_end, "path": str(sim_path)})
        return

    def _fit(market: str) -> dict[str, object] | None:
        pcol = f"{market}_p"
        ycol = f"{market}_y"
        if pcol not in sdf.columns or ycol not in sdf.columns:
            return None
        p = pd.to_numeric(sdf[pcol], errors="coerce")
        y = pd.to_numeric(sdf[ycol], errors="coerce")
        m = p.notna() & y.notna() & (p >= 0.0) & (p <= 1.0)
        n_total = int(m.sum())
        if n_total < 10:
            return None
        p = p[m].astype(float)
        y = y[m].astype(float)

        # Adaptive bins for sparse markets
        n_bins = int(max(3, min(int(bins), int(round(float(np.sqrt(n_total)))))))
        edges = np.linspace(0.0, 1.0, n_bins + 1)

        # Bin by predicted p
        idx = np.digitize(p.to_numpy(), edges[1:-1], right=False)
        p_cal: list[float] = []
        n_bin: list[int] = []
        for bi in range(int(n_bins)):
            mm = idx == bi
            n = int(np.sum(mm))
            if n <= 0:
                # No data -> neutral
                p_cal.append(0.5)
                n_bin.append(0)
                continue
            yy = float(np.sum(y.to_numpy()[mm]))
            # Laplace smoothing: (yy + alpha) / (n + 2*alpha)
            denom = float(n + 2.0 * alpha) if alpha > 0 else float(n)
            num = float(yy + alpha) if alpha > 0 else float(yy)
            pc = float(num / max(1e-9, denom))
            p_cal.append(float(max(0.0, min(1.0, pc))))
            n_bin.append(n)

        # Enforce monotonicity (reliability curve should be non-decreasing)
        try:
            mono = []
            cur = 0.0
            for v in p_cal:
                cur = max(cur, float(v))
                mono.append(cur)
            p_cal = mono
        except Exception:
            pass

        return {
            "bin_edges": [float(x) for x in edges.tolist()],
            "p_cal": [float(x) for x in p_cal],
            "n_bin": n_bin,
            "n": int(len(p)),
        }

    markets_out: dict[str, object] = {}
    for base in (
        "q1_over", "q2_over", "q3_over", "q4_over", "h1_over", "h2_over",
        "q1_cover", "q2_cover", "q3_cover", "q4_cover", "h1_cover", "h2_cover",
    ):
        fit = _fit(base)
        if fit is not None:
            markets_out[base] = fit

    if not markets_out:
        console.print("No eligible markets found (need *_p and *_y with enough rows)", style="yellow")
        return

    out = {
        "anchor": str(a),
        "window_days": int(window),
        "bins": int(bins),
        "alpha": float(alpha),
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "source": str(sim_path),
        "window": {"start": str(w_start), "end": str(w_end)},
        "markets": markets_out,
    }
    out_json = paths.data_processed / f"calibration_period_probs_{anchor_date}.json"
    try:
        import json
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        console.print({"output": str(out_json), "markets": sorted(list(markets_out.keys()))})
    except Exception as e:
        console.print(f"Failed to write period probs calibration JSON: {e}", style="red")


@cli.command("apply-totals-calibration")
@click.option("--date", "date_str", type=str, required=True, help="Target slate date YYYY-MM-DD to adjust predictions for")
@click.option("--calib-date", type=str, required=False, help="Calibration anchor date; defaults to yesterday")
@click.option("--in", "in_path", type=click.Path(dir_okay=False), required=False, help="Input predictions CSV (defaults to data/processed/predictions_<date>.csv if present, else games_predictions_npu_<date>.csv)")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=False, help="Output CSV; default overwrites input")
def apply_totals_calibration_cmd(date_str: str, calib_date: str | None, in_path: str | None, out_path: str | None):
    """Apply saved totals calibration JSON to a predictions CSV.

    Adjusts:
    - Game total: totals/pred_total columns
    - Halves/quarters totals if present (evenly distributes delta)
    """
    console.rule("Apply Totals Calibration")
    try:
        target = pd.to_datetime(date_str).date()
    except Exception:
        console.print("Invalid --date (YYYY-MM-DD)", style="red"); return
    if calib_date:
        try:
            _ = pd.to_datetime(calib_date).date()
        except Exception:
            console.print("Invalid --calib-date (YYYY-MM-DD)", style="red"); return
    else:
        # Default to yesterday
        from datetime import date as _d, timedelta as _td
        calib_date = (_d.today() - _td(days=1)).isoformat()

    calib_path = paths.data_processed / f"calibration_totals_{calib_date}.json"
    if not calib_path.exists():
        console.print(f"Calibration not found: {calib_path}", style="yellow"); return
    try:
        import json
        with open(calib_path, "r", encoding="utf-8") as f:
            calib = json.load(f)
    except Exception as e:
        console.print(f"Failed to read calibration JSON: {e}", style="red"); return

    # Select input predictions
    in_file: Path
    if in_path:
        in_file = Path(in_path)
    else:
        cands = [
            paths.data_processed / f"predictions_{date_str}.csv",
            paths.data_processed / f"games_predictions_npu_{date_str}.csv",
            paths.root / f"predictions_{date_str}.csv",
        ]
        in_file = next((p for p in cands if p.exists()), None)  # type: ignore
        if in_file is None:
            console.print(f"Predictions not found for {date_str}", style="red"); return
    try:
        df = pd.read_csv(in_file)
    except Exception as e:
        console.print(f"Failed to read predictions CSV: {e}", style="red"); return
    if df is None or df.empty:
        console.print("Predictions CSV is empty; nothing to adjust", style="yellow"); return

    from .teams import to_tricode as _to_tri, normalize_team as _norm
    df = df.copy()
    for c in ("home_team","visitor_team"):
        if c in df.columns:
            df[c] = df[c].apply(_norm)
        else:
            df[c] = pd.NA
    df["home_tri"] = df["home_team"].astype(str).map(_to_tri)
    df["away_tri"] = df["visitor_team"].astype(str).map(_to_tri)

    # Fetch calibration pieces
    g = calib.get("global", {}) or {}
    g_bias = float(g.get("game_total_bias", 0.0) or 0.0)
    g_quarters = (g.get("quarters") or {}) if isinstance(g.get("quarters"), dict) else {}
    g_halves = (g.get("halves") or {}) if isinstance(g.get("halves"), dict) else {}
    team_bias: dict = calib.get("team", {}) or {}

    # Column helpers
    def _pick_col(names: list[str]) -> str | None:
        for n in names:
            if n in df.columns:
                return n
        return None
    game_col = _pick_col(["totals","pred_total","model_total","total"])  # adjust this
    # Period columns presence
    q_cols = [c for c in [f"quarters_q{i}_total" for i in range(1,5)] if c in df.columns]
    h_cols = [c for c in [f"halves_h1_total", "halves_h2_total"] if c in df.columns]

    def _weights_from_bias_map(bmap: dict[str, float], keys: list[str]) -> list[float]:
        # Produce non-negative weights from a bias map; if degenerate, return equal weights
        vals = [float(bmap.get(k, 0.0) or 0.0) for k in keys]
        if not vals:
            return []
        mn = min(vals)
        w = [max(0.0, v - mn) for v in vals]
        s = sum(w)
        if s <= 1e-9:
            return [1.0/float(len(vals)) for _ in vals]
        return [x/s for x in w]

    # Precompute weights if any
    q_keys = [f"q{i}" for i in range(1,5)]
    h_keys = [f"h{i}" for i in range(1,3)]
    q_w = _weights_from_bias_map(g_quarters, q_keys) if g_quarters else []
    h_w = _weights_from_bias_map(g_halves, h_keys) if g_halves else []

    # Apply per-row
    def _row_adjust(r: pd.Series) -> pd.Series:
        try:
            h = str(r.get("home_tri") or "").upper(); a2 = str(r.get("away_tri") or "").upper()
        except Exception:
            h = ""; a2 = ""
        tb_h = float(team_bias.get(h, 0.0) or 0.0) if h else 0.0
        tb_a = float(team_bias.get(a2, 0.0) or 0.0) if a2 else 0.0
        delta = float(g_bias + 0.5 * (tb_h + tb_a))
        # Game total
        if game_col and pd.notna(r.get(game_col)):
            try:
                r[game_col] = float(r.get(game_col)) + delta
            except Exception:
                pass
        # Distribute to halves/quarters if present
        if q_cols:
            if q_w and len(q_w) == len(q_cols):
                adds = [delta * w for w in q_w]
            else:
                adds = [delta/4.0 for _ in q_cols]
            for qc, add in zip(q_cols, adds):
                if pd.notna(r.get(qc)):
                    try:
                        r[qc] = float(r.get(qc)) + float(add)
                    except Exception:
                        pass
        if h_cols:
            if h_w and len(h_w) == len(h_cols):
                adds_h = [delta * w for w in h_w]
            else:
                adds_h = [delta/2.0 for _ in h_cols]
            for hc, addh in zip(h_cols, adds_h):
                if pd.notna(r.get(hc)):
                    try:
                        r[hc] = float(r.get(hc)) + float(addh)
                    except Exception:
                        pass
        return r

    df = df.apply(_row_adjust, axis=1)

    # Output
    if out_path:
        out_file = Path(out_path)
    else:
        out_file = in_file
    df.to_csv(out_file, index=False)
    console.print({"input": str(in_file), "output": str(out_file), "calibration": str(calib_path)})


@cli.command("first-basket-recs")
@click.option("--date", "date_str", type=str, required=True, help="Target date YYYY-MM-DD")
@click.option("--topk", type=int, default=3, show_default=True, help="Max picks per game")
@click.option("--min-prob", "min_prob", type=float, default=0.08, show_default=True, help="Minimum probability threshold to include a pick")
@click.option("--cum-target", "cum_target", type=float, default=0.45, show_default=True, help="Stop adding picks when cumulative coverage reaches this threshold")
def first_basket_recs_cmd(date_str: str, topk: int, min_prob: float, cum_target: float):
    """Export first-basket scorer recommendations per game using model probabilities.

    Produces data/processed/first_basket_recs_<date>.csv with columns:
    - date, game_id, team, player_id, player_name, prob_first_basket
    - fair_decimal, fair_american, rank, cum_prob
    """
    console.rule("First Basket: Recommendations")
    probs_path = paths.data_processed / f"first_basket_probs_{date_str}.csv"
    if not probs_path.exists():
        console.print(f"Missing probabilities: {probs_path}", style="red"); return
    try:
        df = pd.read_csv(probs_path)
    except Exception as e:
        console.print(f"Failed to read probabilities: {e}", style="red"); return
    required = {"game_id","team","player_id","player_name","prob_first_basket"}
    if not required.issubset(set(df.columns)):
        console.print(f"Missing required columns in {probs_path.name}: {required - set(df.columns)}", style="red"); return
    # Normalize gid
    df = df.copy()
    # Normalize game_id robustly: coerce to integer (drops any stray decimals), then zero-pad to 10
    try:
        _gid_num = pd.to_numeric(df["game_id"], errors="coerce").astype("Int64")
        df["gid10"] = _gid_num.astype(str).str.replace("<NA>", "", regex=False).str.zfill(10)
    except Exception:
        df["gid10"] = df["game_id"].astype(str).str.replace(".0$","", regex=True).str.replace("^nan$","", regex=True).str.replace("^None$","", regex=True).str.replace("^\\s+$","", regex=True).str.zfill(10)
    # Per-game selections
    rows: list[dict] = []
    for gid, grp in df.groupby("gid10"):
        sub = grp.sort_values("prob_first_basket", ascending=False).copy()
        cum = 0.0; picked = 0; rank = 0
        for _, r in sub.iterrows():
            p = float(r.get("prob_first_basket", 0.0) or 0.0)
            if p < float(min_prob):
                continue
            rank += 1
            # Fair odds
            fair_dec = float("inf") if p <= 0 else (1.0/float(p))
            if p >= 0.5:
                # negative american
                fair_am = -round((p/(1.0-p))*100)
            else:
                fair_am = round(((1.0-p)/p)*100)
            rows.append({
                "date": date_str,
                "game_id": gid,
                "team": r.get("team"),
                "player_id": r.get("player_id"),
                "player_name": r.get("player_name"),
                "prob_first_basket": p,
                "fair_decimal": fair_dec,
                "fair_american": fair_am,
                "rank": rank,
                "cum_prob": cum + p,
            })
            picked += 1
            cum += p
            if picked >= int(topk) or cum >= float(cum_target):
                break
    out = pd.DataFrame(rows)
    out_path = paths.data_processed / f"first_basket_recs_{date_str}.csv"
    out.to_csv(out_path, index=False)
    console.print({"date": date_str, "games": int(out["game_id"].nunique() if not out.empty else 0), "rows": int(len(out)), "output": str(out_path)})


@cli.command("backtest-pbp-markets")
@click.option("--start", "start_date", type=str, required=True, help="Start date YYYY-MM-DD (season start)")
@click.option("--end", "end_date", type=str, required=False, help="End date YYYY-MM-DD; default=today")
@click.option("--ensure-preds", is_flag=True, default=True, show_default=True, help="Generate predictions for any missing days in range")
@click.option("--ensure-pbp", is_flag=True, default=False, show_default=True, help="Fetch PBP logs if missing for games in range (finals only)")
def backtest_pbp_markets_cmd(start_date: str, end_date: str | None, ensure_preds: bool, ensure_pbp: bool):
    """Backtest PBP-derived markets (tip, first-basket, early-threes) over a date range.

    Metrics reported:
    - tip: Brier score, log loss (if outcome available), accuracy at 0.5 threshold, n
    - first-basket: top-1 accuracy, top-5 coverage, mean prob(actual), n
    - early-threes: MAE, RMSE for expected_threes_0_3, and Brier for prob_ge_1 vs indicator, n
    """
    console.rule("Backtest PBP-derived Markets")
    try:
        s = pd.to_datetime(start_date).date()
    except Exception:
        console.print("Invalid --start (YYYY-MM-DD)", style="red"); return
    if end_date:
        try:
            e = pd.to_datetime(end_date).date()
        except Exception:
            console.print("Invalid --end (YYYY-MM-DD)", style="red"); return
    else:
        e = _date.today()
    if e < s:
        console.print("--end must be >= --start", style="red"); return

    # Optionally ensure PBP exists (finals only)
    if ensure_pbp:
        try:
            _ = backfill_pbp(str(s), str(e), only_final=True, rate_delay=0.35)
        except Exception as ex:
            console.print({"warning": f"PBP fetch failed/skipped: {ex}"}, style="yellow")

    # Iterate days and evaluate
    dates = list(pd.date_range(s, e, freq="D").date)
    # Accumulators
    tip_probs = []; tip_outcomes = []
    fb_top1 = 0; fb_top5 = 0; fb_total = 0; fb_probs_actual = []
    thr_errs = []; thr_ge1_probs = []; thr_ge1_outcomes = []
    # Lightweight cache for home/away maps
    _gid_to_homeaway: dict[str, tuple[str,str]] = {}

    def _map_from_schedule(ds: str) -> dict[str, tuple[str,str]]:
        # Prefer local processed schedule to avoid network
        import glob
        from pathlib import Path as _Path
        root = paths.data_processed
        out: dict[str, tuple[str,str]] = {}
        try:
            # Try season schedule first
            cand = list(root.glob("schedule_*.csv"))
            if not cand:
                return {}
            p = cand[0]
            # Prefer 2025_26 if present
            for c in cand:
                if "2025_26" in c.name:
                    p = c; break
            df = pd.read_csv(p)
            cols = {c.lower(): c for c in df.columns}
            if not {"game_id","date_utc","home_tricode","away_tricode"}.issubset(set(cols)):
                return {}
            df[cols["date_utc"]] = pd.to_datetime(df[cols["date_utc"]], errors="coerce").dt.strftime("%Y-%m-%d")
            day = df[df[cols["date_utc"]] == ds].copy()
            if day.empty:
                return {}
            for _, r in day.iterrows():
                gid = str(r[cols["game_id"]]).strip()
                home = str(r[cols["home_tricode"]]).upper().strip()
                away = str(r[cols["away_tricode"]]).upper().strip()
                if gid and home and away:
                    out[gid] = (home, away)
            return out
        except Exception:
            return {}

    def _cdn_map_for_date(ds: str) -> dict[str, tuple[str,str]]:
        # Fallback to CDN if schedule missing
        from .pbp_markets import _gid_team_map_for_date as _map
        try:
            m = _map(ds) or {}
            return {str(k): (v[0], v[1]) for k, v in m.items()}
        except Exception:
            return {}

    def _read_csv_safe(path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    def _norm_gid(x) -> str:
        try:
            sx = str(x).strip()
            # Map CDN schedule ids like '2250xxxx' to official '002250xxxx'
            if len(sx) == 8 and sx.startswith("2250"):
                sx = "00" + sx
            return sx.zfill(10) if sx.isdigit() else sx
        except Exception:
            return str(x)

    for d in track(dates, description="Backtesting"):
        ds = str(d)
        # Load or generate predictions
        tip_path = paths.data_processed / f"tip_winner_probs_{ds}.csv"
        fb_path = paths.data_processed / f"first_basket_probs_{ds}.csv"
        thr_path = paths.data_processed / f"early_threes_{ds}.csv"
        if ensure_preds:
            try:
                if not tip_path.exists() or not fb_path.exists() or not thr_path.exists():
                    _ = predict_tip_for_date(ds)
                    _ = predict_first_basket_for_date(ds)
                    _ = predict_early_threes_for_date(ds)
            except Exception:
                pass
        # Read available predictions
        tip = _read_csv_safe(tip_path) if tip_path.exists() else pd.DataFrame()
        fb = _read_csv_safe(fb_path) if fb_path.exists() else pd.DataFrame()
        thr = _read_csv_safe(thr_path) if thr_path.exists() else pd.DataFrame()
        if tip.empty and fb.empty and thr.empty:
            continue

        # Load PBP combined for outcomes (if exists)
        pbp_comb = paths.data_processed / f"pbp_{ds}.csv"
        pbp_map: dict[str, pd.DataFrame] = {}
        if pbp_comb.exists():
            df = pd.read_csv(pbp_comb)
            if "game_id" in df.columns:
                for gid, grp in df.groupby("game_id"):
                    key = _norm_gid(gid)
                    # Only accept well-formed numeric game ids after normalization
                    if key.isdigit() and len(key) == 10:
                        pbp_map[key] = grp.copy()
        # Fallback to per-game files if combined missing or yielded no valid keys
        if not pbp_map:
            dpg = paths.data_processed / "pbp"
            if dpg.exists():
                for f in dpg.glob("pbp_*.csv"):
                    try:
                        gid = f.stem.replace("pbp_", "").strip()
                        key = _norm_gid(gid)
                        if key.isdigit() and len(key) == 10:
                            pbp_map[key] = pd.read_csv(f)
                    except Exception:
                        continue

        # Early threes outcomes and errors
        if not thr.empty:
            # Build actuals for games present in thr
            actuals = {}
            if pbp_map:
                for gid, gdf in pbp_map.items():
                    # count threes in first 180 sec using existing helper
                    cnt = 0
                    desc_cols = _pbp_desc_cols(gdf)
                    c_time = "PCTIMESTRING" if "PCTIMESTRING" in gdf.columns else ("clock" if "clock" in gdf.columns else None)
                    c_per = "PERIOD" if "PERIOD" in gdf.columns else ("period" if "period" in gdf.columns else None)
                    tmp = gdf.copy()
                    if c_per: tmp = tmp[tmp[c_per] == 1]
                    if c_time: tmp = tmp.sort_values(c_time, ascending=False)
                    for _, r in tmp.iterrows():
                        t = r.get("PCTIMESTRING") or r.get("clock") or r.get("time")
                        sec_left = None
                        if isinstance(t, str) and ":" in t:
                            try:
                                m, s2 = t.split(":"); sec_left = int(m)*60+int(s2)
                            except Exception:
                                sec_left = None
                        if sec_left is None: continue
                        elapsed = 12*60 - sec_left
                        if elapsed is None or elapsed > 180:
                            continue
                        text = " ".join([str(r.get(c, "")) for c in desc_cols]).lower()
                        if ("3pt" in text) and ("makes" in text or "made" in text):
                            cnt += 1
                    actuals[str(gid)] = int(cnt)
            # Join predictions with actuals
            for _, row in thr.iterrows():
                gid = _norm_gid(row.get("game_id"))
                if not gid:
                    continue
                yhat = float(row.get("expected_threes_0_3", row.get("threes_0_3_pred", 0.0)) or 0.0)
                a = actuals.get(gid)
                if a is None:
                    continue
                thr_errs.append(float(a - yhat))
                p_ge1 = float(row.get("prob_ge_1", 1.0 - float(np.exp(-max(0.0, yhat)))))
                thr_ge1_probs.append(p_ge1)
                thr_ge1_outcomes.append(1.0 if a >= 1 else 0.0)

        # Optionally read reconciliation for outcomes (fast path)
        rec_path = paths.data_processed / f"pbp_reconcile_{ds}.csv"
        rec_df = None
        if rec_path.exists():
            try:
                rec_df = pd.read_csv(rec_path)
                if not rec_df.empty and "game_id" in rec_df.columns:
                    rec_df["game_id_norm"] = rec_df["game_id"].astype(str).str.strip().apply(_norm_gid)
            except Exception:
                rec_df = None

        # Early threes via reconciliation (preferred if available and none computed yet)
        if (not thr.empty) and (len(thr_errs) == 0) and (rec_df is not None) and {"game_id_norm","early_threes_actual"}.issubset(set(rec_df.columns)):
            try:
                thr2 = thr.copy()
                thr2["game_id_norm"] = thr2["game_id"].apply(_norm_gid)
                merged = thr2.merge(rec_df[["game_id_norm","early_threes_actual"]], on="game_id_norm", how="inner")
                merged = merged.dropna(subset=["early_threes_actual"]).copy()
                for _, rr in merged.iterrows():
                    yhat = float(rr.get("expected_threes_0_3", rr.get("threes_0_3_pred", 0.0)) or 0.0)
                    a = float(rr.get("early_threes_actual"))
                    thr_errs.append(float(a - yhat))
                    p_ge1 = float(rr.get("prob_ge_1", 1.0 - float(np.exp(-max(0.0, yhat)))))
                    thr_ge1_probs.append(p_ge1)
                    thr_ge1_outcomes.append(1.0 if a >= 1.0 else 0.0)
            except Exception:
                pass

        # First basket outcomes
        if not fb.empty and pbp_map:
            # Map actual first scorer per game
            actual_first: dict[str, dict] = {}
            for gid, gdf in pbp_map.items():
                ev = _pbp_first_fg_event(gdf)
                if ev:
                    actual_first[str(gid)] = ev
            for gid, grp in fb.groupby("game_id"):
                gid_n = _norm_gid(gid)
                ev = actual_first.get(gid_n)
                if not ev:
                    continue
                fb_total += 1
                pname_first = (ev.get("player_name") or "").strip().lower()
                # Top-1
                sub = grp.copy().sort_values("prob_first_basket", ascending=False)
                top = sub.iloc[0]
                top_hit = pname_first in str(top.get("player_name","")) .lower()
                if top_hit:
                    fb_top1 += 1
                # Top-5 coverage and prob(actual)
                hit5 = False; prob_actual = None
                for _, r in sub.head(5).iterrows():
                    if pname_first in str(r.get("player_name","")) .lower():
                        hit5 = True; prob_actual = float(r.get("prob_first_basket", np.nan)); break
                if hit5:
                    fb_top5 += 1
                if prob_actual is None:
                    m = sub[sub["player_name"].astype(str).str.lower().str.contains(pname_first)].head(1)
                    if not m.empty:
                        prob_actual = float(m.iloc[0].get("prob_first_basket", np.nan))
                if prob_actual is not None and not np.isnan(prob_actual):
                    fb_probs_actual.append(float(prob_actual))

        # Tip outcomes
        if not tip.empty and (pbp_map or (rec_df is not None)):
            # If reconciliation file exists, prefer its tip outcomes for simplicity
            if rec_df is not None and {"game_id_norm","tip_outcome_home"}.issubset(set(rec_df.columns)):
                tip2 = tip.copy()
                tip2["game_id_norm"] = tip2["game_id"].apply(_norm_gid)
                merged = tip2.merge(rec_df[["game_id_norm","tip_outcome_home"]], on="game_id_norm", how="inner")
                merged = merged.dropna(subset=["tip_outcome_home"]).copy()
                for _, rr in merged.iterrows():
                    p_home = float(rr.get("prob_home_tip", 0.5))
                    outcome = float(rr.get("tip_outcome_home"))
                    tip_probs.append(p_home)
                    tip_outcomes.append(outcome)
                # Skip the PBP path if reconcile provided outcomes for this date
                continue
            # Build home/away map for this date
            if not _gid_to_homeaway:
                # Prefer local schedule mapping; fallback to CDN
                _gid_to_homeaway = _map_from_schedule(ds) or _cdn_map_for_date(ds)
            for _, r in tip.iterrows():
                gid_n = _norm_gid(r.get("game_id"))
                if gid_n not in pbp_map:
                    continue
                pbp = pbp_map[gid_n]
                ev = _pbp_jump_ball_event(pbp)
                if not ev:
                    continue
                winner_text = (ev.get("winner_text") or "").strip().lower()
                if not winner_text:
                    continue
                # Try both zero-padded and raw keys when mapping home/away
                raw_gid = gid_n.lstrip("0") or gid_n
                home, away = (
                    _gid_to_homeaway.get(gid_n)
                    or _gid_to_homeaway.get(raw_gid)
                    or _gid_to_homeaway.get(gid_n.zfill(10))
                    or (None, None)
                )
                if not (home and away):
                    continue
                outcome = None
                try:
                    from .pbp_markets import _load_rosters_latest as _load_rost
                    rost = _load_rost()
                    if not rost.empty:
                        def _has_name(team):
                            tri_col = "TEAM_ABBREVIATION" if "TEAM_ABBREVIATION" in rost.columns else ("teamTricode" if "teamTricode" in rost.columns else None)
                            sub = rost[rost[tri_col].astype(str).str.upper() == str(team).upper()].copy() if tri_col else rost
                            names = sub.get("PLAYER") or sub.get("PLAYER_NAME") or pd.Series(dtype=str)
                            names = names.astype(str).str.lower()
                            # More robust: match last token and any tokenized subset
                            toks = [t for t in re.split(r"\s+", winner_text) if t]
                            if not toks:
                                return False
                            pat = r"|".join([re.escape(t) for t in toks[-2:]])  # last 1-2 tokens
                            return names.str.contains(pat, regex=True).any()
                        if _has_name(home): outcome = 1.0
                        elif _has_name(away): outcome = 0.0
                except Exception:
                    outcome = None

                # Fallback: infer winner team from PBP playerName occurrences if roster match failed
                if outcome is None:
                    try:
                        cols = set(pbp.columns)
                        name_series = None
                        for c in ("playerName","PLAYER1_NAME","player1_name"):
                            if c in cols:
                                name_series = pbp[c].astype(str).str.lower()
                                break
                        team_series = None
                        for c in ("teamTricode","PLAYER1_TEAM_ABBREVIATION","team_abbr"):
                            if c in cols:
                                team_series = pbp[c].astype(str).str.upper()
                                break
                        if name_series is not None and team_series is not None:
                            toks = [t for t in re.split(r"\s+", winner_text) if t]
                            if toks:
                                pat = r"|".join([re.escape(t) for t in toks[-2:]])
                                mask = name_series.str.contains(pat, regex=True)
                                teams = team_series[mask].value_counts()
                                if not teams.empty:
                                    winner_tri = teams.index[0]
                                    if str(winner_tri).upper() == str(home).upper():
                                        outcome = 1.0
                                    elif str(winner_tri).upper() == str(away).upper():
                                        outcome = 0.0
                    except Exception:
                        pass
                p_home = float(r.get("prob_home_tip", 0.5))
                if outcome is not None:
                    tip_probs.append(p_home)
                    tip_outcomes.append(outcome)

    # Aggregate metrics
    def _brier(p, y):
        return float(np.mean([(pi-yi)**2 for pi, yi in zip(p, y)])) if p and y else np.nan
    def _logloss(p, y, eps=1e-9):
        import math
        if not p or not y:
            return np.nan
        return float(np.mean([-(yi*math.log(max(eps, pi)) + (1-yi)*math.log(max(eps, 1-pi))) for pi, yi in zip(p, y)]))
    tip_brier = _brier(tip_probs, tip_outcomes)
    tip_logloss = _logloss(tip_probs, tip_outcomes)
    tip_acc = float(np.mean([int((pi>=0.5)==(yi==1.0)) for pi, yi in zip(tip_probs, tip_outcomes)])) if tip_probs else np.nan
    fb_top1_acc = (fb_top1 / fb_total) if fb_total else np.nan
    fb_top5_cov = (fb_top5 / fb_total) if fb_total else np.nan
    fb_mean_prob_actual = float(np.mean(fb_probs_actual)) if fb_probs_actual else np.nan
    thr_mae = float(np.mean([abs(e) for e in thr_errs])) if thr_errs else np.nan
    thr_rmse = float(np.sqrt(np.mean([e*e for e in thr_errs]))) if thr_errs else np.nan
    thr_brier = _brier(thr_ge1_probs, thr_ge1_outcomes)

    console.print({
        "range": f"{start_date}..{str(e)}",
        "tip": {"n": len(tip_outcomes), "brier": tip_brier, "logloss": tip_logloss, "acc@0.5": tip_acc},
        "first_basket": {"n": fb_total, "top1_acc": fb_top1_acc, "top5_cov": fb_top5_cov, "mean_prob_actual": fb_mean_prob_actual},
        "early_threes": {"n": len(thr_errs), "mae": thr_mae, "rmse": thr_rmse, "brier_ge1": thr_brier},
    })


@cli.command("calibrate-pbp-markets")
@click.option("--anchor", "anchor_date", type=str, required=False, help="Anchor date YYYY-MM-DD (default=yesterday)")
@click.option("--window", "window_days", type=int, default=7, show_default=True, help="Lookback window in days for calibration")
def calibrate_pbp_markets_cmd(anchor_date: str | None, window_days: int):
    """Compute lightweight calibration for PBP markets using recent reconciliation files.

    Outputs to data/processed/pbp_calibration.csv (appends a row):
    - thr_bias: additive bias for expected_threes_0_3 (early threes)
    - tip_logit_bias: intercept shift in log-odds space for tip prob_home
    - fb_temp: temperature scaling for first-basket candidate score normalization
    """
    console.rule("Calibrate PBP-derived Markets")
    import datetime as _dt
    if anchor_date:
        try:
            anchor = pd.to_datetime(anchor_date).date()
        except Exception:
            console.print("Invalid --anchor (YYYY-MM-DD)", style="red"); return
    else:
        anchor = (_dt.date.today() - _dt.timedelta(days=1))
    start = anchor - _dt.timedelta(days=max(0, int(window_days)-1))
    dates = list(pd.date_range(start, anchor, freq="D").date)
    rows = []
    for d in dates:
        p = paths.data_processed / f"pbp_reconcile_{d}.csv"
        if p.exists():
            try:
                df = pd.read_csv(p)
                df["date"] = str(d)
                rows.append(df)
            except Exception:
                continue
    if not rows:
        console.print("No reconciliation files in window; skipping calibration", style="yellow"); return
    all_df = pd.concat(rows, ignore_index=True)
    # Early threes intercept bias = mean(actual - expected)
    thr_err = pd.to_numeric(all_df.get("early_threes_error"), errors="coerce").dropna()
    thr_bias = float(thr_err.mean()) if len(thr_err)>0 else 0.0

    # Tip: fit a logit intercept b such that mean(sigmoid(logit(p)+b)) ~= mean(y)
    tip_b = 0.0
    try:
        p = pd.to_numeric(all_df.get("tip_prob_home"), errors="coerce").dropna().astype(float).tolist()
        y = pd.to_numeric(all_df.get("tip_outcome_home"), errors="coerce").dropna().astype(float).tolist()
        # Align lengths by inner join on rows with both
        if "tip_prob_home" in all_df.columns and "tip_outcome_home" in all_df.columns:
            tmp = all_df.dropna(subset=["tip_prob_home","tip_outcome_home"]).copy()
            pp = pd.to_numeric(tmp["tip_prob_home"], errors="coerce").astype(float).tolist()
            yy = pd.to_numeric(tmp["tip_outcome_home"], errors="coerce").astype(float).tolist()
            if len(pp) >= 5:
                import math
                probs = [min(max(float(t), 1e-6), 1-1e-6) for t in pp]
                ys = [1.0 if float(t)>=0.5 else 0.0 for t in yy]  # ensure 0/1
                ybar = float(np.mean(ys)) if ys else None
                if ybar is not None:
                    logits = [math.log(pv/(1-pv)) for pv in probs]
                    # Bisection to find b s.t. mean(sigmoid(logit+b)) = ybar
                    def _mean_after(b):
                        vals = [1.0/(1.0+math.exp(-(lv + b))) for lv in logits]
                        return float(np.mean(vals))
                    lo, hi = -5.0, 5.0
                    m_lo, m_hi = _mean_after(lo), _mean_after(hi)
                    # If ybar out of achievable range, pick closest bound
                    if ybar <= m_lo:
                        tip_b = lo
                    elif ybar >= m_hi:
                        tip_b = hi
                    else:
                        for _ in range(40):
                            mid = 0.5*(lo+hi)
                            m_mid = _mean_after(mid)
                            if m_mid < ybar:
                                lo = mid
                            else:
                                hi = mid
                        tip_b = 0.5*(lo+hi)
    except Exception:
        tip_b = 0.0

    # First-basket: choose temperature that maximizes mean probability assigned to actual first scorer
    fb_temp = 1.0
    try:
        # Build per-game actual names from reconciliation in window
        recon = all_df.dropna(subset=["game_id","first_basket_actual_name"]).copy()
        if not recon.empty:
            recon["game_id"] = recon["game_id"].astype(str)
            # Load candidate audits per day in window
            candidate_frames = []
            for d in dates:
                f = paths.data_processed / f"first_basket_candidates_{d}.csv"
                if f.exists():
                    try:
                        cdf = pd.read_csv(f)
                        cdf["game_id"] = cdf["game_id"].astype(str)
                        candidate_frames.append(cdf)
                    except Exception:
                        continue
            if candidate_frames:
                all_cand = pd.concat(candidate_frames, ignore_index=True)
                # For stability add epsilon to raw_score and clip >= 0
                all_cand["raw_score"] = pd.to_numeric(all_cand.get("raw_score"), errors="coerce").fillna(0.0).clip(lower=0.0) + 1e-9
                # Grid search tau in [0.6, 1.5]
                taus = [round(x,2) for x in np.linspace(0.6, 1.5, 19)]
                best_tau, best_mean = 1.0, -1.0
                eval_counts = 0
                for tau in taus:
                    probs_actual: list[float] = []
                    # Iterate games that have both recon actual and candidates
                    for gid, sub in all_cand.groupby("game_id"):
                        row = recon[recon["game_id"].astype(str).str.zfill(10) == str(gid).zfill(10)]
                        if row.empty:
                            continue
                        actual = str(row.iloc[0].get("first_basket_actual_name") or "").strip().lower()
                        if not actual:
                            continue
                        sc = sub.copy()
                        sc["player_name_l"] = sc["player_name"].astype(str).str.lower()
                        denom = float(np.power(sc["raw_score"].values, 1.0/float(tau)).sum())
                        if denom <= 0:
                            continue
                        sc["prob_tau"] = np.power(sc["raw_score"].values, 1.0/float(tau)) / denom
                        # find actual row
                        m = sc[sc["player_name_l"].str.contains(actual, na=False)]
                        if not m.empty:
                            probs_actual.append(float(m.iloc[0]["prob_tau"]))
                    if probs_actual:
                        eval_counts += 1
                        avg = float(np.mean(probs_actual))
                        # Maximize avg; tie-breaker by closeness to 1.0
                        if (avg > best_mean) or (abs(avg - best_mean) <= 1e-9 and abs(tau-1.0) < abs(best_tau-1.0)):
                            best_mean = avg; best_tau = float(tau)
                if eval_counts >= 3:  # require at least 3 games evaluated
                    fb_temp = float(best_tau)
    except Exception:
        fb_temp = 1.0
    cal_path = paths.data_processed / "pbp_calibration.csv"
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([{ "date": str(anchor), "window_days": int(window_days), "thr_bias": thr_bias, "tip_logit_bias": tip_b, "fb_temp": fb_temp }])
    try:
        if cal_path.exists():
            prev = pd.read_csv(cal_path)
            out = pd.concat([prev, new_row], ignore_index=True)
        else:
            out = new_row
        out.to_csv(cal_path, index=False)
    except Exception as e:
        console.print(f"Failed to write calibration: {e}", style="red"); return
    console.print({"anchor": str(anchor), "window_days": int(window_days), "thr_bias": thr_bias, "tip_logit_bias": tip_b, "fb_temp": fb_temp, "output": str(cal_path)})


@cli.command()
@click.option("--input", "input_csv", required=True, type=click.Path(exists=True), help="CSV with columns: date,home_team,visitor_team")
def predict(input_csv: str):
    """Predict using trained models for upcoming games"""
    console.rule("Predict")
    inp = pd.read_csv(input_csv)
    inp["home_team"] = inp["home_team"].apply(normalize_team)
    inp["visitor_team"] = inp["visitor_team"].apply(normalize_team)
    res = _predict_from_matchups(inp)
    out = paths.root / "predictions.csv"
    res.to_csv(out, index=False)
    console.print(f"Saved predictions to {out}")


def _load_prediction_feature_history() -> pd.DataFrame:
    feats_csv = paths.data_processed / "features.csv"
    feats_parquet = paths.data_processed / "features.parquet"
    if feats_csv.exists():
        return pd.read_csv(feats_csv).sort_values("date")
    if feats_parquet.exists():
        return pd.read_parquet(feats_parquet).sort_values("date")
    raise FileNotFoundError("Features not found. Run build-features first.")


def _ensure_game_models_available(hist: pd.DataFrame | None = None) -> bool:
    required = [
        paths.models / "feature_columns.joblib",
        paths.models / "win_prob.joblib",
        paths.models / "spread_margin.joblib",
        paths.models / "totals.joblib",
        paths.models / "halves_models.joblib",
        paths.models / "quarters_models.joblib",
    ]
    missing = [path for path in required if not path.exists()]
    if not missing:
        return False

    if hist is None:
        hist = _load_prediction_feature_history()

    console.print(
        "Missing game model artifacts; bootstrapping baseline models from committed features.",
        style="yellow",
    )
    console.print({"missing_game_models": [path.name for path in missing]})

    from .train import train_models

    metrics = train_models(hist.copy())
    remaining = [path for path in required if not path.exists()]
    if remaining:
        missing_names = ", ".join(path.name for path in remaining)
        raise RuntimeError(f"Failed to bootstrap game models; still missing: {missing_names}")

    metric_summary = {
        key: (round(value, 4) if isinstance(value, float) else value)
        for key, value in metrics.items()
    }
    console.print({"bootstrapped_game_models": True, **metric_summary})
    return True


def _predict_from_matchups(inp: pd.DataFrame) -> pd.DataFrame:
    """Core prediction routine used by predict() and predict-date().

    Expects columns: date, home_team, visitor_team (teams normalized already).
    Returns a DataFrame with predictions for full game and periods.
    """
    # Load features history to bootstrap Elo and recent form
    # Try CSV first (ARM64 compatible), fallback to parquet
    try:
        hist = _load_prediction_feature_history()
    except FileNotFoundError:
        console.print("Features not found. Run build-features first.", style="red")
        raise SystemExit(1)

    try:
        _ensure_game_models_available(hist=hist)
    except Exception as e:
        console.print(f"Failed to bootstrap game models: {e}", style="red")
        raise SystemExit(1)

    elo = Elo()
    # Roll through history to update Elo
    for _, row in track(hist.iterrows(), total=len(hist), description="Updating Elo"):
        if pd.notna(row.get("home_pts")) and pd.notna(row.get("visitor_pts")):
            try:
                elo.update_game(row["home_team"], row["visitor_team"], int(row["home_pts"]), int(row["visitor_pts"]))
            except Exception:
                pass

    # Compute schedule-aware rest for the matchups using history
    rest_df = compute_rest_for_matchups(inp, hist)
    # Build features from Elo + rest
    feat_rows = []
    for _, r in rest_df.iterrows():
        feat_rows.append({
            "elo_diff": elo.get(r["home_team"]) - elo.get(r["visitor_team"]),
            "home_rest_days": r.get("home_rest_days", 2) if pd.notna(r.get("home_rest_days")) else 2,
            "visitor_rest_days": r.get("visitor_rest_days", 2) if pd.notna(r.get("visitor_rest_days")) else 2,
            "home_b2b": r.get("home_b2b", 0) if pd.notna(r.get("home_b2b")) else 0,
            "visitor_b2b": r.get("visitor_b2b", 0) if pd.notna(r.get("visitor_b2b")) else 0,
            "home_team": r["home_team"],
            "visitor_team": r["visitor_team"],
            "date": r.get("date"),
        })
    # Training feature columns
    try:
        feat_cols = joblib.load(paths.models / "feature_columns.joblib")
    except FileNotFoundError:
        feat_cols = ["elo_diff", "home_rest_days", "visitor_rest_days", "home_b2b", "visitor_b2b"]

    # Use enhanced feature building for 45 features
    import numpy as np
    from .features_enhanced import build_features_enhanced
    
    console.print("[Building enhanced features (45 features)...]", style="cyan")
    
    # Combine historical games with new matchups for feature generation
    upcoming = pd.DataFrame(feat_rows)
    upcoming["home_pts"] = np.nan  # No scores yet (future games)
    upcoming["visitor_pts"] = np.nan
    
    # Ensure date columns have same type
    hist["date"] = pd.to_datetime(hist["date"])
    upcoming["date"] = pd.to_datetime(upcoming["date"])
    
    # Append upcoming to history
    combined = pd.concat([hist, upcoming], ignore_index=True).sort_values("date")
    
    # Build all 45 enhanced features
    features_df = build_features_enhanced(
        combined, 
        include_advanced_stats=True,
        include_injuries=True
    )
    
    # Extract only the upcoming games (last N rows)
    enriched = features_df.tail(len(feat_rows))
    
    # Get feature matrix
    X = enriched[feat_cols].fillna(0)

    # Load models - Use NPU-accelerated predictions for ALL models (game + periods)
    try:
        # Import NPU game predictor with period support
        from .games_npu import NPUGamePredictor
        
        # Create NPU predictor (win, spread, total, quarters, halves with NPU acceleration)
        console.print("[NPU] Using NPU-accelerated predictions (ONNX + QNN)", style="green")
        npu_predictor = NPUGamePredictor()
        
        # Convert features to numpy array for NPU inference
        X_np = X.values.astype(np.float32)
        
        # Run NPU predictions for all games with period breakdowns
        # Create result dataframe with original matchup info
        res = pd.DataFrame(feat_rows)[["date", "home_team", "visitor_team"]].copy()
        npu_results = npu_predictor.predict_batch(X_np, include_periods=True)
        
        # Extract main game predictions
        res["home_win_prob"] = [r["win_prob"] for r in npu_results]
        res["pred_margin"] = [r["spread_margin"] for r in npu_results]
        res["pred_total"] = [r["totals"] for r in npu_results]
        
        # Include calibration columns if available
        if npu_results and "win_prob_raw" in npu_results[0]:
            res["home_win_prob_raw"] = [r.get("win_prob_raw", r["win_prob"]) for r in npu_results]
            res["home_win_prob_from_spread"] = [r.get("win_prob_from_spread", r["win_prob"]) for r in npu_results]
        
        # Extract period predictions (halves and quarters)
        for i, r in enumerate(npu_results):
            if "halves" in r:
                for half in ("h1", "h2"):
                    if half in r["halves"]:
                        res.loc[i, f"halves_{half}_win"] = r["halves"][half].get("win", 0.5)
                        res.loc[i, f"halves_{half}_margin"] = r["halves"][half].get("margin", 0.0)
                        res.loc[i, f"halves_{half}_total"] = r["halves"][half].get("total", 100.0)
            
            if "quarters" in r:
                for q in ("q1", "q2", "q3", "q4"):
                    if q in r["quarters"]:
                        res.loc[i, f"quarters_{q}_win"] = r["quarters"][q].get("win", 0.5)
                        res.loc[i, f"quarters_{q}_margin"] = r["quarters"][q].get("margin", 0.0)
                        res.loc[i, f"quarters_{q}_total"] = r["quarters"][q].get("total", 50.0)

        # Apply optional period calibration if all required fields present
        try:
            from .period_calibration import calibrate_period_predictions, CalibrationConfig
            cfg = CalibrationConfig()
            if {"home_team", "visitor_team", "pred_total", "pred_margin"}.issubset(set(res.columns)) or {"home_team", "visitor_team", "pred_total", "spread_margin"}.issubset(set(res.columns)):
                # Ensure unified column names expected by calibrator
                if "pred_total" in res.columns and "totals" not in res.columns:
                    res.rename(columns={"pred_total": "totals"}, inplace=True)
                if "pred_margin" in res.columns and "spread_margin" not in res.columns:
                    res.rename(columns={"pred_margin": "spread_margin"}, inplace=True)
                res = calibrate_period_predictions(res, cfg)
        except Exception:
            pass
        
    except (FileNotFoundError, ImportError) as e:
        console.print(f"⚠️  NPU predictor not available: {e}", style="yellow")
        console.print("Falling back to sklearn models (requires sklearn installed)...", style="yellow")
        # Fallback to sklearn models
        win_model = joblib.load(paths.models / "win_prob.joblib")
        spread_model = joblib.load(paths.models / "spread_margin.joblib")
        total_model = joblib.load(paths.models / "totals.joblib")
        
        res = pd.DataFrame(feat_rows)
        res["home_win_prob"] = win_model.predict_proba(X)[:, 1]
        res["pred_margin"] = spread_model.predict(X)
        res["pred_total"] = total_model.predict(X)
        
        # Load period models (halves/quarters) for fallback
        try:
            halves = joblib.load(paths.models / "halves_models.joblib")
            quarters = joblib.load(paths.models / "quarters_models.joblib")
        except Exception:
            console.print("⚠️  Period models not available in fallback", style="yellow")
            halves = {}
            quarters = {}
        
        for half in ("h1", "h2"):
            if half in halves:
                res[f"halves_{half}_win"] = halves[half]["win"].predict_proba(X)[:, 1]
                res[f"halves_{half}_margin"] = halves[half]["margin"].predict(X)
                res[f"halves_{half}_total"] = halves[half]["total"].predict(X)
        for q in ("q1", "q2", "q3", "q4"):
            if q in quarters:
                res[f"quarters_{q}_win"] = quarters[q]["win"].predict_proba(X)[:, 1]
                res[f"quarters_{q}_margin"] = quarters[q]["margin"].predict(X)
                res[f"quarters_{q}_total"] = quarters[q]["total"].predict(X)
    return res

@cli.command("daily-update")
@click.option("--date", "date_str", type=str, required=False, help="Target date YYYY-MM-DD; defaults to the current US local slate date")
@click.option("--season", type=str, default="2025-26", help="Roster season string (e.g., 2025-26)")
@click.option("--odds-api-key", envvar="ODDS_API_KEY", type=str, required=False, help="OddsAPI key for fetching current odds")
@click.option("--git-push/--no-git-push", default=False, show_default=True, help="Commit and push changes to git at end")
@click.option("--props-books", type=str, default=None, help="Comma-separated bookmaker keys to include in edges")
@click.option("--min-prop-edge", type=float, default=0.03, show_default=True)
@click.option("--use-npu/--no-npu", default=True, show_default=True, help="Use NPU acceleration for training and predictions")
@click.option("--reconcile-days", type=int, default=7, show_default=True, help="Days back to reconcile actuals")
@click.option("--retrain-games/--no-retrain-games", default=False, show_default=True, help="Retrain game models (enhanced models recommended). Default is no retrain to preserve calibrated enhanced models.")
@click.option("--retrain-props/--no-retrain-props", default=False, show_default=True, help="Retrain prop models. Default is no retrain; daily runs use existing ONNX models.")
def daily_update_cmd(date_str: str | None, season: str, odds_api_key: str | None, git_push: bool, props_books: str | None, min_prop_edge: float, use_npu: bool, reconcile_days: int, retrain_games: bool, retrain_props: bool):
    """Enhanced end-to-end daily updater with NPU acceleration, actuals reconciliation, and comprehensive odds fetching."""
    console.rule("Enhanced Daily Update")
    import datetime as _dt
    import sys
    import subprocess
    try:
        target_date = _resolve_target_date(date_str)
    except Exception:
        console.print("Invalid --date (YYYY-MM-DD)", style="red")
        return
    
    console.print(f"[NPU] Running enhanced daily update for {target_date}")
    console.print(f"[INFO] NPU Acceleration: {'Enabled' if use_npu else 'Disabled'}")
    console.print(f"🔄 Reconciliation window: {reconcile_days} days")

    # 1) Schedule (once per season; idempotent to run daily)
    try:
        console.print("📅 Updating schedule...")
        df_sched = fetch_schedule_2025_26()
        console.print({"schedule_rows": int(len(df_sched))})
    except Exception as e:
        console.print(f"Schedule update failed: {e}", style="yellow")

    # 2) Rosters
    try:
        console.print("👥 Updating rosters...")
        df_rosters = fetch_rosters(season=season)
        console.print({"roster_rows": int(len(df_rosters))})
    except Exception as e:
        console.print(f"Rosters update failed: {e}", style="yellow")

    # 3) Player logs (season(s) up to current)
    try:
        console.print("[INFO] Updating player logs...")
        # Infer seasons around target date; for now, update current season only
        fetch_player_logs([season])
        console.print("Player logs refreshed")
    except Exception as e:
        console.print(f"Player logs update failed: {e}", style="yellow")

    # 4) Reconcile historic game actuals (last N days) → writes recon_games_<date>.csv
    try:
        console.print(f"[SEARCH] Reconciling game actuals for last {reconcile_days} days...")
        for days_back in range(1, reconcile_days + 1):
            past_date = target_date - _dt.timedelta(days=days_back)
            try:
                # Invoke reconcile-date via module to avoid Click context issues
                log = paths.data_processed / f".logs/reconcile_games_{past_date}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable or "python", "-m", "nba_betting.cli", "reconcile-date", "--date", str(past_date)]
                with open(log, "w", encoding="utf-8", errors="ignore") as fh:
                    rc = subprocess.run(cmd, cwd=paths.root, stdout=fh, stderr=subprocess.STDOUT, check=False)
                console.print(f"  [OK] Game actuals for {past_date} (rc={rc.returncode})")
            except Exception as e:
                console.print(f"  ⚠️  Game actuals for {past_date}: {e}")
    except Exception as e:
        console.print(f"Game actuals reconciliation failed: {e}", style="yellow")

    # 5) Reconcile historic prop actuals (last N days) and write recon_props_<date>.csv
    try:
        console.print(f"[ACTION] Reconciling prop actuals for last {reconcile_days} days...")
        for days_back in range(1, reconcile_days + 1):
            past_date = target_date - _dt.timedelta(days=days_back)
            try:
                # Call the function directly rather than the CLI command
                from .props_actuals import fetch_prop_actuals_via_nbastatr
                try:
                    df_actuals = fetch_prop_actuals_via_nbastatr(date=str(past_date))
                    if df_actuals is not None and not df_actuals.empty:
                        from .props_actuals import upsert_props_actuals
                        upsert_props_actuals(df_actuals)
                        # Also write a recon CSV for transparency
                        try:
                            outp = paths.data_processed / f"recon_props_{past_date}.csv"
                            outp.parent.mkdir(parents=True, exist_ok=True)
                            df_actuals.to_csv(outp, index=False)
                        except Exception:
                            pass
                        console.print(f"  [OK] Prop actuals for {past_date}")
                except Exception:
                    # Fallback to nba_api
                    from .props_actuals import fetch_prop_actuals_via_nbaapi
                    df_actuals = fetch_prop_actuals_via_nbaapi(str(past_date))
                    if df_actuals is not None and not df_actuals.empty:
                        from .props_actuals import upsert_props_actuals
                        upsert_props_actuals(df_actuals)
                        try:
                            outp = paths.data_processed / f"recon_props_{past_date}.csv"
                            outp.parent.mkdir(parents=True, exist_ok=True)
                            df_actuals.to_csv(outp, index=False)
                        except Exception:
                            pass
                        console.print(f"  [OK] Prop actuals for {past_date}")
            except Exception as e:
                console.print(f"  ⚠️  Prop actuals for {past_date}: {e}")
    except Exception as e:
        console.print(f"Props actuals reconciliation failed: {e}", style="yellow")

    # 6) Optional: rebuild base features and retrain game models
    if retrain_games:
        try:
            console.print("🏗️  Rebuilding game features (for retrain)...")
            feats_raw = paths.data_raw / "games_nba_api.parquet"
            if feats_raw.exists():
                try:
                    df = pd.read_parquet(feats_raw)
                except Exception as e:
                    raise RuntimeError(f"Cannot read {feats_raw.name}; install pyarrow/fastparquet ({e})")
                from .features import build_features
                feats = build_features(df)
                out = paths.data_processed / "features.parquet"
                out.parent.mkdir(parents=True, exist_ok=True)
                feats.to_parquet(out, index=False)
                console.print(f"Features built and saved to {out}")
                if use_npu:
                    console.print("[NPU] Retraining game models with latest data...")
                    from .games_npu import train_game_models_npu
                    train_game_models_npu(retrain=True)
                else:
                    console.print("🖥️  Retraining game models (CPU)...")
                    feats_path = paths.data_processed / "features.parquet"
                    df = pd.read_parquet(feats_path)
                    from .train import train_models
                    _ = train_models(df)
                    console.print("Game models retrained (CPU)")
            else:
                console.print("Raw games not found; skipping full-game retrain.", style="yellow")
        except Exception as e:
            console.print(f"Game model retrain failed: {e}", style="yellow")
    else:
        console.print("⏭️  Skipping game model retrain (use --retrain-games to enable)")

    # 7) Optional: retrain props models
    if retrain_props:
        try:
            console.print("[ACTION] Rebuilding props features...")
            from .props_features import build_props_features
            df = build_props_features()
            console.print(f"Props features built: {len(df)} rows")
            if use_npu:
                console.print("[NPU] Training props models with NPU acceleration...")
                from .props_npu import train_props_models_npu
                train_props_models_npu(alpha=1.0)
            else:
                console.print("🖥️  Training props models (CPU)...")
                from .props_train import train_props_models
                train_props_models(alpha=1.0)
                console.print("Props models trained (CPU)")
        except Exception as e:
            console.print(f"Props model retrain failed: {e}", style="yellow")
    else:
        console.print("⏭️  Skipping props model retrain (use --retrain-props to enable)")

    # 8) Fetch current game odds from OddsAPI and write to CSV
    try:
        if odds_api_key:
            console.print("💰 Fetching current game odds...")
            cfg = OddsApiConfig(api_key=odds_api_key)
            go = fetch_game_odds_current(cfg, pd.to_datetime(target_date))
            if go is not None and not go.empty:
                out_csv = paths.data_raw / f"odds_nba_current_{target_date}.csv"
                go.to_csv(out_csv, index=False)
                console.print({"game_odds_rows": int(len(go)), "output": str(out_csv)})
                
                # Also write to processed for frontend
                proc_csv = paths.data_processed / f"game_odds_{target_date}.csv"
                go.to_csv(proc_csv, index=False)
        else:
            console.print("No OddsAPI key provided; skipping game odds fetch", style="yellow")
    except Exception as e:
        console.print(f"Game odds fetch failed: {e}", style="yellow")

    # 9) Predict today's slate (games) with enhanced, injury-aware features and write predictions_<date>.csv
    try:
        console.print("🎲 Generating game predictions (enhanced + injuries)...")
        # Invoke subcommand to avoid Click context issues (and keep behavior consistent with scripts/daily_update.ps1).
        cmd = [
            sys.executable or "python",
            "-m",
            "nba_betting.cli",
            "predict-date",
            "--date",
            str(target_date),
        ]
        pr = subprocess.run(cmd, cwd=paths.root, check=False)
        if pr.returncode != 0:
            raise RuntimeError(f"predict-date failed (rc={pr.returncode})")

        # Ensure calibrated/blended win probability column is present for evaluation/UI.
        # This writes/updates home_win_prob_cal inside data/processed/predictions_<date>.csv.
        try:
            import sys
            import subprocess
            end_for_training = (target_date - _dt.timedelta(days=1)).strftime("%Y-%m-%d")
            blend_tool = paths.root / "tools" / "games_blend.py"
            if blend_tool.exists():
                cmd = [
                    sys.executable or "python",
                    str(blend_tool),
                    "--train-days",
                    "30",
                    "--end",
                    end_for_training,
                    "--apply-date",
                    target_date.strftime("%Y-%m-%d"),
                ]
                subprocess.run(cmd, cwd=paths.root, check=False)
        except Exception:
            pass
    except Exception as e:
        console.print(f"Game predictions failed: {e}", style="yellow")

    # 10) Fetch current prop odds from OddsAPI and write to CSV
    try:
        if odds_api_key:
            console.print("[ACTION] Fetching current prop odds...")
            cfg = OddsApiConfig(api_key=odds_api_key)
            # Fetch player props for today
            from .odds_api import fetch_player_props_current
            props_odds = fetch_player_props_current(cfg, date=target_date, markets=None, verbose=True)
            if props_odds is not None and not props_odds.empty:
                out_csv = paths.data_raw / f"odds_nba_player_props_{target_date}.csv"
                props_odds.to_csv(out_csv, index=False)
                console.print({"prop_odds_rows": int(len(props_odds)), "output": str(out_csv)})
        else:
            console.print("No OddsAPI key provided; skipping prop odds fetch", style="yellow")
    except Exception as e:
        console.print(f"Prop odds fetch failed: {e}", style="yellow")

    # 11) Props predictions and edges for target date
    try:
        console.print("[ACTION] Generating prop predictions and edges...")
        # Predict props using the canonical CLI subcommand.
        # This ensures SmartSim overrides are applied (minutes/rotations-aware) regardless of NPU mode.
        cmd = [
            sys.executable or "python",
            "-m",
            "nba_betting.cli",
            "predict-props",
            "--date",
            str(target_date),
            "--slate-only",
            "--calibrate",
            "--calib-window",
            "7",
            "--calibrate-player",
            "--player-calib-window",
            "30",
            "--player-min-pairs",
            "6",
            "--player-shrink-k",
            "8",
            "--use-pure-onnx",
            "--use-smart-sim",
            "--smart-sim-n-sims",
            "150",
            "--smart-sim-pbp",
            "--smart-sim-roster-mode",
            "pregame",
            "--smart-sim-out-prefix",
            "smart_sim",
        ]
        pr = subprocess.run(cmd, cwd=paths.root, check=False)
        if pr.returncode != 0:
            raise RuntimeError(f"predict-props failed (rc={pr.returncode})")
        
        # Fetch current props odds and compute edges
        if odds_api_key:
            try:
                sigma = calibrate_sigma_for_date(str(target_date), window_days=30, min_rows=200, defaults=SigmaConfig())
                edges = compute_props_edges(
                    date=str(target_date),
                    sigma=sigma,
                    use_saved=False,
                    mode="current",
                    api_key=odds_api_key,
                    source="auto",
                    predictions_path=None,
                    from_file_only=False
                )
                if edges is not None and not edges.empty:
                    # Apply edge and EV filters
                    edges = edges[(edges["edge"] >= min_prop_edge) & (edges["ev"] >= 0.0)].copy()
                    edges.sort_values(["stat", "edge"], ascending=[True, False], inplace=True)
                    if len(edges) > 200:
                        edges = edges.groupby("stat", group_keys=False).head(max(1, 200 // max(1, edges["stat"].nunique())))
                    out = paths.data_processed / f"props_edges_{target_date}.csv"
                    edges.to_csv(out, index=False)
                    console.print(f"[OK] Props edges saved to {out}")
            except Exception as e:
                console.print(f"Props edges computation failed: {e}", style="yellow")
        else:
            console.print("No OddsAPI key provided; skipping props edges", style="yellow")
    except Exception as e:
        console.print(f"Props edges failed: {e}", style="yellow")

    # 12) Generate frontend-ready recommendation files
    try:
        console.print("📱 Generating frontend recommendations...")
        # Generate game recommendations - simplified version to avoid click context
        try:
            pred = paths.data_processed / f"predictions_{target_date}.csv"
            if pred.exists():
                df = pd.read_csv(pred)
                d = target_date
                recs = []
                def _num(x):
                    try:
                        return float(x)
                    except Exception:
                        return None
                for _, r in df.iterrows():
                    try:
                        home = r.get("home_team"); away = r.get("visitor_team")
                        # ATS
                        pm = _num(r.get("pred_margin")); hs = _num(r.get("home_spread"))
                        if pm is not None and hs is not None:
                            edge_spread = pm - (-hs)
                            if abs(edge_spread) >= 1.0:
                                recs.append({"market":"ATS","side": home if edge_spread>0 else away, "home": home, "away": away, "edge": float(edge_spread), "date": str(d)})
                        # TOTAL
                        pt = _num(r.get("pred_total")); tot = _num(r.get("total"))
                        if pt is not None and tot is not None:
                            edge_total = pt - tot
                            if abs(edge_total) >= 1.5:
                                recs.append({"market":"TOTAL","side": ("Over" if edge_total>0 else "Under"), "home": home, "away": away, "edge": float(edge_total), "date": str(d)})
                    except Exception:
                        continue
                out = paths.data_processed / f"recommendations_{target_date}.csv"
                pd.DataFrame(recs).to_csv(out, index=False)
                console.print(f"[OK] Game recommendations saved to {out}")
        except Exception as e:
            console.print(f"Game recommendations failed: {e}", style="yellow")
        
        # Generate prop recommendations - simplified version
        try:
            # Produce frontend-ready prop cards (CSV) from edges/predictions.
            cmd = [
                sys.executable or "python",
                "-m",
                "nba_betting.cli",
                "export-props-recommendations",
                "--date",
                str(target_date),
            ]
            pr = subprocess.run(cmd, cwd=paths.root, check=False)
            if pr.returncode != 0:
                raise RuntimeError(f"export-props-recommendations failed (rc={pr.returncode})")
            outp = paths.data_processed / f"props_recommendations_{target_date}.csv"
            console.print(f"[OK] Prop recommendations saved to {outp}")
        except Exception as e:
            console.print(f"Prop recommendations failed: {e}", style="yellow")

        # Generate authoritative best-edges snapshots for durable tracking
        try:
            outg, outpr = _export_best_edges_snapshot(
                date_str=str(target_date),
                max_games=10,
                max_props=25,
                overwrite=True,
            )
            console.print(f"[OK] Best edges snapshots saved: {outg.name}, {outpr.name}")
        except Exception as e:
            console.print(f"Best edges snapshot export failed: {e}", style="yellow")
        
        console.print("[OK] Frontend recommendation files generated")
    except Exception as e:
        console.print(f"Frontend recommendations failed: {e}", style="yellow")

    # 13) Git commit and push changes (optional)
    if git_push:
        try:
            console.print("📤 Committing and pushing to git...")
            subprocess.run(["git", "add", "-A"], check=False, cwd=paths.root)
            msg = f"daily update {target_date} {'(NPU)' if use_npu else '(CPU)'}"
            result = subprocess.run(["git", "commit", "-m", msg], check=False, cwd=paths.root, capture_output=True, text=True)
            if result.returncode == 0:
                push_result = subprocess.run(["git", "push"], check=False, cwd=paths.root, capture_output=True, text=True)
                if push_result.returncode == 0:
                    console.print("[OK] Git push complete")
                else:
                    console.print(f"Git push failed: {push_result.stderr}", style="yellow")
            else:
                console.print("No changes to commit", style="blue")
        except Exception as e:
            console.print(f"Git operations failed: {e}", style="yellow")
    
    # 14) Summary report
    console.print("\n🎉 Daily update complete!")
    console.print(f"📅 Date: {target_date}")
    console.print(f"[NPU] NPU: {'Enabled' if use_npu else 'Disabled'}")
    console.print(f"💰 Odds API: {'Used' if odds_api_key else 'Skipped'}")
    console.print(f"📤 Git Push: {'Yes' if git_push else 'No'}")
    console.print("\n📁 Check these files for today's data:")
    console.print(f"  - predictions_{target_date}.csv")
    console.print(f"  - props_predictions_{target_date}.csv")
    console.print(f"  - props_edges_{target_date}.csv") 
    console.print(f"  - recommendations_{target_date}.csv")
    console.print(f"  - props_recommendations_{target_date}.csv")


@cli.command("sync-frontend")
@click.option("--date", "date_str", type=str, required=False, help="Target date YYYY-MM-DD; defaults to the current US local slate date")
@click.option("--cleanup-days", type=int, default=30, show_default=True, help="Clean up files older than N days")
def sync_frontend_cmd(date_str: str | None, cleanup_days: int):
    """Validate and sync all data files for frontend consumption."""
    console.rule("Frontend Data Sync")
    try:
        target_date = _resolve_target_date(date_str)
    except Exception:
        console.print("Invalid --date (YYYY-MM-DD)", style="red")
        return
    
    try:
        from .frontend_sync import validate_and_sync_frontend_data, cleanup_old_files, get_frontend_data_status
        
        console.print(f"🔄 Syncing frontend data for {target_date}...")
        results = validate_and_sync_frontend_data(str(target_date))
        
        console.print("\n[INFO] Validation Results:")
        for data_type, validation in results["validation_results"].items():
            console.print(f"  {data_type}: {validation['rows']} rows")
        
        console.print(f"\n📁 Files Created: {len(results['files_created'])}")
        for file_path in results["files_created"]:
            console.print(f"  [OK] {file_path}")
        
        if results["errors"]:
            console.print(f"\n⚠️  Errors: {len(results['errors'])}")
            for error in results["errors"]:
                console.print(f"  [ERROR] {error}")
        
        # Cleanup old files
        if cleanup_days > 0:
            console.print(f"\n🧹 Cleaning up files older than {cleanup_days} days...")
            cleanup_stats = cleanup_old_files(keep_days=cleanup_days)
            console.print(f"  Removed {cleanup_stats['files_removed']} files")
            console.print(f"  Freed {cleanup_stats['bytes_freed']:,} bytes")
        
        # Show frontend status
        console.print(f"\n📱 Frontend Data Status:")
        status = get_frontend_data_status()
        for file_name, file_info in status["latest_files"].items():
            if file_info["exists"]:
                console.print(f"  [OK] {file_name}")
            else:
                console.print(f"  [ERROR] {file_name} (missing)")
        
        console.print(f"\n🎉 Frontend sync complete!")
        
    except Exception as e:
        console.print(f"Frontend sync failed: {e}", style="red")


@cli.command("frontend-status")
def frontend_status_cmd():
    """Check status of frontend data files."""
    console.rule("Frontend Data Status")
    
    try:
        from .frontend_sync import get_frontend_data_status
        
        status = get_frontend_data_status()
        
        console.print("📁 Latest Files:")
        for file_name, file_info in status["latest_files"].items():
            if file_info["exists"]:
                size_mb = file_info["size_bytes"] / (1024 * 1024)
                console.print(f"  [OK] {file_name} ({size_mb:.2f} MB)")
                console.print(f"     Last modified: {file_info['last_modified']}")
            else:
                console.print(f"  [ERROR] {file_name} (missing)")
        
        console.print(f"\n📅 Recent Files (last 7 days):")
        for date_str, files in status["dated_files"].items():
            available_count = sum(1 for exists in files.values() if exists)
            total_count = len(files)
            console.print(f"  {date_str}: {available_count}/{total_count} files available")
        
        if status["missing_files"]:
            console.print(f"\n⚠️  Missing Files:")
            for file_name in status["missing_files"]:
                console.print(f"  [ERROR] {file_name}")
        
    except Exception as e:
        console.print(f"Status check failed: {e}", style="red")


@cli.command("predict-date")
@click.option("--date", "date_str", type=str, required=False, help="Slate date YYYY-MM-DD; defaults to the current US local slate date")
@click.option("--merge-odds", "merge_odds_csv", type=click.Path(exists=True), required=False, help="Optional CSV of odds to merge. Columns supported: date,home_team,visitor_team,home_ml,away_ml,home_spread,total as well as period markets: h1_spread,h1_total,h2_spread,h2_total,q1_spread,q1_total,q2_spread,q2_total,q3_spread,q3_total,q4_spread,q4_total")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=False, help="Output CSV path; default predictions_YYYY-MM-DD.csv in repo root")
def predict_date_cmd(date_str: str | None, merge_odds_csv: str | None, out_path: str | None):
    """Predict today's (or specified date's) slate using ScoreboardV2, with optional odds merge."""
    console.rule("Predict (Date)")
    # Harden headers to reduce blocks
    try:
        nba_http.STATS_HEADERS.update({
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Origin': 'https://www.nba.com',
            'Referer': 'https://www.nba.com/stats/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
            'Connection': 'keep-alive',
        })
    except Exception:
        pass
    try:
        date_str = _resolve_target_date(date_str).isoformat()
    except Exception:
        console.print("Invalid --date (YYYY-MM-DD)", style="red")
        return

    # Helper: build slate from historical features if NBA API fails
    def _build_slate_from_history(date_str_local: str) -> pd.DataFrame | None:
        feats_path = paths.data_processed / "features.parquet"
        if not feats_path.exists():
            return None
        try:
            dfh = pd.read_parquet(feats_path)
        except ImportError:
            # pyarrow not available (e.g., on ARM64 Windows)
            import sys
            print("Warning: pyarrow not available, skipping parquet fallback", file=sys.stderr)
            return None
        dfh = dfh.copy()
        dfh["date"] = pd.to_datetime(dfh["date"]).dt.date
        try:
            target_d = pd.to_datetime(date_str_local).date()
        except Exception:
            return None
        part = dfh[dfh["date"] == target_d][["date","home_team","visitor_team"]].dropna()
        part["home_team"] = part["home_team"].apply(normalize_team)
        part["visitor_team"] = part["visitor_team"].apply(normalize_team)
        part = part.drop_duplicates()
        return part if not part.empty else None

    slate = None
    # Fallback: build slate from live schedule data, then local schedule artifacts, when API/history fail
    def _build_slate_from_schedule(date_str_local: str) -> pd.DataFrame | None:
        try:
            day = _load_schedule_day(date_str_local)
            if day.empty:
                return None
            abbr_to_full = {str(t["abbreviation"] or "").upper(): str(t["full_name"] or "") for t in static_teams.get_teams()}
            # Build full team names from City + Name to feed normalize_team
            def full_name(city, name):
                city_s = str(city or "").strip()
                name_s = str(name or "").strip()
                return f"{city_s} {name_s}".strip()
            rows = []
            for _, g in day.iterrows():
                home_full = full_name(g.get("home_city"), g.get("home_name")) or abbr_to_full.get(str(g.get("home_tricode") or "").strip().upper(), str(g.get("home_tricode") or "").strip())
                away_full = full_name(g.get("away_city"), g.get("away_name")) or abbr_to_full.get(str(g.get("away_tricode") or "").strip().upper(), str(g.get("away_tricode") or "").strip())
                home = normalize_team(home_full)
                away = normalize_team(away_full)
                rows.append({
                    "date": pd.to_datetime(date_str_local).date(),
                    "home_team": home,
                    "visitor_team": away,
                })
            df = pd.DataFrame(rows)
            return df if not df.empty else None
        except Exception:
            return None

    def _build_slate_from_game_odds(date_str_local: str) -> pd.DataFrame | None:
        try:
            odds_path = paths.data_processed / f"game_odds_{date_str_local}.csv"
            if not odds_path.exists():
                return None
            df = pd.read_csv(odds_path)
            if df is None or df.empty:
                return None
            home_col = "home_team" if "home_team" in df.columns else None
            away_col = "visitor_team" if "visitor_team" in df.columns else ("away_team" if "away_team" in df.columns else None)
            if not home_col or not away_col:
                return None
            out = df[[home_col, away_col]].copy()
            out = out.rename(columns={home_col: "home_team", away_col: "visitor_team"})
            out["home_team"] = out["home_team"].astype(str).map(normalize_team)
            out["visitor_team"] = out["visitor_team"].astype(str).map(normalize_team)
            out["date"] = pd.to_datetime(date_str_local).date()
            out = out[["date", "home_team", "visitor_team"]].dropna().drop_duplicates()
            return out if not out.empty else None
        except Exception:
            return None
    try:
        # Fetch slate from ScoreboardV2
        sb = scoreboardv2.ScoreboardV2(game_date=date_str, day_offset=0, timeout=30)
        nd = sb.get_normalized_dict()
        gh = pd.DataFrame(nd.get("GameHeader", []))
        ls = pd.DataFrame(nd.get("LineScore", []))
        if gh.empty or ls.empty:
            raise RuntimeError("Scoreboard returned empty tables")
        gh_cols = {c.upper(): c for c in gh.columns}
        ls_cols = {c.upper(): c for c in ls.columns}
        required = ["GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID", "GAME_DATE_EST"]
        if "GAME_DATE_EST" not in gh_cols and "GAME_DATE" in gh_cols:
            gh_cols["GAME_DATE_EST"] = gh_cols["GAME_DATE"]
        if not all(k in gh_cols for k in required) or not {"GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION"}.issubset(ls_cols.keys()):
            raise RuntimeError("Scoreboard missing required columns")

        # Map TEAM_ID -> ABBR for this date
        team_abbr_map = {}
        for _, r in ls.iterrows():
            try:
                team_abbr_map[int(r[ls_cols["TEAM_ID"]])] = str(r[ls_cols["TEAM_ABBREVIATION"]])
            except Exception:
                continue
        # Build matchups
        team_list = static_teams.get_teams(); abbr_to_full = {t['abbreviation']: t['full_name'] for t in team_list}
        rows = []
        for _, g in gh.iterrows():
            try:
                home_id = int(g[gh_cols["HOME_TEAM_ID"]]); vis_id = int(g[gh_cols["VISITOR_TEAM_ID"]])
                habbr = team_abbr_map.get(home_id); vabbr = team_abbr_map.get(vis_id)
                if not habbr or not vabbr:
                    continue
                home = normalize_team(abbr_to_full.get(habbr, habbr))
                away = normalize_team(abbr_to_full.get(vabbr, vabbr))
                rows.append({
                    "date": pd.to_datetime(g[gh_cols["GAME_DATE_EST"]]).date(),
                    "home_team": home,
                    "visitor_team": away,
                })
            except Exception:
                continue
        if rows:
            slate = pd.DataFrame(rows)
        else:
            slate = None
    except Exception as e:
        console.print(f"Scoreboard fetch failed ({e}); trying fallbacks for {date_str}.", style="yellow")
        slate = _build_slate_from_history(date_str)
        if slate is None or slate.empty:
            slate = _build_slate_from_schedule(date_str)
        if slate is None or slate.empty:
            slate = _build_slate_from_game_odds(date_str)

    if slate is None or slate.empty:
        console.print(f"No games found on {date_str} (API down and no history/schedule/game-odds fallback).", style="yellow"); return

    # Predict
    res = _predict_from_matchups(slate)

    # Auto-odds: If a CSV path was provided, merge it; otherwise try OddsAPI then Bovada, save standardized game_odds_{date}.csv, and merge
    def _merge_odds_df(odds_df: pd.DataFrame):
        nonlocal res
        if odds_df is None or odds_df.empty:
            return
        o = odds_df.copy()
        # Normalize 'date' to python date on both frames to avoid dtype mismatches
        try:
            if 'date' in o.columns:
                o['date'] = pd.to_datetime(o['date'], errors='coerce').dt.date
        except Exception:
            pass
        try:
            if 'date' in res.columns:
                res['date'] = pd.to_datetime(res['date'], errors='coerce').dt.date
        except Exception:
            pass
        # Normalize names
        if 'home_team' in o.columns:
            o['home_team'] = o['home_team'].apply(normalize_team)
        if 'visitor_team' in o.columns:
            o['visitor_team'] = o['visitor_team'].apply(normalize_team)
        res = res.merge(o, on=['date','home_team','visitor_team'], how='left', suffixes=('', '_odds'))
        # Compute implied probs and edges
        def implied_prob_american(odds):
            try:
                o = float(odds)
            except Exception:
                return pd.NA
            if pd.isna(o):
                return pd.NA
            if o < 0:
                return (-o) / ((-o) + 100.0)
            return 100.0 / (o + 100.0)
        if 'home_ml' in res.columns:
            res['home_implied_prob'] = res['home_ml'].apply(implied_prob_american)
            res['edge_win'] = res['home_win_prob'] - res['home_implied_prob']
        if 'home_spread' in res.columns:
            res['market_home_margin'] = -res['home_spread']
            # Use whichever prediction column exists: prefer 'pred_margin', else 'spread_margin'
            pred_col = 'pred_margin' if 'pred_margin' in res.columns else ('spread_margin' if 'spread_margin' in res.columns else None)
            if pred_col is not None:
                res['edge_spread'] = res[pred_col] - res['market_home_margin']
        if 'total' in res.columns:
            # Use available total prediction column
            total_pred_col = 'pred_total' if 'pred_total' in res.columns else ('totals' if 'totals' in res.columns else None)
            if total_pred_col is not None:
                res['edge_total'] = res[total_pred_col] - res['total']
        # Period edges if columns present
        for half in ("h1","h2"):
            sp_col = f"{half}_spread"; tot_col = f"{half}_total"
            if sp_col in res.columns:
                # Prefer calibrated pred margin if available, else use generic naming if present
                half_pred_col = f"{half}_pred_margin" if f"{half}_pred_margin" in res.columns else None
                if half_pred_col is not None:
                    res[f"edge_{half}_spread"] = res[half_pred_col] - (-res[sp_col])
            if tot_col in res.columns and f"{half}_pred_total" in res.columns:
                res[f"edge_{half}_total"] = res[f"{half}_pred_total"] - res[tot_col]
        for q in ("q1","q2","q3","q4"):
            sp_col = f"{q}_spread"; tot_col = f"{q}_total"
            if sp_col in res.columns:
                q_pred_col = f"{q}_pred_margin" if f"{q}_pred_margin" in res.columns else None
                if q_pred_col is not None:
                    res[f"edge_{q}_spread"] = res[q_pred_col] - (-res[sp_col])
            if tot_col in res.columns and f"{q}_pred_total" in res.columns:
                res[f"edge_{q}_total"] = res[f"{q}_pred_total"] - res[tot_col]

    # 1) If merge-odds CSV provided, use it
    if merge_odds_csv:
        try:
            odds_csv_df = pd.read_csv(merge_odds_csv)
            _merge_odds_df(odds_csv_df)
        except Exception as e:
            console.print(f"Failed to merge odds from CSV: {e}", style="yellow")
    else:
        # 2) Try OddsAPI current; if empty or no key, fall back to Bovada
        import datetime as _dt
        target_date = _dt.datetime.strptime(date_str, "%Y-%m-%d").date()
        game_odds_out = paths.data_processed / f"game_odds_{date_str}.csv"
        odds_out_df = pd.DataFrame()
        # Try OddsAPI first
        api_key = os.environ.get("ODDS_API_KEY") or _load_dotenv_key("ODDS_API_KEY")
        if api_key:
            try:
                cfg = OddsApiConfig(api_key=api_key)
                long_df = fetch_game_odds_current(cfg, pd.to_datetime(target_date))
                if long_df is not None and not long_df.empty:
                    wide = consensus_lines_at_close(long_df)
                    if wide is not None and not wide.empty:
                        # Map to standardized per-game odds row
                        tmp = wide.copy()
                        tmp["date"] = pd.to_datetime(tmp["commence_time"]).dt.date
                        tmp.rename(columns={"away_team": "visitor_team"}, inplace=True)
                        tmp["home_spread"] = tmp.get("spread_point")
                        tmp["away_spread"] = tmp["home_spread"].apply(lambda x: -x if pd.notna(x) else pd.NA)
                        tmp["total"] = tmp.get("total_point")
                        cols = [
                            "date","commence_time","home_team","visitor_team",
                            "home_ml","away_ml","home_spread","away_spread","total"
                        ]
                        keep = [c for c in cols if c in tmp.columns]
                        odds_out_df = tmp[keep].copy()
                        odds_out_df["bookmaker"] = "oddsapi_consensus"
            except Exception as e:
                console.print(f"OddsAPI current odds failed: {e}", style="yellow")
        # Fallback to Bovada if still empty
        if odds_out_df is None or odds_out_df.empty:
            try:
                odds_out_df = fetch_bovada_odds_current(pd.to_datetime(target_date))
            except Exception as e:
                console.print(f"Bovada odds fetch failed: {e}", style="yellow")
        # Save standardized odds and merge
        if odds_out_df is not None and not odds_out_df.empty:
            try:
                game_odds_out.parent.mkdir(parents=True, exist_ok=True)
                odds_out_df.to_csv(game_odds_out, index=False)
                console.print({"game_odds_rows": int(len(odds_out_df)), "output": str(game_odds_out)})
            except Exception as e:
                console.print(f"Failed to save game odds CSV: {e}", style="yellow")
            _merge_odds_df(odds_out_df)

    # Save
    if not out_path:
        out_path = str(paths.data_processed / f"predictions_{date_str}.csv")

    # Guardrail: ensure one row per game.
    # Some upstream merges can accidentally introduce duplicate rows for the same matchup.
    # The frontend indexes by (date, home, visitor) and duplicates can cause arbitrary overwrite.
    try:
        dedupe_keys = [c for c in ("date", "home_team", "visitor_team") if c in res.columns]
        if len(dedupe_keys) == 3:
            before_n = int(len(res))
            tmp = res.copy()

            # Normalize key fields for consistent grouping
            try:
                tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.date
            except Exception:
                pass
            try:
                tmp["home_team"] = tmp["home_team"].apply(normalize_team)
                tmp["visitor_team"] = tmp["visitor_team"].apply(normalize_team)
            except Exception:
                pass

            # Prefer the row with the most populated prediction/market fields.
            score_cols = [
                # core game preds
                "home_win_prob",
                "spread_margin",
                "totals",
                # halves
                "halves_h1_win",
                "halves_h1_margin",
                "halves_h1_total",
                "halves_h2_win",
                "halves_h2_margin",
                "halves_h2_total",
                # quarters
                "quarters_q1_win",
                "quarters_q1_margin",
                "quarters_q1_total",
                "quarters_q2_win",
                "quarters_q2_margin",
                "quarters_q2_total",
                "quarters_q3_win",
                "quarters_q3_margin",
                "quarters_q3_total",
                "quarters_q4_win",
                "quarters_q4_margin",
                "quarters_q4_total",
                # market snapshot
                "home_ml",
                "away_ml",
                "home_spread",
                "total",
                "commence_time",
                "bookmaker",
            ]
            score_cols = [c for c in score_cols if c in tmp.columns]
            if score_cols:
                tmp["__nonnull"] = tmp[score_cols].notna().sum(axis=1).astype(int)
            else:
                tmp["__nonnull"] = 0
            tmp["__has_book"] = (tmp["bookmaker"].notna().astype(int) if "bookmaker" in tmp.columns else 0)
            tmp["__score"] = (tmp["__nonnull"] * 10) + tmp["__has_book"]

            # Stable sort then keep the best-scoring row per game.
            tmp = tmp.sort_values(
                by=["date", "home_team", "visitor_team", "__score"],
                ascending=[True, True, True, False],
                kind="mergesort",
            )
            tmp = tmp.drop_duplicates(subset=["date", "home_team", "visitor_team"], keep="first")
            tmp = tmp.drop(columns=["__nonnull", "__has_book", "__score"], errors="ignore")
            after_n = int(len(tmp))
            if after_n != before_n:
                console.print(f"Deduped predictions rows: {before_n} -> {after_n}", style="yellow")
            res = tmp
    except Exception as e:
        console.print(f"Warning: failed to dedupe predictions rows ({e}); continuing.", style="yellow")

    res.to_csv(out_path, index=False)
    console.print(f"Saved predictions to {out_path}")


@cli.command()
@click.option("--holdout-season", type=int, required=False, help="Evaluate on a single season (e.g., 2024)")
def evaluate(holdout_season: int | None):
    """Evaluate saved models on a holdout season or the latest season."""
    console.rule("Evaluate")
    feats_path = paths.data_processed / "features.parquet"
    if not feats_path.exists():
        console.print("Features not found. Run build-features first.", style="red")
        return
    df = pd.read_parquet(feats_path)
    if holdout_season is None:
        holdout_season = int(df["season"].max())
    test = df[df["season"] == holdout_season].dropna(subset=["target_home_win", "target_margin", "target_total"])  
    if test.empty:
        console.print(f"No data for season {holdout_season}", style="yellow")
        return
    # Use the same features used during training
    try:
        feat_cols = joblib.load(paths.models / "feature_columns.joblib")
    except FileNotFoundError:
        feat_cols = ["elo_diff", "home_rest_days", "visitor_rest_days", "home_b2b", "visitor_b2b"]
    X = test[feat_cols].fillna(0)
    y_win = test["target_home_win"].astype(int)
    y_margin = test["target_margin"].astype(float)
    y_total = test["target_total"].astype(float)

    import joblib
    from sklearn.metrics import log_loss, mean_squared_error  # type: ignore
    import numpy as np

    try:
        win_model = joblib.load(paths.models / "win_prob.joblib")
        spread_model = joblib.load(paths.models / "spread_margin.joblib")
        total_model = joblib.load(paths.models / "totals.joblib")
    except FileNotFoundError:
        console.print("Models not found. Run train first.", style="red")
        return

    p = win_model.predict_proba(X)[:, 1]
    ll = log_loss(y_win, p, labels=[0, 1])
    rmse_m = float(np.sqrt(mean_squared_error(y_margin, spread_model.predict(X))))
    rmse_t = float(np.sqrt(mean_squared_error(y_total, total_model.predict(X))))
    console.print({
        "season": holdout_season,
        "win_logloss": float(ll),
        "spread_rmse": float(rmse_m),
        "total_rmse": float(rmse_t),
    })


@cli.command()
@click.option("--start", type=int, required=False, help="Start season (e.g., 2018)")
@click.option("--end", type=int, required=False, help="End season inclusive (e.g., 2024)")
@click.option("--last-n", type=int, required=False, help="Evaluate last N seasons if start/end not provided")
def backtest(start: int | None, end: int | None, last_n: int | None):
    """Evaluate models across multiple seasons and summarize metrics."""
    console.rule("Backtest")
    feats_path = paths.data_processed / "features.parquet"
    if not feats_path.exists():
        console.print("Features not found. Run build-features first.", style="red")
        return
    df = pd.read_parquet(feats_path)
    all_seasons = sorted(df["season"].dropna().unique().tolist())
    if start is None and end is None:
        if last_n is None:
            last_n = 5
        seasons = all_seasons[-last_n:]
    else:
        if start is None:
            start = all_seasons[0]
        if end is None:
            end = all_seasons[-1]
        seasons = [s for s in all_seasons if start <= s <= end]
    if not seasons:
        console.print("No seasons selected for backtest.", style="yellow")
        return

    # Load models once
    try:
        win_model = joblib.load(paths.models / "win_prob.joblib")
        spread_model = joblib.load(paths.models / "spread_margin.joblib")
        total_model = joblib.load(paths.models / "totals.joblib")
    except FileNotFoundError:
        console.print("Models not found. Run train first.", style="red")
        return

    from sklearn.metrics import log_loss, mean_squared_error  # type: ignore
    import numpy as np

    try:
        feat_cols = joblib.load(paths.models / "feature_columns.joblib")
    except FileNotFoundError:
        feat_cols = ["elo_diff", "home_rest_days", "visitor_rest_days", "home_b2b", "visitor_b2b"]
    rows = []
    for s in seasons:
        part = df[df["season"] == s].dropna(subset=["target_home_win", "target_margin", "target_total"]).copy()
        if part.empty:
            continue
        X = part[feat_cols].fillna(0)
        y_win = part["target_home_win"].astype(int)
        y_margin = part["target_margin"].astype(float)
        y_total = part["target_total"].astype(float)
        p = win_model.predict_proba(X)[:, 1]
        ll = log_loss(y_win, p, labels=[0, 1])
        rmse_m = float(np.sqrt(mean_squared_error(y_margin, spread_model.predict(X))))
        rmse_t = float(np.sqrt(mean_squared_error(y_total, total_model.predict(X))))
        rows.append({"season": int(s), "win_logloss": float(ll), "spread_rmse": rmse_m, "total_rmse": rmse_t, "n_games": int(len(part))})

    if not rows:
        console.print("No data available for the requested seasons.", style="yellow")
        return

    res_df = pd.DataFrame(rows).sort_values("season")
    console.print(res_df)
    agg = {
        "seasons": f"{res_df['season'].min()}-{res_df['season'].max()}",
        "avg_win_logloss": float(res_df["win_logloss"].mean()),
        "avg_spread_rmse": float(res_df["spread_rmse"].mean()),
        "avg_total_rmse": float(res_df["total_rmse"].mean()),
        "total_games": int(res_df["n_games"].sum()),
    }
    console.print(agg)

    # Save CSV
    out = paths.data_processed / "backtest_metrics.csv"
    res_df.to_csv(out, index=False)
    console.print(f"Saved per-season metrics to {out}")


@cli.command("backtest-periods")
@click.option("--start", type=int, required=False, help="Start season (e.g., 2018)")
@click.option("--end", type=int, required=False, help="End season inclusive (e.g., 2024)")
@click.option("--last-n", type=int, required=False, help="Evaluate last N seasons if start/end not provided")
def backtest_periods(start: int | None, end: int | None, last_n: int | None):
    """Backtest halves and quarters models across seasons."""
    console.rule("Backtest (Periods)")
    feats_path = paths.data_processed / "features.parquet"
    if not feats_path.exists():
        console.print("Features not found. Run build-features first.", style="red")
        return
    df = pd.read_parquet(feats_path)
    all_seasons = sorted(df["season"].dropna().unique().tolist())
    if start is None and end is None:
        if last_n is None:
            last_n = 5
        seasons = all_seasons[-last_n:]
    else:
        if start is None:
            start = all_seasons[0]
        if end is None:
            end = all_seasons[-1]
        seasons = [s for s in all_seasons if start <= s <= end]
    if not seasons:
        console.print("No seasons selected for backtest.", style="yellow")
        return

    # Load models
    try:
        halves = joblib.load(paths.models / "halves_models.joblib")
        quarters = joblib.load(paths.models / "quarters_models.joblib")
        feat_cols = joblib.load(paths.models / "feature_columns.joblib")
    except FileNotFoundError:
        console.print("Models not found. Run train first.", style="red")
        return

    from sklearn.metrics import log_loss, mean_squared_error  # type: ignore
    import numpy as np

    results = []
    # Evaluate halves
    for half in ("h1", "h2"):
        if half not in halves:
            continue
        rows = []
        for s in seasons:
            part = df[df["season"] == s].dropna(subset=[f"target_{half}_home_win", f"target_{half}_margin", f"target_{half}_total"]).copy()
            if part.empty:
                continue
            X = part[feat_cols].fillna(0)
            y_win = part[f"target_{half}_home_win"].astype(int)
            y_margin = part[f"target_{half}_margin"].astype(float)
            y_total = part[f"target_{half}_total"].astype(float)
            p = halves[half]["win"].predict_proba(X)[:, 1]
            ll = log_loss(y_win, p, labels=[0, 1])
            rmse_m = float(np.sqrt(mean_squared_error(y_margin, halves[half]["margin"].predict(X))))
            rmse_t = float(np.sqrt(mean_squared_error(y_total, halves[half]["total"].predict(X))))
            rows.append({"season": int(s), "period": half, "win_logloss": float(ll), "spread_rmse": rmse_m, "total_rmse": rmse_t, "n": int(len(part))})
        if rows:
            results.extend(rows)

    # Evaluate quarters
    for q in ("q1", "q2", "q3", "q4"):
        if q not in quarters:
            continue
        rows = []
        for s in seasons:
            part = df[df["season"] == s].dropna(subset=[f"target_{q}_home_win", f"target_{q}_margin", f"target_{q}_total"]).copy()
            if part.empty:
                continue
            X = part[feat_cols].fillna(0)
            y_win = part[f"target_{q}_home_win"].astype(int)
            y_margin = part[f"target_{q}_margin"].astype(float)
            y_total = part[f"target_{q}_total"].astype(float)
            p = quarters[q]["win"].predict_proba(X)[:, 1]
            ll = log_loss(y_win, p, labels=[0, 1])
            rmse_m = float(np.sqrt(mean_squared_error(y_margin, quarters[q]["margin"].predict(X))))
            rmse_t = float(np.sqrt(mean_squared_error(y_total, quarters[q]["total"].predict(X))))
            rows.append({"season": int(s), "period": q, "win_logloss": float(ll), "spread_rmse": rmse_m, "total_rmse": rmse_t, "n": int(len(part))})
        if rows:
            results.extend(rows)

    if not results:
        console.print("No period data available to backtest.", style="yellow")
        return

    out_df = pd.DataFrame(results).sort_values(["period", "season"])    
    console.print(out_df)
    out_path = paths.data_processed / "backtest_periods_metrics.csv"
    out_df.to_csv(out_path, index=False)
    console.print(f"Saved period backtest metrics to {out_path}")


@cli.command("enrich-periods")
@click.option("--rate-delay", type=float, default=0.6, help="Delay between requests in seconds")
@click.option("--max-workers", type=int, default=4, help="Concurrent workers for period fetch")
@click.option("--limit", type=int, default=None, help="Limit number of games to enrich (for quick tests)")
@click.option("--seasons", type=str, default=None, help="Comma-separated season end years to enrich (e.g., 2023,2024)")
@click.option("--verbose", is_flag=True, default=False, help="Print progress while fetching")
def enrich_periods_cmd(rate_delay: float, max_workers: int, limit: int | None, seasons: str | None, verbose: bool):
    """Enrich existing raw games with quarter/OT line scores.

    Useful when LeagueGameLog is timing out or blocked; only calls BoxScoreSummaryV2
    per missing game. Writes progress to data/raw/_period_fetch_progress.txt.
    """
    console.rule("Enrich Periods")
    season_list = None
    if seasons:
        try:
            season_list = [int(s.strip()) for s in seasons.split(',') if s.strip()]
        except Exception:
            season_list = None
    df = enrich_periods_existing(rate_delay=rate_delay, verbose=verbose, max_workers=max_workers, limit=limit, seasons=season_list)
    console.print(f"Updated raw with periods; rows={len(df)}")


@cli.command("backfill-odds")
@click.option("--api-key", envvar="ODDS_API_KEY", type=str, required=False, help="OddsAPI key (or set env ODDS_API_KEY)")
@click.option("--start", type=str, required=True, help="Start date ISO (e.g., 2016-10-01T00:00:00Z)")
@click.option("--end", type=str, required=True, help="End date ISO (e.g., 2025-06-30T23:59:59Z)")
@click.option("--step-days", type=int, default=5, help="Days between historical snapshots (cost control)")
@click.option("--regions", type=str, default="us", help="OddsAPI regions (e.g., us,us2,uk)")
@click.option("--markets", type=str, default="h2h,spreads,totals", help="Markets to fetch")
@click.option("--verbose", is_flag=True, default=False)
def backfill_odds_cmd(api_key: str | None, start: str, end: str, step_days: int, regions: str, markets: str, verbose: bool):
    """Backfill NBA historical odds to data/raw/odds_nba.(parquet|csv)."""
    console.rule("Backfill (OddsAPI historical)")
    import datetime as _dt
    if not api_key:
        console.print("Provide --api-key or set ODDS_API_KEY env.", style="red"); return
    try:
        start_dt = _dt.datetime.fromisoformat(start.replace("Z","+00:00")).replace(tzinfo=None)
        end_dt = _dt.datetime.fromisoformat(end.replace("Z","+00:00")).replace(tzinfo=None)
    except Exception:
        console.print("Invalid start/end ISO datetimes.", style="red"); return
    cfg = OddsApiConfig(api_key=api_key, regions=regions, markets=markets, odds_format="american")
    df = backfill_historical_odds(cfg, start_dt, end_dt, step_days=step_days, verbose=verbose)
    console.print(f"Odds rows now: {0 if df is None else len(df)}")


@cli.command("make-closing-lines")
def make_closing_lines_cmd():
    """Compute consensus closing lines per event from data/raw/odds_nba and save to data/processed/closing_lines.parquet."""
    console.rule("Build closing lines")
    odds_parq = paths.data_raw / "odds_nba.parquet"
    odds_csv = paths.data_raw / "odds_nba.csv"
    if odds_parq.exists():
        odds_df = pd.read_parquet(odds_parq)
    elif odds_csv.exists():
        odds_df = pd.read_csv(odds_csv)
    else:
        console.print("No odds data found. Run backfill-odds.", style="red"); return
    wide = consensus_lines_at_close(odds_df)
    out = paths.data_processed / "closing_lines.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(out, index=False)
    console.print(f"Saved {len(wide)} rows to {out}")


@cli.command("export-closing-lines-csv")
@click.option("--date", "date_str", type=str, required=True, help="Game date (YYYY-MM-DD) to export consensus closing lines for")
@click.option("--out", "out_path", type=click.Path(), required=False, help="Optional output CSV path (defaults to data/processed/closing_lines_YYYY-MM-DD.csv)")
def export_closing_lines_csv_cmd(date_str: str, out_path: str | None):
    """Export per-date consensus closing lines to a CSV the frontend can consume.

    Looks for data/processed/closing_lines.parquet. If missing, falls back to
    data/raw/odds_nba.(parquet|csv) and computes consensus via consensus_lines_at_close.

    Output columns: date,home_team,visitor_team,home_ml,away_ml,home_spread,total,bookmaker
    (plus spread/total price columns if available).
    """
    console.rule("Export closing lines CSV")
    try:
        target_date = pd.to_datetime(date_str).date()
    except Exception:
        console.print("Invalid --date. Use YYYY-MM-DD.", style="red"); return

    # Load precomputed closing lines if available
    clos_parq = paths.data_processed / "closing_lines.parquet"
    wide = None
    if clos_parq.exists():
        try:
            wide = pd.read_parquet(clos_parq)
        except Exception as e:
            # No parquet engine available on ARM64? Fallback to computing from raw odds files
            console.print(f"Unable to read {clos_parq} ({e}); falling back to raw odds.", style="yellow")
            wide = None
    else:
        # Fallback: compute from raw odds
        odds_parq = paths.data_raw / "odds_nba.parquet"
        odds_csv = paths.data_raw / "odds_nba.csv"
        if odds_parq.exists():
            try:
                odds_df = pd.read_parquet(odds_parq)
            except Exception as e:
                console.print(f"Unable to read {odds_parq} ({e}); trying CSV fallback...", style="yellow")
                odds_df = None
        elif odds_csv.exists():
            odds_df = pd.read_csv(odds_csv)
        else:
            console.print("No odds data found. Run backfill-odds or make-closing-lines first.", style="red"); return
        if odds_df is None and odds_csv.exists():
            odds_df = pd.read_csv(odds_csv)
        if odds_df is None or odds_df.empty:
            console.print("No odds data available to compute closings.", style="yellow"); return
        wide = consensus_lines_at_close(odds_df)

    if wide is None or wide.empty:
        # Last-resort fallback: use per-date game_odds_{date}.csv if present
        game_odds_csv = paths.data_processed / f"game_odds_{target_date}.csv"
        if game_odds_csv.exists():
            console.print(f"No precomputed closing lines; using {game_odds_csv} as fallback.", style="yellow")
            try:
                wide = pd.read_csv(game_odds_csv)
            except Exception as e:
                console.print(f"Failed to read {game_odds_csv}: {e}", style="red"); return
        else:
            console.print("No closing lines available.", style="yellow"); return

    # Filter to the requested date by US/Eastern calendar day (handles international/UTC offsets)
    df = wide.copy()
    if "commence_time" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["commence_time"], utc=True).dt.tz_convert("US/Eastern").dt.date
        except Exception:
            # Fallback: naive date (UTC)
            df["date"] = pd.to_datetime(df["commence_time"]).dt.date
    elif "date" in df.columns:
        # Already present in fallback game_odds
        try:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        except Exception:
            pass
    df = df[df["date"] == target_date]
    if df.empty:
        console.print(f"No events found on {target_date}.", style="yellow"); return

    # Normalize and map to export schema
    if "home_team" in df.columns:
        df["home_team"] = df["home_team"].apply(normalize_team)
    # Visitor team = away team normalized
    if "visitor_team" not in df.columns and "away_team" in df.columns:
        df["visitor_team"] = df["away_team"].apply(normalize_team)
    # Map numeric columns
    if "home_spread" not in df.columns:
        df["home_spread"] = df.get("spread_point")
    if "total" not in df.columns:
        df["total"] = df.get("total_point")
    if "bookmaker" not in df.columns:
        df["bookmaker"] = "consensus"

    keep = [
        "date","home_team","visitor_team",
        "home_ml","away_ml","home_spread","total","bookmaker",
    ]
    # Include prices if present
    opt = [
        "home_spread_price","away_spread_price","total_over_price","total_under_price"
    ]
    for c in opt:
        if c in df.columns:
            keep.append(c)

    export = df[keep].sort_values(["home_team","visitor_team"]).copy()
    # Ensure date is ISO string
    export["date"] = export["date"].astype(str)

    # Decide output path
    if out_path is None:
        out_path = str(paths.data_processed / f"closing_lines_{target_date}.csv")
    outp = click.Path()(out_path)
    # Ensure folder
    paths.data_processed.mkdir(parents=True, exist_ok=True)
    export.to_csv(out_path, index=False)
    console.print({"rows": int(len(export)), "output": out_path})

@cli.command("attach-closing-lines")
def attach_closing_lines_cmd():
    """Merge consensus closing lines onto features and save as features_with_market.parquet."""
    console.rule("Attach closing lines to features")
    feats_path = paths.data_processed / "features.parquet"
    clos_path = paths.data_processed / "closing_lines.parquet"
    if not feats_path.exists():
        console.print("Features not found. Run build-features first.", style="red"); return
    if not clos_path.exists():
        console.print("Closing lines not found. Run make-closing-lines first.", style="red"); return
    df = pd.read_parquet(feats_path)
    cl = pd.read_parquet(clos_path)
    # Prepare keys: normalize team names and align dates
    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"]).dt.date
    tmp["home_team_norm"] = tmp["home_team"].apply(normalize_team)
    tmp["visitor_team_norm"] = tmp["visitor_team"].apply(normalize_team)
    cl = cl.copy()
    cl["date"] = pd.to_datetime(cl["commence_time"]).dt.date
    cl["home_team_norm"] = cl["home_team"].apply(normalize_team)
    cl["visitor_team_norm"] = cl["away_team"].apply(normalize_team)
    keep = [
        "event_id","date","home_team_norm","visitor_team_norm",
        "home_ml","away_ml","spread_point","home_spread_price","away_spread_price",
        "total_point","total_over_price","total_under_price"
    ]
    cl_small = cl[keep]
    merged = tmp.merge(
        cl_small,
        on=["date","home_team_norm","visitor_team_norm"],
        how="left",
        suffixes=("","_cl")
    )
    out = paths.data_processed / "features_with_market.parquet"
    merged.to_parquet(out, index=False)
    console.print(f"Saved merged dataset with market columns to {out} (rows={len(merged)})")


@cli.command("calibrate-win")
@click.option("--season", type=int, required=False, help="Only compute for a single season")
@click.option("--bins", type=int, default=10, help="Number of probability bins (deciles by default)")
def calibrate_win_cmd(season: int|None, bins: int):
    """Compute calibration for win probabilities (decile reliability and log-loss)."""
    console.rule("Calibrate Win Probabilities")
    feats_path = paths.data_processed / "features.parquet"
    if not feats_path.exists():
        console.print("Features not found. Run build-features first.", style="red"); return
    df = pd.read_parquet(feats_path)
    # Load win model and features
    try:
        feat_cols = joblib.load(paths.models / "feature_columns.joblib")
        win_model = joblib.load(paths.models / "win_prob.joblib")
    except FileNotFoundError:
        console.print("Models not found. Run train first.", style="red"); return
    data = df.dropna(subset=["target_home_win"]).copy()
    if season is not None:
        data = data[data["season"] == season]
        if data.empty:
            console.print(f"No data for season {season}", style="yellow"); return
    X = data[feat_cols].fillna(0)
    y = data["target_home_win"].astype(int)
    p = win_model.predict_proba(X)[:, 1]

    # Bin probabilities into equal-width intervals [0,1]
    data_cal = pd.DataFrame({"p": p, "y": y})
    data_cal["bin"] = pd.cut(data_cal["p"], bins=bins, include_lowest=True)
    summary = data_cal.groupby("bin").agg(
        n=("y", "size"),
        p_mean=("p", "mean"),
        y_rate=("y", "mean")
    ).reset_index()

    # Log-loss
    from sklearn.metrics import log_loss  # type: ignore
    ll = float(log_loss(y, p, labels=[0, 1]))
    summary["log_loss_overall"] = ll

    out = paths.data_processed / (f"calibration_win_{season}.csv" if season is not None else "calibration_win_all.csv")
    summary.to_csv(out, index=False)
    console.print(summary)
    console.print(f"Saved calibration to {out}")

@cli.command("backtest-vs-market")
@click.option("--start", type=int, required=False, help="Start season (e.g., 2018)")
@click.option("--end", type=int, required=False, help="End season inclusive (e.g., 2025)")
@click.option("--last-n", type=int, required=False, help="Evaluate last N seasons if start/end not provided")
@click.option("--win-edge", type=float, default=0.03, help="Min edge to bet moneyline (model - market implied)")
@click.option("--spread-edge", type=float, default=1.0, help="Min absolute edge (points) to bet spread")
@click.option("--total-edge", type=float, default=1.5, help="Min absolute edge (points) to bet total")
@click.option("--default-spread-price", type=int, default=-110, help="Assumed price for ATS when side price unavailable")
@click.option("--default-total-price", type=int, default=-110, help="Assumed price for totals when side price unavailable")
@click.option("--stake-mode", type=click.Choice(["flat","kelly","half-kelly"]), default="flat", help="Staking method for bankroll simulation")
@click.option("--bankroll-start", type=float, default=100.0, help="Starting bankroll for simulation")
@click.option("--kelly-cap", type=float, default=0.05, help="Max fraction of bankroll per bet for Kelly staking")
@click.option("--sigma-margin", type=float, default=13.5, help="Sigma (points) for ATS probability model")
@click.option("--sigma-total", type=float, default=19.5, help="Sigma (points) for Totals probability model")
def backtest_vs_market_cmd(start: int|None, end: int|None, last_n: int|None, win_edge: float, spread_edge: float, total_edge: float, default_spread_price: int, default_total_price: int, stake_mode: str, bankroll_start: float, kelly_cap: float, sigma_margin: float, sigma_total: float):
    """Simulate ROI vs consensus closing lines using model predictions.

    Bets:
    - Moneyline: bet home or away if model implied edge >= win-edge.
    - Spread: bet ATS on the side of model margin if |pred - market| >= spread-edge.
    - Total: bet Over/Under if |pred - total| >= total-edge.

    Prices: uses closing consensus prices when available, otherwise assumed -110 for ATS/O/U. One bet per market per game.
    """
    console.rule("Backtest vs Market")
    feats_path = paths.data_processed / "features_with_market.parquet"
    if not feats_path.exists():
        console.print("features_with_market.parquet not found. Run attach-closing-lines first.", style="red"); return

    df = pd.read_parquet(feats_path)
    all_seasons = sorted(df["season"].dropna().unique().tolist())
    if start is None and end is None:
        if last_n is None:
            last_n = 5
        seasons = all_seasons[-last_n:]
    else:
        if start is None:
            start = all_seasons[0]
        if end is None:
            end = all_seasons[-1]
        seasons = [s for s in all_seasons if start <= s <= end]
    if not seasons:
        console.print("No seasons selected for backtest.", style="yellow"); return

    # Load models and feature columns
    try:
        feat_cols = joblib.load(paths.models / "feature_columns.joblib")
        win_model = joblib.load(paths.models / "win_prob.joblib")
        spread_model = joblib.load(paths.models / "spread_margin.joblib")
        total_model = joblib.load(paths.models / "totals.joblib")
    except FileNotFoundError:
        console.print("Models not found. Run train first.", style="red"); return

    def implied_prob_american(o: float|int|None):
        if o is None or pd.isna(o):
            return None
        o = float(o)
        if o < 0:
            return (-o) / ((-o) + 100.0)
        return 100.0 / (o + 100.0)

    def profit_from_american(price: float|int, stake: float = 1.0) -> float:
        price = float(price)
        if price > 0:
            return stake * (price / 100.0)
        else:
            return stake * (100.0 / abs(price))

    def american_to_b(price: float|int) -> float:
        price = float(price)
        return (price / 100.0) if price > 0 else (100.0 / abs(price))

    import math
    def norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    season_rows = []
    summary_bets = []
    for s in seasons:
        part = df[(df["season"] == s)].copy()
        if part.empty:
            continue
        X = part[feat_cols].fillna(0)
        part["pred_home_win_prob"] = win_model.predict_proba(X)[:, 1]
        part["pred_margin"] = spread_model.predict(X)
        part["pred_total"] = total_model.predict(X)
        # Market values
        part["market_home_margin"] = -part["spread_point"]
        part["market_total"] = part["total_point"]

        # Moneyline bets
        ml_bets = []
        for _, r in part.dropna(subset=["home_ml","away_ml"]).iterrows():
            ph = r["pred_home_win_prob"]
            ih = implied_prob_american(r["home_ml"]) or 0
            ia = implied_prob_american(r["away_ml"]) or 0
            home_edge = ph - ih
            away_edge = (1 - ph) - ia
            if home_edge >= win_edge and home_edge >= away_edge:
                outcome = 1 if (r.get("home_pts") > r.get("visitor_pts")) else 0 if pd.notna(r.get("home_pts")) else None
                price = r["home_ml"]
                ml_bets.append({"season": s, "market": "ML", "side": "home", "edge": home_edge, "price": price, "won": outcome, "date": pd.to_datetime(r.get("date")).date(), "p_model": ph})
            elif away_edge >= win_edge:
                outcome = 1 if (r.get("home_pts") < r.get("visitor_pts")) else 0 if pd.notna(r.get("home_pts")) else None
                price = r["away_ml"]
                ml_bets.append({"season": s, "market": "ML", "side": "away", "edge": away_edge, "price": price, "won": outcome, "date": pd.to_datetime(r.get("date")).date(), "p_model": (1 - ph)})

    # Spread bets (use both-side prices if available; assume -110 if missing)
        ats_bets = []
        for _, r in part.dropna(subset=["market_home_margin"]).iterrows():
            edge = r["pred_margin"] - r["market_home_margin"]
            if abs(edge) >= spread_edge:
                if edge > 0:
                    # Bet home ATS
                    won = None
                    if pd.notna(r.get("home_pts")):
                        margin = r.get("home_pts") - r.get("visitor_pts")
                        spread = r["market_home_margin"]
                        won = 1 if margin > spread else 0 if margin < spread else 0  # push counts as 0 profit
                    price = r.get("home_spread_price") if pd.notna(r.get("home_spread_price")) else default_spread_price
                    # probability of cover via normal assumption
                    mu = r["pred_margin"]; T = r["market_home_margin"]
                    p_cov = 1 - norm_cdf((T - mu) / max(1e-6, sigma_margin))
                    ats_bets.append({"season": s, "market": "ATS", "side": "home", "edge": edge, "price": price, "won": won, "date": pd.to_datetime(r.get("date")).date(), "p_model": p_cov})
                else:
                    # Bet away ATS
                    won = None
                    if pd.notna(r.get("home_pts")):
                        margin = r.get("home_pts") - r.get("visitor_pts")
                        spread = r["market_home_margin"]
                        # Away +X wins if margin < spread
                        won = 1 if margin < spread else 0 if margin > spread else 0
                    price = r.get("away_spread_price") if pd.notna(r.get("away_spread_price")) else default_spread_price
                    mu = r["pred_margin"]; T = r["market_home_margin"]
                    p_cov = norm_cdf((T - mu) / max(1e-6, sigma_margin))
                    ats_bets.append({"season": s, "market": "ATS", "side": "away", "edge": -edge, "price": price, "won": won, "date": pd.to_datetime(r.get("date")).date(), "p_model": p_cov})

    # Totals bets (use Over/Under prices if available)
        tot_bets = []
        for _, r in part.dropna(subset=["market_total"]).iterrows():
            diff = r["pred_total"] - r["market_total"]
            if abs(diff) >= total_edge:
                if diff > 0:
                    # Over
                    won = None
                    if pd.notna(r.get("home_pts")):
                        total = r.get("home_pts") + r.get("visitor_pts")
                        won = 1 if total > r["market_total"] else 0 if total < r["market_total"] else 0
                    price = r.get("total_over_price") if pd.notna(r.get("total_over_price")) else default_total_price
                    mu = r["pred_total"]; T = r["market_total"]
                    p_over = 1 - norm_cdf((T - mu) / max(1e-6, sigma_total))
                    tot_bets.append({"season": s, "market": "TOTAL", "side": "over", "edge": diff, "price": price, "won": won, "date": pd.to_datetime(r.get("date")).date(), "p_model": p_over})
                else:
                    # Under
                    won = None
                    if pd.notna(r.get("home_pts")):
                        total = r.get("home_pts") + r.get("visitor_pts")
                        won = 1 if total < r["market_total"] else 0 if total > r["market_total"] else 0
                    price = r.get("total_under_price") if pd.notna(r.get("total_under_price")) else default_total_price
                    mu = r["pred_total"]; T = r["market_total"]
                    p_under = norm_cdf((T - mu) / max(1e-6, sigma_total))
                    tot_bets.append({"season": s, "market": "TOTAL", "side": "under", "edge": -diff, "price": price, "won": won, "date": pd.to_datetime(r.get("date")).date(), "p_model": p_under})

        # Aggregate ROI
        def summarize(bets: list[dict], label: str) -> dict:
            if not bets:
                return {"season": s, "market": label, "n_bets": 0, "roi": 0.0, "hit_rate": 0.0}
            settled = [b for b in bets if b["won"] is not None]
            units = 0.0
            hits = 0
            edges = []
            evs = []
            for b in settled:
                if b["won"] == 1:
                    units += profit_from_american(b["price"], 1.0)
                    hits += 1
                elif b["won"] == 0:
                    units -= 1.0
                # Expected value per unit (using model probability if present)
                p_mod = b.get("p_model")
                if p_mod is not None and not pd.isna(p_mod):
                    b_odds = american_to_b(b["price"])  # decimal net odds
                    ev = p_mod * b_odds - (1 - p_mod) * 1.0
                    evs.append(ev)
                e = b.get("edge")
                if e is not None:
                    edges.append(float(e))
            n = len(settled)
            roi = units / n if n > 0 else 0.0
            hr = hits / n if n > 0 else 0.0
            return {
                "season": s,
                "market": label,
                "n_bets": n,
                "roi": float(roi),
                "hit_rate": float(hr),
                "avg_edge": float(pd.Series(edges).mean()) if edges else None,
                "avg_ev": float(pd.Series(evs).mean()) if evs else None,
            }

        ml_sum = summarize(ml_bets, "ML")
        ats_sum = summarize(ats_bets, "ATS")
        tot_sum = summarize(tot_bets, "TOTAL")
        season_rows.extend([ml_sum, ats_sum, tot_sum])
        # Compute implied prob and Kelly fraction for ledger
        def enrich(b):
            out = []
            for bet in b:
                price = bet["price"]
                p = bet.get("p_model")
                imp = implied_prob_american(price)
                kelly = None
                if p is not None and not pd.isna(p):
                    bdec = american_to_b(price)
                    q = 1 - p
                    kelly = max(0.0, (bdec * p - q) / bdec)
                bet["implied_prob"] = float(imp) if imp is not None else None
                bet["kelly_fraction"] = float(kelly) if kelly is not None else None
                # expected value per unit
                if p is not None and not pd.isna(p):
                    ev = p * american_to_b(price) - (1 - p)
                    bet["ev_unit"] = float(ev)
                out.append(bet)
            return out
        ml_bets = enrich(ml_bets)
        ats_bets = enrich(ats_bets)
        tot_bets = enrich(tot_bets)
        summary_bets.extend(ml_bets + ats_bets + tot_bets)

    # Results (unit-based)
    res_df = pd.DataFrame(season_rows).sort_values(["market","season"]) if season_rows else pd.DataFrame()
    console.print(res_df)
    out_season = paths.data_processed / "backtest_vs_market.csv"
    res_df.to_csv(out_season, index=False)
    # Save raw bet ledger
    ledger = pd.DataFrame(summary_bets)
    out_ledger = paths.data_processed / "backtest_vs_market_ledger.csv"
    ledger.to_csv(out_ledger, index=False)
    console.print(f"Saved summaries to {out_season} and ledger to {out_ledger}")

    # Bankroll simulation per season and market
    if not ledger.empty:
        ledger["date"] = pd.to_datetime(ledger["date"]).dt.date
        bank_summaries = []
        curves = []
        for (s, m), group in ledger.groupby(["season","market"], sort=True):
            g = group.sort_values("date").copy()
            bankroll = float(bankroll_start)
            for idx, row in g.iterrows():
                price = row["price"]
                b = american_to_b(price)
                p = row.get("p_model")
                stake = 1.0  # default flat
                if stake_mode in ("kelly","half-kelly") and p is not None and not pd.isna(p):
                    q = 1.0 - float(p)
                    f_star = max(0.0, (b * float(p) - q) / b)
                    if stake_mode == "half-kelly":
                        f_star *= 0.5
                    f_star = min(f_star, float(kelly_cap))
                    stake = bankroll * f_star
                pnl = 0.0
                if row["won"] == 1:
                    pnl = profit_from_american(price, stake)
                elif row["won"] == 0:
                    pnl = -stake
                bankroll += pnl
                curves.append({"season": s, "market": m, "date": row["date"], "stake": float(stake), "price": float(price), "p_model": float(p) if p is not None else None, "pnl": float(pnl), "bankroll": float(bankroll)})
            roi = (bankroll / bankroll_start) - 1.0
            bank_summaries.append({"season": s, "market": m, "start": float(bankroll_start), "end": float(bankroll), "roi_bankroll": float(roi)})
        bank_df = pd.DataFrame(bank_summaries).sort_values(["market","season"]) if bank_summaries else pd.DataFrame()
        curve_df = pd.DataFrame(curves).sort_values(["market","season","date"]) if curves else pd.DataFrame()
        out_bank = paths.data_processed / "backtest_vs_market_bankroll.csv"
        out_curve = paths.data_processed / "backtest_vs_market_bankroll_curve.csv"
        bank_df.to_csv(out_bank, index=False)
        curve_df.to_csv(out_curve, index=False)
        console.print(f"Saved bankroll summaries to {out_bank} and curves to {out_curve}")


@cli.command("make-synthetic-period-lines")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), default=str(paths.data_processed / "period_lines_synthetic.csv"), help="Output CSV path for synthetic period lines")
def make_synthetic_period_lines_cmd(out_path: str):
    """Create synthetic period (halves/quarters) lines from full-game closers using empirical ratios.

    - Totals: scale full-game total by average scoring share per period (computed from raw games with period data).
    - Spreads: scale full-game home spread by time fraction (0.5 for halves, 0.25 per quarter).
    Produces a CSV compatible with backtest-periods-vs-market.
    """
    console.rule("Make synthetic period lines")
    # Load raw with periods to estimate scoring shares
    raw_parq = paths.data_raw / "games_nba_api.parquet"
    raw_csv = paths.data_raw / "games_nba_api.csv"
    if raw_parq.exists():
        raw = pd.read_parquet(raw_parq)
    elif raw_csv.exists():
        raw = pd.read_csv(raw_csv)
    else:
        console.print("No raw games file found.", style="red"); return
    # Ensure totals available
    if "total_points" not in raw.columns:
        if {"home_pts","visitor_pts"}.issubset(raw.columns):
            raw["total_points"] = raw[["home_pts","visitor_pts"]].sum(axis=1)
        else:
            console.print("Raw missing total points.", style="red"); return
    # Compute shares where available (robust to NaNs)
    shares = {}
    # Halves
    for half in ("h1","h2"):
        hcol = f"home_{half}"; vcol = f"visitor_{half}"
        if hcol in raw.columns and vcol in raw.columns:
            hp = pd.to_numeric(raw[hcol], errors="coerce")
            vp = pd.to_numeric(raw[vcol], errors="coerce")
            tp = pd.to_numeric(raw["total_points"], errors="coerce")
            mask = hp.notna() & vp.notna() & tp.notna() & (tp > 0)
            ratio = (hp[mask] + vp[mask]) / tp[mask]
            shares[half] = float(ratio.mean()) if not ratio.empty else None
        else:
            shares[half] = None
    # Quarters
    for q in ("q1","q2","q3","q4"):
        hcol = f"home_{q}"; vcol = f"visitor_{q}"
        if hcol in raw.columns and vcol in raw.columns:
            hp = pd.to_numeric(raw[hcol], errors="coerce")
            vp = pd.to_numeric(raw[vcol], errors="coerce")
            tp = pd.to_numeric(raw["total_points"], errors="coerce")
            mask = hp.notna() & vp.notna() & tp.notna() & (tp > 0)
            ratio = (hp[mask] + vp[mask]) / tp[mask]
            shares[q] = float(ratio.mean()) if not ratio.empty else None
        else:
            shares[q] = None
    # Fallbacks if missing: use time fractions
    shares.setdefault("h1", 0.5); shares.setdefault("h2", 0.5)
    for q in ("q1","q2","q3","q4"):
        shares.setdefault(q, 0.25)

    # Load features_with_market to get full-game spread/total per matchup
    fwm = paths.data_processed / "features_with_market.parquet"
    feats = paths.data_processed / "features.parquet"
    if fwm.exists():
        base = pd.read_parquet(fwm)
    elif feats.exists():
        base = pd.read_parquet(feats)
    else:
        console.print("No features found. Run build-features first.", style="red"); return
    base = base.copy()
    base["date"] = pd.to_datetime(base["date"]).dt.date
    # Ensure we have market columns (spread_point, total_point). If missing, drop.
    if "spread_point" not in base.columns and "total_point" not in base.columns:
        console.print("No closing lines attached. Run make-closing-lines and attach-closing-lines first.", style="red"); return

    rows = []
    for _, r in base.iterrows():
        total = r.get("total_point")
        spread = r.get("spread_point")  # home spread (negative if favored)
        if pd.isna(total) and pd.isna(spread):
            continue
        out = {
            "date": r.get("date"),
            "home_team": r.get("home_team"),
            "visitor_team": r.get("visitor_team"),
        }
        # Totals
        if pd.notna(total):
            out["h1_total"] = float(total) * shares["h1"]
            out["h2_total"] = float(total) * shares["h2"]
            for q in ("q1","q2","q3","q4"):
                out[f"{q}_total"] = float(total) * shares[q]
        # Spreads (scale by time fraction)
        if pd.notna(spread):
            out["h1_spread"] = float(spread) * 0.5
            out["h2_spread"] = float(spread) * 0.5
            for q in ("q1","q2","q3","q4"):
                out[f"{q}_spread"] = float(spread) * 0.25
        if any(k.endswith('_total') or k.endswith('_spread') for k in out.keys()):
            rows.append(out)
    out_df = pd.DataFrame(rows)
    from pathlib import Path
    out_p = Path(out_path) if out_path else (paths.data_processed / "period_lines_synthetic.csv")
    # If relative, write under data/processed
    if not out_p.is_absolute():
        out_p = paths.data_processed / out_p
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_p, index=False)
    console.print(f"Saved synthetic period lines to {out_p} (rows={len(out_df)})")

@cli.command("backtest-periods-vs-market")
@click.option("--odds-csv", type=click.Path(exists=True), required=True, help="CSV containing period odds by game: date,home_team,visitor_team and any of: h1_spread,h1_total,h2_spread,h2_total,q1_spread,q1_total,q2_spread,q2_total,q3_spread,q3_total,q4_spread,q4_total. Optional price columns: *_home_spread_price,*_away_spread_price,*_total_over_price,*_total_under_price.")
@click.option("--start", type=int, required=False, help="Start season (e.g., 2018)")
@click.option("--end", type=int, required=False, help="End season inclusive (e.g., 2025)")
@click.option("--last-n", type=int, required=False, help="Evaluate last N seasons if start/end not provided")
@click.option("--spread-edge", type=float, default=1.0, help="Min absolute edge (points) to bet period spreads")
@click.option("--total-edge", type=float, default=1.5, help="Min absolute edge (points) to bet period totals")
@click.option("--default-spread-price", type=int, default=-110, help="Assumed price for ATS when side price unavailable")
@click.option("--default-total-price", type=int, default=-110, help="Assumed price for totals when side price unavailable")
@click.option("--stake-mode", type=click.Choice(["flat","kelly","half-kelly"]), default="flat", help="Staking method for bankroll simulation")
@click.option("--bankroll-start", type=float, default=100.0, help="Starting bankroll for simulation")
@click.option("--kelly-cap", type=float, default=0.05, help="Max fraction of bankroll per bet for Kelly staking")
@click.option("--sigma-margin-half", type=float, default=9.6, help="Sigma (points) for Half ATS probability model; ~13.5/sqrt(2)")
@click.option("--sigma-total-half", type=float, default=13.8, help="Sigma (points) for Half Totals probability model; ~19.5/sqrt(2)")
@click.option("--sigma-margin-quarter", type=float, default=6.8, help="Sigma (points) for Quarter ATS probability model; ~13.5/2")
@click.option("--sigma-total-quarter", type=float, default=9.8, help="Sigma (points) for Quarter Totals probability model; ~19.5/2")
def backtest_periods_vs_market_cmd(odds_csv: str, start: int|None, end: int|None, last_n: int|None, spread_edge: float, total_edge: float, default_spread_price: int, default_total_price: int, stake_mode: str, bankroll_start: float, kelly_cap: float, sigma_margin_half: float, sigma_total_half: float, sigma_margin_quarter: float, sigma_total_quarter: float):
    """Simulate ROI vs period (halves/quarters) odds using model period predictions.

    Requires a CSV with period markets per game. Bets placed when model edges exceed thresholds; bankroll simulated by season and market.
    """
    console.rule("Backtest Periods vs Market")
    feats_path = paths.data_processed / "features.parquet"
    if not feats_path.exists():
        console.print("features.parquet not found. Run build-features first.", style="red"); return

    df = pd.read_parquet(feats_path)
    all_seasons = sorted(df["season"].dropna().unique().tolist())
    if start is None and end is None:
        if last_n is None:
            last_n = 5
        seasons = all_seasons[-last_n:]
    else:
        if start is None:
            start = all_seasons[0]
        if end is None:
            end = all_seasons[-1]
        seasons = [s for s in all_seasons if start <= s <= end]
    if not seasons:
        console.print("No seasons selected for backtest.", style="yellow"); return

    # Load models and feature columns
    try:
        feat_cols = joblib.load(paths.models / "feature_columns.joblib")
        halves = joblib.load(paths.models / "halves_models.joblib")
        quarters = joblib.load(paths.models / "quarters_models.joblib")
    except FileNotFoundError:
        console.print("Period models not found. Run train first.", style="red"); return

    # Read period odds CSV
    odds = pd.read_csv(odds_csv)
    # Normalize keys
    if 'date' in odds.columns:
        odds['date'] = pd.to_datetime(odds['date']).dt.date
    odds['home_team'] = odds['home_team'].apply(normalize_team)
    odds['visitor_team'] = odds['visitor_team'].apply(normalize_team)

    # Prepare feature base with period predictions
    base = df[df['season'].isin(seasons)].copy()
    base['date'] = pd.to_datetime(base['date']).dt.date
    base['home_team'] = base['home_team'].apply(normalize_team)
    base['visitor_team'] = base['visitor_team'].apply(normalize_team)
    X = base[feat_cols].fillna(0)
    # Halves predictions
    for half in ("h1","h2"):
        if half in halves:
            base[f"{half}_pred_margin"] = halves[half]["margin"].predict(X)
            base[f"{half}_pred_total"] = halves[half]["total"].predict(X)
    # Quarters predictions
    for q in ("q1","q2","q3","q4"):
        if q in quarters:
            base[f"{q}_pred_margin"] = quarters[q]["margin"].predict(X)
            base[f"{q}_pred_total"] = quarters[q]["total"].predict(X)

    # Merge odds
    merged = base.merge(odds, on=['date','home_team','visitor_team'], how='left', suffixes=("","_odds"))

    import math
    def norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def american_to_b(price: float|int) -> float:
        price = float(price)
        return (price / 100.0) if price > 0 else (100.0 / abs(price))

    def profit_from_american(price: float|int, stake: float = 1.0) -> float:
        price = float(price)
        return stake * (price / 100.0) if price > 0 else stake * (100.0 / abs(price))

    period_specs = [
        ("h1", sigma_margin_half, sigma_total_half),
        ("h2", sigma_margin_half, sigma_total_half),
        ("q1", sigma_margin_quarter, sigma_total_quarter),
        ("q2", sigma_margin_quarter, sigma_total_quarter),
        ("q3", sigma_margin_quarter, sigma_total_quarter),
        ("q4", sigma_margin_quarter, sigma_total_quarter),
    ]

    season_rows = []
    ledger_rows = []
    for s in seasons:
        part = merged[merged['season'] == s].copy()
        if part.empty:
            continue
        # For each period, form bets from available odds columns
        for period, sig_m, sig_t in period_specs:
            # Spread bets
            sp_col = f"{period}_spread"
            sp_hp = f"{period}_home_spread_price"; sp_ap = f"{period}_away_spread_price"
            pred_m = f"{period}_pred_margin"
            tgt_m = f"target_{period}_margin"
            if sp_col in part.columns and pred_m in part.columns:
                for _, r in part.dropna(subset=[sp_col, pred_m]).iterrows():
                    edge = r[pred_m] - (-r[sp_col])
                    if abs(edge) >= spread_edge:
                        if edge > 0:
                            won = None
                            if tgt_m in part.columns and pd.notna(r.get(tgt_m)):
                                spread = -r[sp_col]
                                margin = float(r.get(tgt_m))
                                won = 1 if margin > spread else 0 if margin < spread else 0
                            price = r.get(sp_hp) if pd.notna(r.get(sp_hp)) else default_spread_price
                            mu = r[pred_m]; T = -r[sp_col]
                            p_cov = 1 - norm_cdf((T - mu) / max(1e-6, sig_m))
                            ledger_rows.append({"season": s, "market": f"{period}_ATS", "side": "home", "edge": edge, "price": price, "won": won, "date": pd.to_datetime(r.get("date")).date(), "p_model": p_cov})
                        else:
                            won = None
                            if tgt_m in part.columns and pd.notna(r.get(tgt_m)):
                                spread = -r[sp_col]
                                margin = float(r.get(tgt_m))
                                won = 1 if margin < spread else 0 if margin > spread else 0
                            price = r.get(sp_ap) if pd.notna(r.get(sp_ap)) else default_spread_price
                            mu = r[pred_m]; T = -r[sp_col]
                            p_cov = norm_cdf((T - mu) / max(1e-6, sig_m))
                            ledger_rows.append({"season": s, "market": f"{period}_ATS", "side": "away", "edge": -edge, "price": price, "won": won, "date": pd.to_datetime(r.get("date")).date(), "p_model": p_cov})

            # Totals bets
            tot_col = f"{period}_total"
            over_p = f"{period}_total_over_price"; under_p = f"{period}_total_under_price"
            pred_t = f"{period}_pred_total"
            tgt_t = f"target_{period}_total"
            if tot_col in part.columns and pred_t in part.columns:
                for _, r in part.dropna(subset=[tot_col, pred_t]).iterrows():
                    diff = r[pred_t] - r[tot_col]
                    if abs(diff) >= total_edge:
                        if diff > 0:
                            won = None
                            if tgt_t in part.columns and pd.notna(r.get(tgt_t)):
                                total = float(r.get(tgt_t))
                                won = 1 if total > r[tot_col] else 0 if total < r[tot_col] else 0
                            price = r.get(over_p) if pd.notna(r.get(over_p)) else default_total_price
                            mu = r[pred_t]; T = r[tot_col]
                            p_over = 1 - norm_cdf((T - mu) / max(1e-6, sig_t))
                            ledger_rows.append({"season": s, "market": f"{period}_TOTAL", "side": "over", "edge": diff, "price": price, "won": won, "date": pd.to_datetime(r.get("date")).date(), "p_model": p_over})
                        else:
                            won = None
                            if tgt_t in part.columns and pd.notna(r.get(tgt_t)):
                                total = float(r.get(tgt_t))
                                won = 1 if total < r[tot_col] else 0 if total > r[tot_col] else 0
                            price = r.get(under_p) if pd.notna(r.get(under_p)) else default_total_price
                            mu = r[pred_t]; T = r[tot_col]
                            p_under = norm_cdf((T - mu) / max(1e-6, sig_t))
                            ledger_rows.append({"season": s, "market": f"{period}_TOTAL", "side": "under", "edge": -diff, "price": price, "won": won, "date": pd.to_datetime(r.get("date")).date(), "p_model": p_under})

        # Summarize per season by market
    def summarize(ledger: pd.DataFrame, label_prefix: str) -> list[dict]:
        out = []
        for (s, m), grp in ledger.groupby(["season","market"], sort=True):
            units = 0.0; hits = 0; n = 0; edges=[]; evs=[]
            for _, b in grp.dropna(subset=["won"]).iterrows():
                n += 1
                if b["won"] == 1:
                    units += profit_from_american(b["price"], 1.0)
                    hits += 1
                elif b["won"] == 0:
                    units -= 1.0
                if pd.notna(b.get("edge")):
                    edges.append(float(b["edge"]))
                p = b.get("p_model")
                if p is not None and not pd.isna(p):
                    ev = float(p) * american_to_b(b["price"]) - (1 - float(p))
                    evs.append(ev)
            roi = (units / n) if n > 0 else 0.0
            hr = (hits / n) if n > 0 else 0.0
            out.append({"season": int(s), "market": m, "n_bets": n, "roi": float(roi), "hit_rate": float(hr), "avg_edge": float(pd.Series(edges).mean()) if edges else None, "avg_ev": float(pd.Series(evs).mean()) if evs else None})
        return out

    # Build ledger DataFrame
    ledger = pd.DataFrame(ledger_rows)
    res_df = pd.DataFrame(summarize(ledger, "period")) if not ledger.empty else pd.DataFrame()
    console.print(res_df)
    out_season = paths.data_processed / "backtest_periods_vs_market.csv"
    res_df.to_csv(out_season, index=False)
    out_ledger = paths.data_processed / "backtest_periods_vs_market_ledger.csv"
    ledger.to_csv(out_ledger, index=False)
    console.print(f"Saved summaries to {out_season} and ledger to {out_ledger}")

    # Bankroll simulation by season and market
    if not ledger.empty:
        ledger["date"] = pd.to_datetime(ledger["date"]).dt.date
        bank_summaries = []
        curves = []
        for (s, m), group in ledger.groupby(["season","market"], sort=True):
            g = group.sort_values("date").copy()
            bankroll = float(bankroll_start)
            for _, row in g.iterrows():
                price = row["price"]
                b = american_to_b(price)
                p = row.get("p_model")
                stake = 1.0
                if stake_mode in ("kelly","half-kelly") and p is not None and not pd.isna(p):
                    q = 1.0 - float(p)
                    f_star = max(0.0, (b * float(p) - q) / b)
                    if stake_mode == "half-kelly":
                        f_star *= 0.5
                    f_star = min(f_star, float(kelly_cap))
                    stake = bankroll * f_star
                pnl = 0.0
                if row["won"] == 1:
                    pnl = profit_from_american(price, stake)
                elif row["won"] == 0:
                    pnl = -stake
                bankroll += pnl
                curves.append({"season": s, "market": m, "date": row["date"], "stake": float(stake), "price": float(price), "p_model": float(p) if p is not None else None, "pnl": float(pnl), "bankroll": float(bankroll)})
            roi = (bankroll / bankroll_start) - 1.0
            bank_summaries.append({"season": s, "market": m, "start": float(bankroll_start), "end": float(bankroll), "roi_bankroll": float(roi)})
        bank_df = pd.DataFrame(bank_summaries).sort_values(["market","season"]) if bank_summaries else pd.DataFrame()
        curve_df = pd.DataFrame(curves).sort_values(["market","season","date"]) if curves else pd.DataFrame()
        out_bank = paths.data_processed / "backtest_periods_vs_market_bankroll.csv"
        out_curve = paths.data_processed / "backtest_periods_vs_market_bankroll_curve.csv"
        bank_df.to_csv(out_bank, index=False)
        curve_df.to_csv(out_curve, index=False)
        console.print(f"Saved bankroll summaries to {out_bank} and curves to {out_curve}")

@cli.command("reconcile-date")
@click.option("--date", "date_str", type=str, required=True, help="Game date YYYY-MM-DD to reconcile")
@click.option("--predictions", "pred_path", type=click.Path(exists=False, dir_okay=False), required=False, help="Optional predictions CSV path; defaults to data/processed/predictions_YYYY-MM-DD.csv then repo-root predictions_YYYY-MM-DD.csv")
def reconcile_date_cmd(date_str: str, pred_path: str | None):
    """Build reconciliation CSV for a date by joining predictions with NBA final scores.

    Writes data/processed/recon_games_YYYY-MM-DD.csv
    Columns: date, home_team, visitor_team, home_tri, away_tri, home_pts, visitor_pts, pred_margin, pred_total, actual_margin, total_actual, margin_error, total_error
    """
    console.rule("Reconcile (Date)")
    # Locate predictions file
    from .config import paths as _paths
    try:
        target_date = pd.to_datetime(date_str).date()
    except Exception:
        console.print("Invalid --date (YYYY-MM-DD)", style="red"); return
    default_proc = _paths.data_processed / f"predictions_{target_date}.csv"
    default_root = _paths.root / f"predictions_{target_date}.csv"
    pred_file = None
    if pred_path:
        from pathlib import Path as _P
        p = _P(pred_path)
        pred_file = p if p.is_absolute() else (_paths.root / pred_path)
        if not pred_file.exists():
            pred_file = None
    if pred_file is None:
        pred_file = default_proc if default_proc.exists() else (default_root if default_root.exists() else None)
    if pred_file is None:
        console.print("Predictions CSV not found for date.", style="yellow"); return
    try:
        preds = pd.read_csv(pred_file)
    except Exception as e:
        console.print(f"Failed to read predictions: {e}", style="red"); return

    # Normalize prediction columns to a consistent schema.
    # Some pipelines output totals/spread_margin (or model_total) rather than pred_total/pred_margin.
    def _first_col(df: pd.DataFrame, names: list[str]) -> str | None:
        for n in names:
            if n in df.columns:
                return n
        return None

    preds = preds.copy()
    total_src = _first_col(preds, ["totals", "model_total", "total"])
    margin_src = _first_col(preds, ["spread_margin", "model_margin", "margin"])

    # Create or backfill pred_total / pred_margin from other common columns.
    if "pred_total" not in preds.columns:
        if total_src is not None:
            preds["pred_total"] = preds[total_src]
    else:
        if total_src is not None:
            preds["pred_total"] = preds["pred_total"].where(preds["pred_total"].notna(), preds[total_src])

    if "pred_margin" not in preds.columns:
        if margin_src is not None:
            preds["pred_margin"] = preds[margin_src]
    else:
        if margin_src is not None:
            preds["pred_margin"] = preds["pred_margin"].where(preds["pred_margin"].notna(), preds[margin_src])
    # Normalize to tricodes using nba_api static map
    try:
        team_list = static_teams.get_teams()
        full_to_abbr = {str(t.get('full_name')).upper(): str(t.get('abbreviation')).upper() for t in team_list}
        alt = {
            "LOS ANGELES CLIPPERS": "LAC",
            "LA CLIPPERS": "LAC",
            "PHOENIX SUNS": "PHX",
            "GOLDEN STATE WARRIORS": "GSW",
            "SAN ANTONIO SPURS": "SAS",
            "NEW YORK KNICKS": "NYK",
            "BROOKLYN NETS": "BKN",
            "UTAH JAZZ": "UTA",
        }
        def to_tri(name: str) -> str:
            if name is None:
                return ""
            s = str(name).strip().upper()
            if s in full_to_abbr:
                return full_to_abbr[s]
            if s in alt:
                return alt[s]
            if len(s) <= 4:
                return s
            return s
        preds["home_tri"] = preds.get("home_team").apply(to_tri)
        preds["away_tri"] = preds.get("visitor_team").apply(to_tri)
    except Exception:
        preds["home_tri"] = preds.get("home_team").astype(str).str.upper()
        preds["away_tri"] = preds.get("visitor_team").astype(str).str.upper()
    # Prefer processed finals CSV if available; else fetch via APIs with fallbacks
    try:
        finals = None
        # 0) Use previously exported finals if present
        try:
            from .config import paths as _paths2
            fpath = _paths2.data_processed / f"finals_{target_date}.csv"
            if fpath.exists():
                fdf = pd.read_csv(fpath)
                # Normalize column names
                cols = {c.lower(): c for c in fdf.columns}
                need = {"home_tri","away_tri","home_pts","visitor_pts"}
                if need.issubset(set(cols.keys())):
                    finals = fdf[[cols["home_tri"], cols["away_tri"], cols["home_pts"], cols["visitor_pts"]]].copy()
                    finals.columns = ["home_tri","away_tri","home_pts","visitor_pts"]
        except Exception:
            finals = None

        # 1) If no processed finals, fetch via ScoreboardV2
        try:
            nba_http.STATS_HEADERS.update({
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Origin': 'https://www.nba.com',
                'Referer': 'https://www.nba.com/stats/',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
                'Connection': 'keep-alive',
            })
        except Exception:
            pass
        def fetch_finals_for(game_date: str) -> pd.DataFrame:
            tries = 0
            finals_local = pd.DataFrame()
            while tries < 2 and (finals_local is None or finals_local.empty):
                try:
                    sb = scoreboardv2.ScoreboardV2(game_date=game_date, day_offset=0, timeout=45)
                    nd = sb.get_normalized_dict()
                    gh = pd.DataFrame(nd.get("GameHeader", []))
                    ls = pd.DataFrame(nd.get("LineScore", []))
                    if not gh.empty and not ls.empty:
                        cgh = {c.upper(): c for c in gh.columns}
                        cls = {c.upper(): c for c in ls.columns}
                        team_rows = {}
                        for _, r in ls.iterrows():
                            try:
                                tid = int(r[cls["TEAM_ID"]])
                                tri = str(r[cls["TEAM_ABBREVIATION"]]).upper()
                                pts = None
                                if "PTS" in cls:
                                    try:
                                        pts = int(r[cls["PTS"]])
                                    except Exception:
                                        pts = None
                                team_rows[tid] = {"tri": tri, "pts": pts}
                            except Exception:
                                continue
                        out_rows = []
                        for _, g in gh.iterrows():
                            try:
                                hid = int(g[cgh["HOME_TEAM_ID"]]); vid = int(g[cgh["VISITOR_TEAM_ID"]])
                                gid = str(g.get(cgh.get("GAME_ID", "GAME_ID"), "")).strip()
                                h = team_rows.get(hid, {}); v = team_rows.get(vid, {})
                                out_rows.append({
                                    "game_id": gid,
                                    "home_tri": str(h.get("tri") or "").upper(),
                                    "away_tri": str(v.get("tri") or "").upper(),
                                    "home_pts": h.get("pts"),
                                    "visitor_pts": v.get("pts"),
                                })
                            except Exception:
                                continue
                        finals_local = pd.DataFrame(out_rows)
                    break
                except Exception:
                    tries += 1
                    time.sleep(3)
            return finals_local

        if finals is None or finals.empty:
            finals = fetch_finals_for(str(target_date))
        if finals is None or finals.empty:
            # Try +1 day then -1 day to handle timezone/date slippage
            from datetime import timedelta as _td
            finals = fetch_finals_for(str(target_date + _td(days=1)))
            if finals is None or finals.empty:
                finals = fetch_finals_for(str(target_date - _td(days=1)))

        # Preseason often lacks PTS in nba_api; fallback to NBA CDN daily scoreboard
        def fetch_finals_via_nba_cdn(dt) -> pd.DataFrame:
            try:
                import requests as _req
                ymd = pd.to_datetime(dt).strftime('%Y%m%d')
                url = f"https://cdn.nba.com/static/json/liveData/scoreboard/scoreboard_{ymd}.json"
                headers = {
                    'User-Agent': 'Mozilla/5.0',
                    'Accept': 'application/json, text/plain, */*',
                    'Origin': 'https://www.nba.com',
                    'Referer': 'https://www.nba.com/'
                }
                resp = _req.get(url, headers=headers, timeout=30)
                if resp.status_code != 200:
                    return pd.DataFrame()
                js = resp.json()
                games = (js or {}).get('scoreboard', {}).get('games', [])
                rows = []
                for g in games:
                    try:
                        home = g.get('homeTeam', {})
                        away = g.get('awayTeam', {})
                        htri = str(home.get('triCode') or '').upper()
                        atri = str(away.get('triCode') or '').upper()
                        # Scores can be strings; coerce to int if possible
                        def _to_int(x):
                            try:
                                return int(x)
                            except Exception:
                                try:
                                    return int(float(x))
                                except Exception:
                                    return None
                        hpts = _to_int(home.get('score'))
                        apts = _to_int(away.get('score'))
                        rows.append({
                            'home_tri': htri,
                            'away_tri': atri,
                            'home_pts': hpts,
                            'visitor_pts': apts,
                        })
                    except Exception:
                        continue
                return pd.DataFrame(rows)
            except Exception:
                return pd.DataFrame()

        # Use CDN results if nba_api finals missing or all pts are NaN/None
        if finals is None or finals.empty or not (pd.to_numeric(finals.get('home_pts'), errors='coerce').notna().any() and pd.to_numeric(finals.get('visitor_pts'), errors='coerce').notna().any()):
            cdn_df = fetch_finals_via_nba_cdn(target_date)
            if cdn_df is not None and not cdn_df.empty:
                finals = cdn_df

        # If still missing points but we have game_ids from ScoreboardV2, try BoxScoreTraditionalV3 to compute team points
        def fill_pts_via_boxscore(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty or 'game_id' not in df.columns:
                return df
            rows = []
            for _, row in df.iterrows():
                try:
                    gid = str(row.get('game_id') or '').strip()
                    if not gid:
                        rows.append(row)
                        continue
                    bs = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=gid, timeout=45)
                    nd = bs.get_normalized_dict()
                    tstats = pd.DataFrame(nd.get('TeamStats', []))
                    htri = str(row.get('home_tri') or '').upper(); atri = str(row.get('away_tri') or '').upper()
                    hpts = row.get('home_pts'); apts = row.get('visitor_pts')
                    if not tstats.empty:
                        c = {c.upper(): c for c in tstats.columns}
                        for _, tr in tstats.iterrows():
                            tri = str(tr.get(c.get('TEAM_ABBREVIATION','TEAM_ABBREVIATION'), '')).upper()
                            try:
                                pts = int(tr.get(c.get('PTS','PTS')))
                            except Exception:
                                pts = None
                            if tri == htri:
                                hpts = pts
                            elif tri == atri:
                                apts = pts
                    new_row = row.copy()
                    new_row['home_pts'] = hpts
                    new_row['visitor_pts'] = apts
                    rows.append(new_row)
                except Exception:
                    rows.append(row)
            return pd.DataFrame(rows)

        if finals is not None and not finals.empty and not (pd.to_numeric(finals.get('home_pts'), errors='coerce').notna().any() and pd.to_numeric(finals.get('visitor_pts'), errors='coerce').notna().any()):
            finals = fill_pts_via_boxscore(finals)

        # Last resort: derive team totals from recon_props CSV if available (sum of player PTS often equals team score in preseason box files)
        if finals is not None and not finals.empty and (finals['home_pts'].isna().all() or finals['visitor_pts'].isna().all()):
            try:
                # Try processed folder by default
                from .config import paths as _pp
                rppath = _pp.data_processed / f"recon_props_{target_date}.csv"
                if rppath.exists():
                    dfp = pd.read_csv(rppath)
                    # Build {team_abbr: total_pts}
                    pts_by_team = dfp.groupby('team_abbr', dropna=False)['pts'].sum(min_count=1).to_dict()
                    def fill_row(row):
                        htri = str(row.get('home_tri') or '').upper(); atri = str(row.get('away_tri') or '').upper()
                        row = row.copy()
                        if pd.isna(row.get('home_pts')) and htri in pts_by_team:
                            row['home_pts'] = float(pts_by_team[htri])
                        if pd.isna(row.get('visitor_pts')) and atri in pts_by_team:
                            row['visitor_pts'] = float(pts_by_team[atri])
                        return row
                    finals = finals.apply(fill_row, axis=1)
            except Exception:
                pass
    except Exception as e:
        console.print(f"Scoreboard fetch failed: {e}", style="red"); return
    if finals is None or finals.empty:
        console.print("No finals found for date; writing reconciliation with empty finals.", style="yellow")
        merged = preds.copy()
        # Ensure score columns exist with NaN
        merged["home_pts"] = pd.NA
        merged["visitor_pts"] = pd.NA
    else:
        merged = preds.merge(finals, on=["home_tri","away_tri"], how="left")
    # Compute errors
    merged["pred_margin"] = pd.to_numeric(merged.get("pred_margin"), errors="coerce")
    merged["pred_total"] = pd.to_numeric(merged.get("pred_total"), errors="coerce")
    merged["home_pts"] = pd.to_numeric(merged.get("home_pts"), errors="coerce")
    merged["visitor_pts"] = pd.to_numeric(merged.get("visitor_pts"), errors="coerce")
    merged["actual_margin"] = merged["home_pts"] - merged["visitor_pts"]
    merged["total_actual"] = merged[["home_pts","visitor_pts"]].sum(axis=1)
    merged["margin_error"] = merged["pred_margin"] - merged["actual_margin"]
    merged["total_error"] = merged["pred_total"] - merged["total_actual"]
    keep = [
        "date","home_team","visitor_team","home_tri","away_tri",
        "home_pts","visitor_pts","pred_margin","pred_total",
        "actual_margin","total_actual","margin_error","total_error"
    ]
    if "date" not in merged.columns:
        merged["date"] = str(target_date)
    out_df = merged[keep]
    out = _paths.data_processed / f"recon_games_{target_date}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    console.print({"date": str(target_date), "rows": int(len(out_df)), "output": str(out)})


# ============================================================================
# IMPROVEMENT COMMANDS
# ============================================================================

@cli.command()
@click.option("--season", type=int, default=2025, help="NBA season year (e.g., 2025 for 2024-25)")
@click.option("--as-of", "as_of", type=str, default=None, help="Optional cutoff date YYYY-MM-DD (prevents future leakage when using cached boxscores)")
def fetch_advanced_stats(season: int, as_of: str | None):
    """Fetch pace, efficiency, and Four Factors from Basketball Reference."""
    console.rule("Fetch Advanced Stats")
    try:
        from .scrapers import BasketballReferenceScraper
        from .advanced_stats_boxscores import compute_team_advanced_stats_from_boxscores
        from .advanced_stats_player_logs import compute_team_advanced_stats_from_player_logs
        
        scraper = BasketballReferenceScraper()
        
        console.print(f"Fetching team stats for {season} season...")
        stats = scraper.get_team_stats(season)

        # If Basketball Reference returns league-average fallback (or empty), build from cached boxscores.
        try:
            is_empty = stats is None or stats.empty
            is_constant = False
            if not is_empty:
                for c in ("pace", "off_rtg", "def_rtg"):
                    if c in stats.columns and pd.to_numeric(stats[c], errors="coerce").std(skipna=True) <= 1e-9:
                        is_constant = True
            if is_empty or is_constant:
                if is_empty:
                    console.print("Basketball Reference returned no rows; falling back to boxscore-derived stats.", style="yellow")
                else:
                    console.print("Basketball Reference returned league-average fallback (no team variance); falling back to boxscore-derived stats.", style="yellow")

                stats_bs = compute_team_advanced_stats_from_boxscores(season, as_of=as_of)
                if stats_bs is not None and not stats_bs.empty:
                    stats = stats_bs
                    console.print(f"[OK] Built advanced stats from cached boxscores (teams={len(stats)}).", style="green")
                else:
                    console.print("Boxscore-derived advanced stats unavailable; trying player-log-derived stats.", style="yellow")
                    try:
                        stats_pl = compute_team_advanced_stats_from_player_logs(season, as_of=as_of)
                        if stats_pl is not None and not stats_pl.empty:
                            stats = stats_pl
                            console.print(f"[OK] Built advanced stats from cached player logs (teams={len(stats)}).", style="green")
                        else:
                            console.print("Player-log-derived advanced stats unavailable; keeping Basketball Reference results.", style="yellow")
                    except Exception:
                        console.print("Player-log-derived advanced stats failed; keeping Basketball Reference results.", style="yellow")
        except Exception:
            pass

        # Optional augmentation: add-on columns from local cached data.
        # This keeps BR as primary for pace/ratings/four-factors, but enriches the output
        # without requiring any new external sources.
        try:
            opt_cols = ["fg3a_rate", "fg3_pct", "ts_pct", "ast_per_100"]
            local = compute_team_advanced_stats_from_boxscores(season, as_of=as_of)
            if local is None or local.empty:
                local = compute_team_advanced_stats_from_player_logs(season, as_of=as_of)

            if local is not None and not local.empty and "team" in local.columns:
                local = local.copy()
                local["team"] = local["team"].astype(str).str.upper().str.strip()
                local_opt = local[[c for c in (["team"] + opt_cols) if c in local.columns]].copy()

                if stats is not None and not stats.empty and "team" in stats.columns:
                    stats = stats.copy()
                    stats["team"] = stats["team"].astype(str).str.upper().str.strip()
                    stats = stats.merge(local_opt, on="team", how="left", suffixes=("", "_local"))
                    for c in opt_cols:
                        lc = f"{c}_local"
                        if lc in stats.columns:
                            if c not in stats.columns:
                                stats[c] = stats[lc]
                            else:
                                stats[c] = stats[c].combine_first(stats[lc])
                            stats = stats.drop(columns=[lc])
                    console.print("[OK] Augmented advanced stats with local add-ons (3P/TS/AST).", style="green")
        except Exception:
            pass
        
        if stats.empty:
            console.print("No stats fetched", style="yellow")
            return
        
        # Save to processed directory
        if as_of:
            safe = str(as_of).strip().replace(":", "-")
            output_path = paths.data_processed / f"team_advanced_stats_{season}_asof_{safe}.csv"
        else:
            output_path = paths.data_processed / f"team_advanced_stats_{season}.csv"
        stats.to_csv(output_path, index=False)
        
        console.print(f"[OK] Saved {len(stats)} teams to {output_path}", style="green")
        console.print(stats.head(10))
        
    except Exception as e:
        console.print(f"Error fetching advanced stats: {e}", style="red")


@cli.command()
@click.option("--date", "date_str", type=str, default=None, help="Target date YYYY-MM-DD (uses NBA official report when available).")
def fetch_injuries(date_str: str | None):
    """Fetch current injury reports (prefers NBA official)."""
    console.rule("Fetch Injury Reports")
    try:
        from .scrapers import NBAInjuryDatabase
        
        db = NBAInjuryDatabase()
        if date_str:
            console.print(f"Fetching injury reports for {date_str} (NBA official preferred)...")
        else:
            console.print("Fetching injury reports (NBA official preferred)...")

        injuries = db.update_injuries(date_str=date_str)
        
        if injuries.empty:
            console.print("No injuries fetched", style="yellow")
            return
        
        console.print(f"[OK] Saved {len(injuries)} injury records", style="green")
        
        # Show summary by team
        summary = injuries.groupby(['team', 'status']).size().reset_index(name='count')
        console.print("\nInjury Summary:")
        console.print(summary)
        
    except Exception as e:
        console.print(f"Error fetching injuries: {e}", style="red")


@cli.command()
@click.option("--days", type=int, default=30, help="Number of days to analyze")
def performance_report(days: int):
    """Generate performance report for model predictions."""
    console.rule(f"Performance Report (Last {days} Days)")
    try:
        from .performance import PerformanceTracker
        
        tracker = PerformanceTracker()
        report = tracker.generate_performance_report(days_back=days)
        
        tracker.print_performance_summary(report)
        
    except Exception as e:
        console.print(f"Error generating performance report: {e}", style="red")


@cli.command()
@click.option("--confidence", type=float, default=0.55, help="Minimum confidence threshold (0.5-1.0)")
@click.option("--days", type=int, default=30, help="Number of days to analyze")
def calculate_roi(confidence: float, days: int):
    """Calculate ROI for betting strategy."""
    console.rule(f"ROI Analysis (Last {days} Days, Confidence ≥ {confidence})")
    try:
        from .performance import PerformanceTracker
        
        tracker = PerformanceTracker()
        df = tracker.load_predictions_and_results()
        
        if df.empty:
            console.print("No data available for ROI calculation", style="yellow")
            return
        
        # Calculate ROI for different bet types
        for bet_type in ['moneyline']:
            console.print(f"\n{bet_type.upper()} ROI:")
            roi = tracker.calculate_roi(df, bet_type=bet_type, confidence_threshold=confidence)
            
            console.print(f"  Total Bets: {roi['total_bets']}")
            console.print(f"  Win Rate: {roi['win_rate']:.1f}%")
            console.print(f"  Total Profit: ${roi['total_profit']:.2f}")
            console.print(f"  ROI: {roi['roi']:.1f}%", style="green" if roi['roi'] > 0 else "red")
        
    except Exception as e:
        console.print(f"Error calculating ROI: {e}", style="red")


@cli.command()
def run_all_improvements():
    """Run all improvement tasks: fetch stats, injuries, and generate performance report."""
    console.rule("[NPU] Running All Improvements")
    
    # 1. Fetch advanced stats
    console.print("\n[1/3] Fetching advanced statistics...", style="cyan")
    try:
        from .scrapers import BasketballReferenceScraper
        scraper = BasketballReferenceScraper()
        stats = scraper.get_team_stats(2025)
        if not stats.empty:
            output_path = paths.data_processed / "team_advanced_stats_2025.csv"
            stats.to_csv(output_path, index=False)
            console.print(f"[OK] Saved {len(stats)} teams to {output_path}", style="green")
        else:
            console.print("⚠️  No stats fetched", style="yellow")
    except Exception as e:
        console.print(f"[ERROR] Error fetching stats: {e}", style="red")
    
    # 2. Fetch injury data
    console.print("\n[2/3] Fetching injury reports...", style="cyan")
    try:
        from .scrapers import NBAInjuryDatabase
        db = NBAInjuryDatabase()
        injuries = db.update_injuries()
        if not injuries.empty:
            console.print(f"[OK] Saved {len(injuries)} injury records", style="green")
            summary = injuries.groupby(['team', 'status']).size().reset_index(name='count')
            console.print(summary)
        else:
            console.print("⚠️  No injuries fetched", style="yellow")
    except Exception as e:
        console.print(f"[ERROR] Error fetching injuries: {e}", style="red")
    
    # 3. Generate performance report
    console.print("\n[3/3] Generating performance report...", style="cyan")
    try:
        from .performance import PerformanceTracker
        tracker = PerformanceTracker()
        report = tracker.generate_performance_report(days_back=30)
        tracker.print_performance_summary(report)
    except Exception as e:
        console.print(f"[ERROR] Error generating report: {e}", style="red")
    
    console.print("\n[OK] All improvements completed!", style="green bold")


@cli.command("backtest-period-calibration")
@click.option("--start", "start_date", type=str, required=True, help="Start date YYYY-MM-DD")
@click.option("--end", "end_date", type=str, required=True, help="End date YYYY-MM-DD")
@click.option("--weights", type=str, default="0.0,0.3,0.6,0.8,1.0", show_default=True, help="Comma-separated blend weights to test (0=model only, 1=shares only)")
def backtest_period_calibration_cmd(start_date: str, end_date: str, weights: str):
    """Backtest quarter/half calibration over a date range and report MAE for totals and margins.

    For each date:
    - Generates predictions with periods using NPU models (uncalibrated first)
    - Applies calibration across a list of blend weights without re-running models
    - Compares to actual line scores from data/raw/games_nba_api.csv
    - Reports aggregate MAE per weight for: Q1..Q4 totals, H1/H2 totals, game total, and quarter margins
    """
    console.rule("Backtest Period Calibration")
    try:
        s = pd.to_datetime(start_date).date(); e = pd.to_datetime(end_date).date()
    except Exception:
        console.print("Invalid --start/--end (YYYY-MM-DD)", style="red"); return
    if e < s:
        console.print("--end must be >= --start", style="red"); return
    # Parse weight list
    try:
        w_list = [float(x.strip()) for x in weights.split(',') if x.strip()]
        w_list = [w for w in w_list if 0.0 <= w <= 1.0]
        assert len(w_list) > 0
    except Exception:
        console.print("--weights must be comma-separated floats in [0,1]", style="red"); return

    # Load raw games for actuals
    raw_csv = paths.data_raw / "games_nba_api.csv"
    if not raw_csv.exists():
        console.print(f"Raw games not found: {raw_csv}. Run fetch first.", style="red"); return
    raw = pd.read_csv(raw_csv)
    # Normalize columns
    rename_map = {}
    if 'date_est' in raw.columns:
        rename_map['date_est'] = 'date'
    if 'home_team_tri' in raw.columns:
        rename_map.update({'home_team_tri': 'home_team', 'visitor_team_tri': 'visitor_team'})
    if 'home_score' in raw.columns:
        rename_map['home_score'] = 'home_pts'
    if 'visitor_score' in raw.columns:
        rename_map['visitor_score'] = 'visitor_pts'
    if rename_map:
        raw = raw.rename(columns=rename_map)
    # Restrict to date range and required cols
    need_cols = [
        'date','home_team','visitor_team','home_pts','visitor_pts',
        'home_q1','home_q2','home_q3','home_q4','visitor_q1','visitor_q2','visitor_q3','visitor_q4'
    ]
    for c in need_cols:
        if c not in raw.columns:
            console.print(f"Raw games missing column: {c}", style="red"); return
    raw['date'] = pd.to_datetime(raw['date'], errors='coerce').dt.date
    raw = raw[(raw['date'] >= s) & (raw['date'] <= e)].copy()
    if raw.empty:
        console.print("No raw games in date range", style="yellow"); return

    # Normalize team names to tricodes for robust merging
    try:
        from .teams import to_tricode as _to_tri
        raw['home_tri'] = raw['home_team'].astype(str).map(lambda x: _to_tri(x))
        raw['visitor_tri'] = raw['visitor_team'].astype(str).map(lambda x: _to_tri(x))
    except Exception:
        raw['home_tri'] = raw['home_team'].astype(str).str.upper()
        raw['visitor_tri'] = raw['visitor_team'].astype(str).str.upper()

    # Build date list
    dates = sorted(raw['date'].unique())

    # Accumulators per weight
    metrics = {w: {
        'sum_abs_q_total': 0.0,
        'cnt_q_total': 0,
        'sum_abs_h_total': 0.0,
        'cnt_h_total': 0,
        'sum_abs_game_total': 0.0,
        'cnt_game_total': 0,
        'sum_abs_q_margin': 0.0,
        'cnt_q_margin': 0,
    } for w in w_list}

    from .games_npu import predict_games_npu
    from .period_calibration import calibrate_period_predictions, CalibrationConfig, load_or_build_team_period_shares

    # Baseline aggregators
    metrics_equal = {
        'sum_abs_q_total': 0.0,
        'cnt_q_total': 0,
        'sum_abs_h_total': 0.0,
        'cnt_h_total': 0,
        'sum_abs_game_total': 0.0,
        'cnt_game_total': 0,
        'sum_abs_q_margin': 0.0,
        'cnt_q_margin': 0,
    }
    metrics_league = {k: 0 if isinstance(v, int) else 0.0 for k, v in metrics_equal.items()}
    # Load league shares once for league-only baseline
    try:
        _team_shares_df, _league_share_vec = load_or_build_team_period_shares(force_recompute=False)
    except Exception:
        _team_shares_df, _league_share_vec = None, np.array([0.25,0.25,0.25,0.25], dtype=float)

    for d in track(dates, description="Backtesting", total=len(dates)):
        ds = pd.Timestamp(d).strftime('%Y-%m-%d')
        # Build features for this date using the same path as predict-games-npu
        features_path = paths.data_processed / "features.parquet"
        csv_fallback = paths.data_processed / "features.csv"
        features_df = None
        try:
            if features_path.exists():
                df_fe = pd.read_parquet(features_path)
                df_fe['date'] = pd.to_datetime(df_fe['date']).dt.date
                features_df = df_fe[df_fe['date'] == d]
            elif csv_fallback.exists():
                df_fe = pd.read_csv(csv_fallback)
                if 'date' in df_fe.columns:
                    df_fe['date'] = pd.to_datetime(df_fe['date']).dt.date
                    features_df = df_fe[df_fe['date'] == d]
        except Exception:
            features_df = None
        if features_df is None or features_df.empty:
            # Fallback: construct slate directly from raw for this date
            try:
                day_raw = raw[raw['date'] == d][['date','home_team','visitor_team']].dropna()
                if not day_raw.empty:
                    slate_df = day_raw.copy()
                    slate_df['home_pts'] = np.nan
                    slate_df['visitor_pts'] = np.nan
                    # Combine with raw history to let feature builder compute form/adv stats
                    games = pd.concat([raw.copy(), slate_df], ignore_index=True, sort=False)
                    games['date'] = pd.to_datetime(games['date'])
                    from .features_enhanced import build_features_enhanced
                    feats2 = build_features_enhanced(games, include_advanced_stats=True, include_injuries=True, season=2025)
                    feats2['date'] = pd.to_datetime(feats2['date']).dt.date
                    features_df = feats2[feats2['date'] == d]
            except Exception:
                pass
        if features_df is None or features_df.empty:
            continue

        # Predict once (uncalibrated) to get period columns
        try:
            base_preds = predict_games_npu(features_df, include_periods=True, calibrate_periods=False)
        except Exception:
            continue

        # Join with actuals for that date
        act = raw[raw['date'] == d].copy()
        # Compute actual totals and margins per quarter and halves
        act["q1_total"] = act["home_q1"] + act["visitor_q1"]
        act["q2_total"] = act["home_q2"] + act["visitor_q2"]
        act["q3_total"] = act["home_q3"] + act["visitor_q3"]
        act["q4_total"] = act["home_q4"] + act["visitor_q4"]
        act["h1_total"] = act["q1_total"] + act["q2_total"]
        act["h2_total"] = act["q3_total"] + act["q4_total"]
        act["game_total"] = act["home_pts"] + act["visitor_pts"]
        act["q1_margin"] = act["home_q1"] - act["visitor_q1"]
        act["q2_margin"] = act["home_q2"] - act["visitor_q2"]
        act["q3_margin"] = act["home_q3"] - act["visitor_q3"]
        act["q4_margin"] = act["home_q4"] - act["visitor_q4"]

        # Merge keys (tricodes preferred)
        try:
            from .teams import to_tricode as _to_tri
            base_preds['home_tri'] = base_preds['home_team'].astype(str).map(lambda x: _to_tri(x))
            base_preds['visitor_tri'] = base_preds['visitor_team'].astype(str).map(lambda x: _to_tri(x))
            act['home_tri'] = act['home_team'].astype(str).map(lambda x: _to_tri(x))
            act['visitor_tri'] = act['visitor_team'].astype(str).map(lambda x: _to_tri(x))
        except Exception:
            base_preds['home_tri'] = base_preds['home_team'].astype(str).str.upper()
            base_preds['visitor_tri'] = base_preds['visitor_team'].astype(str).str.upper()
            act['home_tri'] = act['home_team'].astype(str).str.upper()
            act['visitor_tri'] = act['visitor_team'].astype(str).str.upper()
        merged_base = pd.merge(
            base_preds,
            act[[
                'home_tri','visitor_tri','q1_total','q2_total','q3_total','q4_total',
                'h1_total','h2_total','game_total','q1_margin','q2_margin','q3_margin','q4_margin']
            ],
            left_on=['home_tri','visitor_tri'], right_on=['home_tri','visitor_tri'], how='inner'
        )
        if merged_base.empty:
            continue

        # Evaluate each weight by applying calibration
        import numpy as _np
        for w in w_list:
            cfg = CalibrationConfig(totals_blend_weight=w)
            try:
                preds = calibrate_period_predictions(merged_base.copy(), cfg)
            except Exception:
                continue
            # Collect absolute errors with NaN-safe handling and proper counts
            # Quarter totals (stack all quarters)
            q_abs = []
            for i in (1,2,3,4):
                diff = _np.abs(preds.get(f"quarters_q{i}_total", _np.nan) - preds.get(f"q{i}_total", _np.nan))
                q_abs.append(diff.values)
            q_abs = _np.concatenate(q_abs, axis=0)
            if q_abs.size:
                q_mask = _np.isfinite(q_abs)
                metrics[w]['sum_abs_q_total'] += float(_np.nansum(q_abs[q_mask]))
                metrics[w]['cnt_q_total'] += int(q_mask.sum())

            # Halves totals
            h1 = _np.abs(preds.get("halves_h1_total", _np.nan) - preds.get("h1_total", _np.nan)).values
            h2 = _np.abs(preds.get("halves_h2_total", _np.nan) - preds.get("h2_total", _np.nan)).values
            for h_arr in (h1, h2):
                h_mask = _np.isfinite(h_arr)
                metrics[w]['sum_abs_h_total'] += float(_np.nansum(h_arr[h_mask]))
                metrics[w]['cnt_h_total'] += int(h_mask.sum())

            # Game totals (prefer 'totals' else 'pred_total')
            g_col = 'totals' if 'totals' in preds.columns else ('pred_total' if 'pred_total' in preds.columns else None)
            if g_col is not None:
                g_abs = _np.abs(preds[g_col].values - preds["game_total"].values)
                g_mask = _np.isfinite(g_abs)
                metrics[w]['sum_abs_game_total'] += float(_np.nansum(g_abs[g_mask]))
                metrics[w]['cnt_game_total'] += int(g_mask.sum())

            # Quarter margins
            qm_abs = []
            for i in (1,2,3,4):
                diff = _np.abs(preds.get(f"quarters_q{i}_margin", _np.nan) - preds.get(f"q{i}_margin", _np.nan))
                qm_abs.append(diff.values)
            qm_abs = _np.concatenate(qm_abs, axis=0)
            if qm_abs.size:
                qm_mask = _np.isfinite(qm_abs)
                metrics[w]['sum_abs_q_margin'] += float(_np.nansum(qm_abs[qm_mask]))
                metrics[w]['cnt_q_margin'] += int(qm_mask.sum())

        # Baseline A: Equal split of predicted game total (and uniform margin)
        try:
            preds_eq = merged_base.copy()
            g_col = 'totals' if 'totals' in preds_eq.columns else ('pred_total' if 'pred_total' in preds_eq.columns else None)
            if g_col is not None:
                g = preds_eq[g_col].astype(float)
                sp = preds_eq.get('spread_margin', 0.0).astype(float)
                for i in (1,2,3,4):
                    preds_eq[f'quarters_q{i}_total'] = g / 4.0
                    preds_eq[f'quarters_q{i}_margin'] = sp / 4.0
                preds_eq['halves_h1_total'] = preds_eq['quarters_q1_total'] + preds_eq['quarters_q2_total']
                preds_eq['halves_h2_total'] = preds_eq['quarters_q3_total'] + preds_eq['quarters_q4_total']
                preds_eq['halves_h1_margin'] = preds_eq['quarters_q1_margin'] + preds_eq['quarters_q2_margin']
                preds_eq['halves_h2_margin'] = preds_eq['quarters_q3_margin'] + preds_eq['quarters_q4_margin']
                # Accumulate
                q_abs = []
                for i in (1,2,3,4):
                    diff = _np.abs(preds_eq.get(f"quarters_q{i}_total", _np.nan) - preds_eq.get(f"q{i}_total", _np.nan))
                    q_abs.append(diff.values)
                q_abs = _np.concatenate(q_abs, axis=0)
                if q_abs.size:
                    q_mask = _np.isfinite(q_abs)
                    metrics_equal['sum_abs_q_total'] += float(_np.nansum(q_abs[q_mask]))
                    metrics_equal['cnt_q_total'] += int(q_mask.sum())
                for h_pred, h_act in (("halves_h1_total","h1_total"),("halves_h2_total","h2_total")):
                    arr = _np.abs(preds_eq.get(h_pred, _np.nan) - preds_eq.get(h_act, _np.nan)).values
                    m = _np.isfinite(arr)
                    metrics_equal['sum_abs_h_total'] += float(_np.nansum(arr[m]))
                    metrics_equal['cnt_h_total'] += int(m.sum())
                if g_col is not None and 'game_total' in preds_eq.columns:
                    g_abs = _np.abs(preds_eq[g_col].values - preds_eq['game_total'].values)
                    g_mask = _np.isfinite(g_abs)
                    metrics_equal['sum_abs_game_total'] += float(_np.nansum(g_abs[g_mask]))
                    metrics_equal['cnt_game_total'] += int(g_mask.sum())
                qm_abs = []
                for i in (1,2,3,4):
                    diff = _np.abs(preds_eq.get(f"quarters_q{i}_margin", _np.nan) - preds_eq.get(f"q{i}_margin", _np.nan))
                    qm_abs.append(diff.values)
                qm_abs = _np.concatenate(qm_abs, axis=0)
                if qm_abs.size:
                    qm_mask = _np.isfinite(qm_abs)
                    metrics_equal['sum_abs_q_margin'] += float(_np.nansum(qm_abs[qm_mask]))
                    metrics_equal['cnt_q_margin'] += int(qm_mask.sum())
        except Exception:
            pass

        # Baseline B: League-average shares split (and uniform margin)
        try:
            preds_league = merged_base.copy()
            g_col = 'totals' if 'totals' in preds_league.columns else ('pred_total' if 'pred_total' in preds_league.columns else None)
            if g_col is not None:
                g = preds_league[g_col].astype(float)
                sp = preds_league.get('spread_margin', 0.0).astype(float)
                # Apply constant league shares to every row
                for i, share in enumerate(_league_share_vec, start=1):
                    preds_league[f'quarters_q{i}_total'] = g * float(share)
                    preds_league[f'quarters_q{i}_margin'] = sp / 4.0
                preds_league['halves_h1_total'] = preds_league['quarters_q1_total'] + preds_league['quarters_q2_total']
                preds_league['halves_h2_total'] = preds_league['quarters_q3_total'] + preds_league['quarters_q4_total']
                preds_league['halves_h1_margin'] = preds_league['quarters_q1_margin'] + preds_league['quarters_q2_margin']
                preds_league['halves_h2_margin'] = preds_league['quarters_q3_margin'] + preds_league['quarters_q4_margin']
                # Accumulate
                q_abs = []
                for i in (1,2,3,4):
                    diff = _np.abs(preds_league.get(f"quarters_q{i}_total", _np.nan) - preds_league.get(f"q{i}_total", _np.nan))
                    q_abs.append(diff.values)
                q_abs = _np.concatenate(q_abs, axis=0)
                if q_abs.size:
                    q_mask = _np.isfinite(q_abs)
                    metrics_league['sum_abs_q_total'] += float(_np.nansum(q_abs[q_mask]))
                    metrics_league['cnt_q_total'] += int(q_mask.sum())
                for h_pred, h_act in (("halves_h1_total","h1_total"),("halves_h2_total","h2_total")):
                    arr = _np.abs(preds_league.get(h_pred, _np.nan) - preds_league.get(h_act, _np.nan)).values
                    m = _np.isfinite(arr)
                    metrics_league['sum_abs_h_total'] += float(_np.nansum(arr[m]))
                    metrics_league['cnt_h_total'] += int(m.sum())
                if g_col is not None and 'game_total' in preds_league.columns:
                    g_abs = _np.abs(preds_league[g_col].values - preds_league['game_total'].values)
                    g_mask = _np.isfinite(g_abs)
                    metrics_league['sum_abs_game_total'] += float(_np.nansum(g_abs[g_mask]))
                    metrics_league['cnt_game_total'] += int(g_mask.sum())
                qm_abs = []
                for i in (1,2,3,4):
                    diff = _np.abs(preds_league.get(f"quarters_q{i}_margin", _np.nan) - preds_league.get(f"q{i}_margin", _np.nan))
                    qm_abs.append(diff.values)
                qm_abs = _np.concatenate(qm_abs, axis=0)
                if qm_abs.size:
                    qm_mask = _np.isfinite(qm_abs)
                    metrics_league['sum_abs_q_margin'] += float(_np.nansum(qm_abs[qm_mask]))
                    metrics_league['cnt_q_margin'] += int(qm_mask.sum())
        except Exception:
            pass

    # Summarize
    rows = []
    for w in w_list:
        rows.append({
            'method': 'calibrated',
            'weight': w,
            'games': metrics[w]['cnt_game_total'],
            'MAE_quarters_total': round((metrics[w]['sum_abs_q_total'] / metrics[w]['cnt_q_total']) if metrics[w]['cnt_q_total'] else float('nan'), 3),
            'MAE_halves_total': round((metrics[w]['sum_abs_h_total'] / metrics[w]['cnt_h_total']) if metrics[w]['cnt_h_total'] else float('nan'), 3),
            'MAE_game_total': round((metrics[w]['sum_abs_game_total'] / metrics[w]['cnt_game_total']) if metrics[w]['cnt_game_total'] else float('nan'), 3),
            'MAE_quarters_margin': round((metrics[w]['sum_abs_q_margin'] / metrics[w]['cnt_q_margin']) if metrics[w]['cnt_q_margin'] else float('nan'), 3),
        })
    # Append baselines
    rows.append({
        'method': 'baseline_equal',
        'weight': None,
        'games': metrics_equal['cnt_game_total'],
        'MAE_quarters_total': round((metrics_equal['sum_abs_q_total'] / metrics_equal['cnt_q_total']) if metrics_equal['cnt_q_total'] else float('nan'), 3),
        'MAE_halves_total': round((metrics_equal['sum_abs_h_total'] / metrics_equal['cnt_h_total']) if metrics_equal['cnt_h_total'] else float('nan'), 3),
        'MAE_game_total': round((metrics_equal['sum_abs_game_total'] / metrics_equal['cnt_game_total']) if metrics_equal['cnt_game_total'] else float('nan'), 3),
        'MAE_quarters_margin': round((metrics_equal['sum_abs_q_margin'] / metrics_equal['cnt_q_margin']) if metrics_equal['cnt_q_margin'] else float('nan'), 3),
    })
    rows.append({
        'method': 'baseline_league',
        'weight': None,
        'games': metrics_league['cnt_game_total'],
        'MAE_quarters_total': round((metrics_league['sum_abs_q_total'] / metrics_league['cnt_q_total']) if metrics_league['cnt_q_total'] else float('nan'), 3),
        'MAE_halves_total': round((metrics_league['sum_abs_h_total'] / metrics_league['cnt_h_total']) if metrics_league['cnt_h_total'] else float('nan'), 3),
        'MAE_game_total': round((metrics_league['sum_abs_game_total'] / metrics_league['cnt_game_total']) if metrics_league['cnt_game_total'] else float('nan'), 3),
        'MAE_quarters_margin': round((metrics_league['sum_abs_q_margin'] / metrics_league['cnt_q_margin']) if metrics_league['cnt_q_margin'] else float('nan'), 3),
    })
    out_df = pd.DataFrame(rows).sort_values('MAE_quarters_total')
    console.print(out_df)
    # Save optional CSV summary
    try:
        p = paths.data_processed / f"backtest_period_calibration_{start_date}_to_{end_date}.csv"
        out_df.to_csv(p, index=False)
        console.print({"summary": str(p)})
    except Exception:
        pass


if __name__ == "__main__":
    cli()
