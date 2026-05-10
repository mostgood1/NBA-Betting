#!/usr/bin/env python3
"""Daily Live Lens audit report: JSONL signals -> scored rows + markdown summary.

Reads (for a given date):
- data/processed/live_lens_signals_<date>.jsonl
- data/processed/recon_games_<date>.csv (game totals)
- data/processed/recon_quarters_<date>.csv (half/quarter totals)
- data/processed/recon_props_<date>.csv (player props)

Writes:
- data/processed/reports/live_lens_audit_<date>.md
- data/processed/reports/live_lens_scored_<date>.csv

This is an *audit* loop (did our live signals line up with outcomes?).
It does not claim EV; it reports predictive errors and directional hits.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import date as _date
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path((os.getenv("NBA_BETTING_DATA_ROOT") or "").strip()).expanduser() if (os.getenv("NBA_BETTING_DATA_ROOT") or "").strip() else (ROOT / "data")
PROCESSED = DATA_ROOT / "processed"
REPORTS = PROCESSED / "reports"
LIVE_LENS_DIR = Path((os.getenv("NBA_LIVE_LENS_DIR") or os.getenv("LIVE_LENS_DIR") or "").strip() or str(PROCESSED))


def _parse_date(s: str) -> _date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _iso(d: _date) -> str:
    return d.isoformat()


def _n(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _canon_nba_game_id(game_id: Any) -> str:
    try:
        raw = str(game_id or "").strip()
    except Exception:
        return ""
    digits = "".join([c for c in raw if c.isdigit()])
    if len(digits) == 8:
        return "00" + digits
    if len(digits) == 9:
        return "0" + digits
    return digits


def _is_canon_gid(gid: str | None) -> bool:
    g = str(gid or "").strip()
    return len(g) == 10 and g.isdigit()


def _load_gid_map(ds: str) -> dict[tuple[str, str], str]:
    """Build (home_tri, away_tri) -> canonical gid map.

    Prefer game_cards_<date>.csv because recon_games_<date>.csv typically does not
    contain NBA game_id.
    """
    path = PROCESSED / f"game_cards_{ds}.csv"
    df = _load_csv(path)
    if df is None or df.empty:
        df = pd.DataFrame()
    if "game_id" not in df.columns:
        df = pd.DataFrame()
    if "home_tri" not in df.columns or "away_tri" not in df.columns:
        df = pd.DataFrame()

    out: dict[tuple[str, str], str] = {}
    if not df.empty:
        for _, r in df.iterrows():
            home = _safe_upper(r.get("home_tri"))
            away = _safe_upper(r.get("away_tri"))
            gid = _canon_nba_game_id(r.get("game_id"))
            if home and away and _is_canon_gid(gid):
                out[(home, away)] = gid
    if out:
        return out

    # Fallback: schedule JSON contains tricodes + canonical NBA game_id.
    sched_path = PROCESSED / "schedule_2025_26.json"
    if not sched_path.exists():
        return {}
    try:
        raw = sched_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(data, list):
        return {}

    for g in data:
        if not isinstance(g, dict):
            continue
        date_est = str(g.get("date_est") or "")
        if date_est[:10] != ds:
            continue
        home = _safe_upper(g.get("home_tricode"))
        away = _safe_upper(g.get("away_tricode"))
        gid = _canon_nba_game_id(g.get("game_id"))
        if home and away and _is_canon_gid(gid):
            out[(home, away)] = gid
    return out


def _resolve_gid_for_props(obj: dict[str, Any], gid_map: dict[tuple[str, str], str]) -> str | None:
    # Prefer numeric NBA game id if present.
    gid0 = _canon_nba_game_id(obj.get("game_id_canon") or obj.get("game_id"))
    if _is_canon_gid(gid0):
        return gid0

    home = _safe_upper(obj.get("home"))
    away = _safe_upper(obj.get("away"))
    if home and away:
        gid = gid_map.get((home, away))
        if gid:
            return gid

    # Support simple team-based ids like "BKN@ATL".
    raw = str(obj.get("game_id") or "").strip().upper()
    m = re.match(r"^([A-Z]{3})\s*@\s*([A-Z]{3})$", raw)
    if m:
        away2, home2 = m.group(1), m.group(2)
        gid = gid_map.get((home2, away2))
        if gid:
            return gid

    return None


def _resolve_gid_for_game(obj: dict[str, Any], gid_map: dict[tuple[str, str], str]) -> str | None:
    """Resolve a canonical NBA gid for non-prop markets.

    Live Lens logs sometimes use matchup ids like "WAS@ATL"; recon quarters/props
    are keyed by the canonical 10-digit NBA game_id.
    """
    gid = _resolve_gid_for_props(obj, gid_map)
    return gid


def _norm_player_name(s: str) -> str:
    if s is None:
        return ""
    t = str(s)
    if "(" in t:
        t = t.split("(", 1)[0]
    t = t.replace("-", " ")
    t = t.replace(".", "").replace("'", "").replace(",", " ").strip()
    for suf in [" JR", " SR", " II", " III", " IV"]:
        if t.upper().endswith(suf):
            t = t[: -len(suf)]
    try:
        import unicodedata as _ud

        t = _ud.normalize("NFKD", t)
        t = t.encode("ascii", "ignore").decode("ascii")
    except Exception:
        pass
    return t.upper().strip()


def _live_stat_key(x: Any) -> str:
    s = str(x or "").strip().lower()
    m = {
        "points": "pts",
        "point": "pts",
        "pts": "pts",
        "rebounds": "reb",
        "rebound": "reb",
        "reb": "reb",
        "assists": "ast",
        "assist": "ast",
        "ast": "ast",
        "3pt": "threes",
        "3pm": "threes",
        "threes": "threes",
        "threes_made": "threes",
        "steals": "stl",
        "steal": "stl",
        "stl": "stl",
        "blocks": "blk",
        "block": "blk",
        "blk": "blk",
        "turnovers": "tov",
        "turnover": "tov",
        "tov": "tov",
        "pra": "pra",
        "points+rebounds+assists": "pra",
    }
    return m.get(s, s)


def _safe_upper(x: Any) -> str | None:
    try:
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        s = str(x).strip().upper()
        if not s or s in {"NAN", "NONE", "NULL"}:
            return None
        return s
    except Exception:
        return None


@dataclass(frozen=True)
class ScoredRow:
    date: str
    market: str
    horizon: str | None
    signal_key: str | None
    game_id: str | None
    home: str | None
    away: str | None
    team_tri: str | None
    player: str | None
    name_key: str | None
    stat: str | None
    stat_key: str | None
    side: str | None
    klass: str | None
    elapsed: float | None
    live_line: float | None
    edge: float | None
    pred: float | None
    act: float | None
    result: str | None
    err: float | None
    interval_drift_on: int | None
    recent_window_on: int | None
    endgame_foul_on: int | None


def _load_signals(ds: str) -> list[dict[str, Any]]:
    fp = LIVE_LENS_DIR / f"live_lens_signals_{ds}.jsonl"
    if not fp.exists():
        return []
    out: list[dict[str, Any]] = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _prep_recon_games(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "game_id" in out.columns:
        out["_gid"] = out["game_id"].map(_canon_nba_game_id)
    else:
        out["_gid"] = ""
    if "home_tri" in out.columns:
        out["home_tri"] = out["home_tri"].astype(str).str.strip().str.upper()
    if "away_tri" in out.columns:
        out["away_tri"] = out["away_tri"].astype(str).str.strip().str.upper()
    return out


def _prep_recon_quarters(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "game_id" in out.columns:
        out["_gid"] = out["game_id"].map(_canon_nba_game_id)
    else:
        out["_gid"] = ""
    if "home_tri" in out.columns:
        out["home_tri"] = out["home_tri"].astype(str).str.strip().str.upper()
    if "away_tri" in out.columns:
        out["away_tri"] = out["away_tri"].astype(str).str.strip().str.upper()
    return out


def _prep_recon_props(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "game_id" in out.columns:
        out["_gid"] = out["game_id"].map(_canon_nba_game_id)
    else:
        out["_gid"] = ""
    if "player_name" in out.columns:
        out["_name_key"] = out["player_name"].astype(str).map(_norm_player_name)
    else:
        out["_name_key"] = ""
    for c in ("pts", "reb", "ast", "threes", "stl", "blk", "tov", "pra"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _actual_total(market: str, horizon: str | None, gid: str | None, home: str | None, away: str | None, rg: pd.DataFrame, rq: pd.DataFrame) -> float | None:
    if market == "total":
        if rg.empty:
            return None
        if gid:
            hit = rg[rg.get("_gid") == gid]
            if not hit.empty:
                return _n(hit.iloc[0].get("total_actual"))
        if home and away:
            hit = rg[(rg.get("home_tri") == home) & (rg.get("away_tri") == away)]
            if not hit.empty:
                return _n(hit.iloc[0].get("total_actual"))
        return None

    if market in {"half_total", "quarter_total"}:
        if rq.empty:
            return None
        hit = pd.DataFrame()
        if gid and "_gid" in rq.columns:
            hit = rq[rq.get("_gid") == gid]
        if hit.empty and home and away:
            hit = rq[(rq.get("home_tri") == home) & (rq.get("away_tri") == away)]
        if hit.empty:
            return None
        if hit.empty:
            return None

        row = hit.iloc[0]
        if market == "half_total":
            if horizon == "h1":
                return _n(row.get("actual_h1_total"))
            if horizon == "h2":
                return _n(row.get("actual_h2_total"))
            return None

        if market == "quarter_total":
            hz = horizon or ""
            if hz in {"q1", "q2", "q3", "q4"}:
                return _n(row.get(f"actual_{hz}_total"))
            return None

    return None


def _result_for_side(side: str | None, act: float | None, line: float | None) -> str | None:
    if act is None or line is None:
        return None
    s = (side or "").strip().lower()
    if act == line:
        return "push"
    if s == "over":
        return "win" if act > line else "loss"
    if s == "under":
        return "win" if act < line else "loss"
    return None


def _dedup_first_bets(df: pd.DataFrame) -> pd.DataFrame:
        """De-dup repeated tick signals by treating the first BET as the graded decision.

        Live Lens can emit the same BET signal repeatedly across ticks. For accuracy
        reporting we want to grade the *first* actionable BET for a given decision.

        Decision keys:
            - player_prop: (market, horizon, game, player, stat, side)
            - totals/quarters/halves/ats: (market, horizon, game, side)

        Notes:
            - We intentionally ignore live_line/line so later line updates do not
                create new decisions.
            - Ordering uses elapsed (ascending) then original row order.
        """
        if df is None or df.empty:
                return pd.DataFrame() if df is None else df
        if "klass" not in df.columns:
                return pd.DataFrame()

        d0 = df.copy()
        d0["_idx"] = range(len(d0))
        d0["_elapsed"] = pd.to_numeric(d0.get("elapsed"), errors="coerce")
        klass = d0["klass"].astype(str).str.upper()
        bets = d0[klass == "BET"].copy()
        if bets.empty:
                return bets

        m = bets.get("market", pd.Series([""] * len(bets), index=bets.index)).astype(str).str.strip().str.lower()
        horizon = bets.get("horizon", pd.Series([""] * len(bets), index=bets.index)).astype(str).fillna("").str.strip().str.lower()
        gid = bets.get("game_id", pd.Series([""] * len(bets), index=bets.index)).astype(str).fillna("").str.strip()
        home = bets.get("home", pd.Series([""] * len(bets), index=bets.index)).astype(str).fillna("").str.strip().str.upper()
        away = bets.get("away", pd.Series([""] * len(bets), index=bets.index)).astype(str).fillna("").str.strip().str.upper()
        side = bets.get("side", pd.Series([""] * len(bets), index=bets.index)).astype(str).fillna("").str.strip().str.lower()
        name_key = bets.get("name_key", pd.Series([""] * len(bets), index=bets.index)).astype(str).fillna("").str.strip().str.lower()
        stat_key = bets.get("stat_key", pd.Series([""] * len(bets), index=bets.index)).astype(str).fillna("").str.strip().str.lower()

        gid2 = gid.where(gid.str.len() > 0, other=(home + "@" + away))
        key_player = m + "|" + horizon + "|" + gid2 + "|" + name_key + "|" + stat_key + "|" + side
        key_other = m + "|" + horizon + "|" + gid2 + "|" + side
        key = key_other.where(m != "player_prop", other=key_player)
        bets["_dedup_key"] = key

        bets = bets.sort_values(["_elapsed", "_idx"], ascending=[True, True], na_position="last")
        bets = bets.drop_duplicates(subset=["_dedup_key"], keep="first")
        bets = bets.drop(columns=["_idx", "_elapsed", "_dedup_key"], errors="ignore")
        return bets.reset_index(drop=True)


def _metrics(df: pd.DataFrame) -> dict[str, float]:
    if df is None or df.empty:
        return {"n": 0}
    err = df["pred"].astype(float) - df["act"].astype(float)
    mae = float(err.abs().mean())
    rmse = float(math.sqrt(float((err**2).mean())))
    bias = float(err.mean())
    return {"n": int(len(df)), "mae": mae, "rmse": rmse, "bias": bias}


def _hit_rate(df: pd.DataFrame) -> dict[str, float]:
    if df is None or df.empty:
        return {"n": 0}
    d = df[df["result"].notna()].copy()
    if d.empty:
        return {"n": int(len(df))}
    wins = int((d["result"].astype(str) == "win").sum())
    losses = int((d["result"].astype(str) == "loss").sum())
    pushes = int((d["result"].astype(str) == "push").sum())
    denom = wins + losses
    hr = float(wins) / float(denom) if denom > 0 else float("nan")
    return {"n": int(len(df)), "wins": wins, "losses": losses, "pushes": pushes, "hit_rate": hr}


def _parse_tags(obj: dict[str, Any]) -> list[str]:
    """Extract a normalized tag list from a raw signal payload.

    We support multiple potential fields for NCAAB parity and forward-compat:
      - tags: [..] or "a,b"
      - driver_tags: [..]
      - signal_tags: [..]

    We intentionally do NOT infer tags from player names or signal_key to avoid
    exploding cardinality.
    """
    cand: list[Any] = []
    for k in ("tags", "driver_tags", "signal_tags"):
        v = obj.get(k)
        if v is None:
            continue
        if isinstance(v, list):
            cand.extend(v)
        elif isinstance(v, str):
            cand.extend([x.strip() for x in v.replace(";", ",").split(",")])
        else:
            cand.append(v)

    out: list[str] = []
    seen: set[str] = set()
    for x in cand:
        try:
            s = str(x or "").strip()
        except Exception:
            s = ""
        if not s:
            continue
        s2 = re.sub(r"\s+", "_", s.strip().lower())
        if not s2 or s2 in {"none", "null", "nan"}:
            continue
        if s2 not in seen:
            seen.add(s2)
            out.append(s2)
    return out


def _driver_tags_from_tags(tags: Any) -> list[str]:
    items = tags if isinstance(tags, list) else ([] if tags is None else [str(tags)])
    out: list[str] = []
    seen: set[str] = set()
    for t in items:
        s = str(t or "").strip().lower()
        if not s:
            continue
        if s.startswith("market:") or s.startswith("horizon:") or s.startswith("klass:") or s.startswith("stat:"):
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _derive_tags(
    *,
    obj: dict[str, Any],
    market: str,
    horizon: str | None,
    klass: str | None,
    stat_key: str | None,
    interval_drift_on: int | None,
    recent_window_on: int | None,
    endgame_foul_on: int | None,
) -> list[str]:
    tags: list[str] = []

    def _nf(x: Any) -> float | None:
        try:
            if x is None:
                return None
            if isinstance(x, float) and math.isnan(x):
                return None
            return float(x)
        except Exception:
            return None

    def _add(t: str | None):
        if not t:
            return
        t2 = re.sub(r"\s+", "_", str(t).strip().lower())
        if not t2 or t2 in {"none", "null", "nan"}:
            return
        tags.append(t2)

    # Raw, client-provided tags (if any).
    for t in _parse_tags(obj):
        _add(t)

    # Stable, low-cardinality derived tags.
    _add(f"market:{market}")
    if horizon:
        _add(f"horizon:{horizon}")
    if klass:
        _add(f"klass:{klass}")
    if market == "player_prop" and stat_key:
        _add(f"stat:{stat_key}")

    # Player-prop driver tags (low-cardinality, decision-shaping context).
    # Source of truth for these drivers is the client-side adjustPlayerPropSignal() policy.
    try:
        if market == "player_prop":
            ctx = obj.get("context")

            # Top-level fields (present in live_lens_signal payload for props)
            mp = _nf(obj.get("mp"))
            pf = _nf(obj.get("pf"))

            inj = None
            rot_on = None
            rot_off_sec = None
            sim_vs_line = None
            proj_min_final = None
            exp_min_eff = None
            adj_risk = None
            adj_support = None

            if isinstance(ctx, dict):
                inj = ctx.get("injury_flag")
                rot_on = ctx.get("rot_on_court")
                rot_off_sec = _nf(ctx.get("rot_cur_off_sec"))
                sim_vs_line = _nf(ctx.get("sim_vs_line"))
                proj_min_final = _nf(ctx.get("proj_min_final"))
                exp_min_eff = _nf(ctx.get("exp_min_eff"))
                adj_risk = _nf(ctx.get("adj_risk"))
                adj_support = _nf(ctx.get("adj_support"))

            # Injury / rotation availability
            if inj is not None:
                _add("injury:on" if bool(inj) else "injury:off")
            if rot_on is not None:
                _add("rot:on_court" if bool(rot_on) else "rot:off_court")
            if rot_off_sec is not None:
                if rot_off_sec >= 720:
                    _add("rot_off:12m+")
                elif rot_off_sec >= 480:
                    _add("rot_off:8-12m")
                elif rot_off_sec >= 240:
                    _add("rot_off:4-8m")
                else:
                    _add("rot_off:<4m")

            # Foul trouble
            if pf is not None:
                if pf >= 5:
                    _add("pf:5+")
                elif pf >= 4:
                    _add("pf:4")
                elif pf >= 2:
                    _add("pf:2-3")
                else:
                    _add("pf:0-1")

            # Minutes remaining (proj - mp)
            proj = proj_min_final if proj_min_final is not None else exp_min_eff
            rem = (proj - mp) if (proj is not None and mp is not None) else None
            if proj is not None:
                if proj < 16:
                    _add("proj_min:<16")
                elif proj < 24:
                    _add("proj_min:16-24")
                elif proj < 32:
                    _add("proj_min:24-32")
                else:
                    _add("proj_min:32+")
            if rem is not None:
                if rem < 1.5:
                    _add("rem_min:<1.5")
                elif rem < 3.5:
                    _add("rem_min:1.5-3.5")
                else:
                    _add("rem_min:3.5+")
            if mp is not None and mp < 3.0:
                _add("mp:<3")

            # SmartSim agreement / disagreement (coarse)
            edge = _nf(obj.get("edge"))
            if edge is None:
                edge = _nf(obj.get("pace_vs_line"))
            if edge is not None and sim_vs_line is not None and edge != 0 and sim_vs_line != 0:
                if (edge * sim_vs_line) < 0 and abs(edge) >= 2.0 and abs(sim_vs_line) >= 2.0:
                    _add("sim:disagree")
                elif (edge * sim_vs_line) > 0 and abs(edge) >= 2.0 and abs(sim_vs_line) >= 2.0:
                    _add("sim:agree")

            # Adjusted risk/support (from client-side adjustment)
            if adj_risk is not None:
                if adj_risk >= 4.0:
                    _add("risk:4+")
                elif adj_risk >= 2.0:
                    _add("risk:2-4")
                else:
                    _add("risk:<2")
            if adj_support is not None:
                if adj_support >= 0.8:
                    _add("support:high")
                elif adj_support <= -0.8:
                    _add("support:low")
                else:
                    _add("support:mid")

            # Market-quality / bettability tags (available in server-side live prop logs).
            line_source = str(obj.get("line_source") or (ctx.get("line_source") if isinstance(ctx, dict) else "") or "").strip().lower() or None
            if line_source:
                _add(f"line_src:{line_source}")

            line_live_age_sec = _nf(obj.get("line_live_age_sec"))
            if line_live_age_sec is None and isinstance(ctx, dict):
                line_live_age_sec = _nf(ctx.get("line_live_age_sec"))
            if line_source == "oddsapi":
                if line_live_age_sec is None:
                    _add("line_age:unknown")
                elif line_live_age_sec < 180:
                    _add("line_age:fresh")
                elif line_live_age_sec < 600:
                    _add("line_age:warm")
                else:
                    _add("line_age:stale")

            line_live_span = _nf(obj.get("line_live_span"))
            if line_live_span is None and isinstance(ctx, dict):
                line_live_span = _nf(ctx.get("line_live_span"))
            if line_live_span is not None:
                if line_live_span < 0.5:
                    _add("market_span:tight")
                elif line_live_span < 1.0:
                    _add("market_span:mid")
                else:
                    _add("market_span:wide")

            line_live_n = _nf(obj.get("line_live_n"))
            if line_live_n is None and isinstance(ctx, dict):
                line_live_n = _nf(ctx.get("line_live_n"))
            if line_live_n is not None:
                if line_live_n <= 1:
                    _add("books:1")
                elif line_live_n <= 2:
                    _add("books:2")
                else:
                    _add("books:3+")

            price_hold = _nf(obj.get("price_hold"))
            if price_hold is None and isinstance(ctx, dict):
                price_hold = _nf(ctx.get("price_hold"))
            if price_hold is not None:
                if price_hold <= 0.04:
                    _add("hold:low")
                elif price_hold <= 0.08:
                    _add("hold:mid")
                else:
                    _add("hold:high")

            bettable = obj.get("bettable")
            if bettable is None and isinstance(ctx, dict):
                bettable = ctx.get("bettable")
            if bettable is not None:
                _add("bettable:yes" if bool(bettable) else "bettable:no")

            bettable_score = _nf(obj.get("bettable_score"))
            if bettable_score is None and isinstance(ctx, dict):
                bettable_score = _nf(ctx.get("bettable_score"))
            if bettable_score is not None:
                if bettable_score >= 0.75:
                    _add("bet_score:high")
                elif bettable_score >= 0.45:
                    _add("bet_score:mid")
                else:
                    _add("bet_score:low")

            edge_sigma = _nf(obj.get("edge_sigma"))
            if edge_sigma is None and isinstance(ctx, dict):
                edge_sigma = _nf(ctx.get("edge_sigma"))
            if edge_sigma is not None:
                if edge_sigma < 0.5:
                    _add("edge_sigma:<0.5")
                elif edge_sigma < 1.0:
                    _add("edge_sigma:0.5-1.0")
                else:
                    _add("edge_sigma:1+")

            bettable_reasons = obj.get("bettable_reasons")
            if bettable_reasons is None and isinstance(ctx, dict):
                bettable_reasons = ctx.get("bettable_reasons")
            if isinstance(bettable_reasons, (list, tuple, set)):
                for reason in bettable_reasons:
                    rs = re.sub(r"[^a-z0-9_]+", "_", str(reason or "").strip().lower()).strip("_")
                    if rs:
                        _add(f"gate:{rs}")

            pace_mult = _nf((ctx.get("pace_mult") if isinstance(ctx, dict) else None) or obj.get("pace_mult"))
            if pace_mult is not None:
                if pace_mult < 0.97:
                    _add("pace_mult:down")
                elif pace_mult > 1.03:
                    _add("pace_mult:up")
                else:
                    _add("pace_mult:flat")

            role_mult = _nf((ctx.get("role_mult") if isinstance(ctx, dict) else None) or obj.get("role_mult"))
            if role_mult is not None:
                if role_mult < 0.97:
                    _add("role_mult:down")
                elif role_mult > 1.03:
                    _add("role_mult:up")
                else:
                    _add("role_mult:flat")

            hot_cold_mult = _nf((ctx.get("hot_cold_mult") if isinstance(ctx, dict) else None) or obj.get("hot_cold_mult"))
            if hot_cold_mult is not None:
                if hot_cold_mult < 0.97:
                    _add("hot:down")
                elif hot_cold_mult > 1.03:
                    _add("hot:up")
                else:
                    _add("hot:flat")

            foul_mult = _nf((ctx.get("foul_mult") if isinstance(ctx, dict) else None) or obj.get("foul_mult"))
            if foul_mult is not None:
                if foul_mult < 0.97:
                    _add("foul_adj:down")
                elif foul_mult > 1.03:
                    _add("foul_adj:up")
                else:
                    _add("foul_adj:flat")
    except Exception:
        pass

    # Totals driver tags
    try:
        if market in {"total", "half_total", "quarter_total"}:
            ctx = obj.get("context")
            _add("ctx:present" if isinstance(ctx, dict) else "ctx:missing")
            lam = _nf(ctx.get("edge_shrink_lambda")) if isinstance(ctx, dict) else None
            if lam is not None:
                if lam < 0.25:
                    _add("shrink:0-0.25")
                elif lam < 0.5:
                    _add("shrink:0.25-0.5")
                elif lam < 0.75:
                    _add("shrink:0.5-0.75")
                else:
                    _add("shrink:0.75+")

            # Scope/time buckets (low-cardinality; helps optimize when signals work).
            scope_min = 48.0 if market == "total" else (24.0 if market == "half_total" else 12.0)
            rem = _nf(obj.get("remaining"))
            if rem is not None and scope_min > 0:
                frac = max(0.0, min(1.0, float(rem) / float(scope_min)))
                if frac > 0.66:
                    _add("time:early")
                elif frac > 0.33:
                    _add("time:mid")
                else:
                    _add("time:late")

            # Margin buckets (game state).
            margin = _nf(obj.get("margin_home"))
            if margin is not None:
                am = abs(float(margin))
                if am <= 4.0:
                    _add("mgn:close")
                elif am <= 10.0:
                    _add("mgn:mid")
                else:
                    _add("mgn:blowout")

            # Quarter scope availability (q1/q3 have UI scope columns; q2/q4 may be missing).
            if market == "quarter_total" and isinstance(ctx, dict):
                sp = ctx.get("scope_present")
                if sp is not None:
                    _add("scope:present" if int(sp) == 1 else "scope:missing")

            # Pace / possessions / PPP condition tags (only when context carries these).
            if isinstance(ctx, dict):
                pace_ratio = _nf(ctx.get("pace_ratio"))
                if pace_ratio is not None:
                    if pace_ratio < 0.96:
                        _add("pace:slow")
                    elif pace_ratio > 1.04:
                        _add("pace:fast")
                    else:
                        _add("pace:normal")

                poss_live = _nf(ctx.get("poss_live"))
                poss_exp = _nf(ctx.get("poss_expected_so_far"))
                if poss_exp is None:
                    poss_exp = _nf(ctx.get("poss_expected"))
                if poss_live is not None and poss_exp is not None and float(poss_exp) > 1e-9:
                    pr = float(poss_live) / float(poss_exp)
                    if pr < 0.9:
                        _add("poss:low")
                    elif pr > 1.1:
                        _add("poss:high")
                    else:
                        _add("poss:normal")

                ppp_delta = _nf(ctx.get("eff_ppp_delta"))
                if ppp_delta is None:
                    exp_ppp = _nf(ctx.get("exp_ppp"))
                    act_ppp = _nf(ctx.get("act_ppp"))
                    if exp_ppp is not None and act_ppp is not None:
                        ppp_delta = float(act_ppp) - float(exp_ppp)
                if ppp_delta is not None:
                    if ppp_delta > 0.03:
                        _add("ppp:hot")
                    elif ppp_delta < -0.03:
                        _add("ppp:cold")
                    else:
                        _add("ppp:normal")

                # Adjustment magnitude buckets (when adjustments are on).
                sa = ctx.get("scope_adjustments")
                src = sa if isinstance(sa, dict) else ctx

                id_adj = _nf(src.get("interval_drift_adj"))
                if id_adj is not None and interval_drift_on is not None and int(interval_drift_on) == 1:
                    mag = abs(float(id_adj))
                    if mag < 0.5:
                        _add("interval_drift_mag:small")
                    elif mag < 1.5:
                        _add("interval_drift_mag:med")
                    else:
                        _add("interval_drift_mag:large")

                rw_w = _nf(src.get("recent_window_w"))
                if rw_w is not None and recent_window_on is not None and int(recent_window_on) == 1:
                    ww = abs(float(rw_w))
                    if ww < 0.25:
                        _add("recent_window_w:small")
                    elif ww < 0.6:
                        _add("recent_window_w:med")
                    else:
                        _add("recent_window_w:large")

                foul_adj = _nf(src.get("endgame_foul_adj"))
                if foul_adj is not None and endgame_foul_on is not None and int(endgame_foul_on) == 1:
                    mag = abs(float(foul_adj))
                    if mag < 0.5:
                        _add("endgame_foul_mag:small")
                    elif mag < 1.5:
                        _add("endgame_foul_mag:med")
                    else:
                        _add("endgame_foul_mag:large")
    except Exception:
        pass

    # ATS driver tags
    try:
        if market == "ats":
            ctx = obj.get("context")
            _add("ctx:present" if isinstance(ctx, dict) else "ctx:missing")

            margin = _nf(obj.get("margin_home"))
            if margin is not None:
                am = abs(float(margin))
                if am <= 4.0:
                    _add("mgn:close")
                elif am <= 10.0:
                    _add("mgn:mid")
                else:
                    _add("mgn:blowout")

            if isinstance(ctx, dict):
                pick_home = ctx.get("pick_home")
                if pick_home is not None:
                    _add("ats_side:home" if int(pick_home) == 1 else "ats_side:away")

                spr = _nf(ctx.get("spr_home"))
                if spr is not None:
                    s = abs(float(spr))
                    if s < 3.0:
                        _add("spr:<3")
                    elif s < 7.0:
                        _add("spr:3-7")
                    else:
                        _add("spr:7+")

                elapsed = _nf(ctx.get("elapsed_min"))
                if elapsed is not None:
                    frac = max(0.0, min(1.0, float(elapsed) / 48.0))
                    if frac < 0.34:
                        _add("time:early")
                    elif frac < 0.67:
                        _add("time:mid")
                    else:
                        _add("time:late")
    except Exception:
        pass

    # Adjustment toggles for totals signals.
    if interval_drift_on is not None:
        _add("interval_drift:on" if int(interval_drift_on) == 1 else "interval_drift:off")
    if recent_window_on is not None:
        _add("recent_window:on" if int(recent_window_on) == 1 else "recent_window:off")
    if endgame_foul_on is not None:
        _add("endgame_foul:on" if int(endgame_foul_on) == 1 else "endgame_foul:off")

    # Dedupe while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _flag_adjustments(ctx: Any) -> tuple[int | None, int | None, int | None]:
    if not isinstance(ctx, dict):
        return (None, None, None)
    # Some markets nest these under context.scope_adjustments; others place them directly in context.
    sa = ctx.get("scope_adjustments")
    src = sa if isinstance(sa, dict) else ctx

    interval_on = 1 if abs(float(_n(src.get("interval_drift_adj")) or 0.0)) > 1e-9 else 0

    recent_on = 0
    if abs(float(_n(src.get("recent_window_pace_adj")) or 0.0)) > 1e-9:
        recent_on = 1
    if abs(float(_n(src.get("recent_window_eff_adj")) or 0.0)) > 1e-9:
        recent_on = 1
    if float(_n(src.get("recent_window_w")) or 0.0) > 1e-9:
        recent_on = 1

    foul_on = 1 if abs(float(_n(src.get("endgame_foul_adj")) or 0.0)) > 1e-9 else 0
    if float(_n(src.get("endgame_foul_w")) or 0.0) > 1e-9:
        foul_on = 1
    return (interval_on, recent_on, foul_on)


def _score_day(ds: str, *, dedup_policy: str = "none", include_model_lines: bool = False) -> pd.DataFrame:
    sigs = _load_signals(ds)
    if not sigs:
        return pd.DataFrame()

    rg_path = PROCESSED / f"recon_games_{ds}.csv"
    rq_path = PROCESSED / f"recon_quarters_{ds}.csv"
    rp_path = PROCESSED / f"recon_props_{ds}.csv"

    rg = _prep_recon_games(_load_csv(rg_path))
    rq = _prep_recon_quarters(_load_csv(rq_path))
    rp = _prep_recon_props(_load_csv(rp_path))

    rg_ok = int(not rg.empty)
    rq_ok = int(not rq.empty)
    rp_ok = int(not rp.empty)

    gid_map = _load_gid_map(ds)

    prop_index: dict[tuple[str, str], dict[str, Any]] = {}
    if not rp.empty:
        for _, r in rp.iterrows():
            gid = str(r.get("_gid") or "")
            nk = str(r.get("_name_key") or "")
            if gid and nk and (gid, nk) not in prop_index:
                prop_index[(gid, nk)] = dict(r)

    scored: list[dict[str, Any]] = []

    for obj in sigs:
        market = str(obj.get("market") or "").strip()

        if market in {"total", "half_total", "quarter_total"}:
            horizon = str(obj.get("horizon") or "").strip().lower() or None
            gid = _resolve_gid_for_game(obj, gid_map)
            home = _safe_upper(obj.get("home"))
            away = _safe_upper(obj.get("away"))
            live_line = _n(obj.get("live_line"))
            edge = _n(obj.get("edge_adj")) if market == "total" else _n(obj.get("edge"))
            pred = _n(obj.get("pred"))
            if pred is None:
                pred = (live_line + edge) if (live_line is not None and edge is not None) else None
            act = _actual_total(market, horizon, gid, home, away, rg, rq)
            missing_reason = ""
            if act is None:
                if market == "half_total" and horizon not in {"h1", "h2"}:
                    missing_reason = "unsupported_horizon"
                elif market == "quarter_total" and horizon not in {"q1", "q2", "q3", "q4"}:
                    missing_reason = "unsupported_horizon"
                if market == "total" and rg_ok == 0:
                    missing_reason = "missing_recon_games"
                elif market in {"half_total", "quarter_total"} and rq_ok == 0:
                    missing_reason = "missing_recon_quarters"
                else:
                    missing_reason = missing_reason or "join_failed"
            else:
                missing_reason = ""
            side = str(obj.get("side") or "").strip().lower() or None
            result = _result_for_side(side, act, live_line)
            err = (pred - act) if (pred is not None and act is not None) else None
            ctx = obj.get("context")
            interval_on, recent_on, foul_on = _flag_adjustments(ctx)
            w_pace = None
            edge_shrink_lambda = None
            edge_shrink_lambda_poss = None
            edge_shrink_lambda_time = None
            if isinstance(ctx, dict):
                w_pace = _n(ctx.get("w_pace"))
                edge_shrink_lambda = _n(ctx.get("edge_shrink_lambda"))
                edge_shrink_lambda_poss = _n(ctx.get("edge_shrink_lambda_poss"))
                edge_shrink_lambda_time = _n(ctx.get("edge_shrink_lambda_time"))
            shape_summary = str(obj.get("shape_summary") or "").strip() or None
            shape_reasons = obj.get("shape_reasons") if isinstance(obj.get("shape_reasons"), list) else []
            recent_window_pace_adj = _n(ctx.get("recent_window_pace_adj")) if isinstance(ctx, dict) else None
            recent_window_eff_adj = _n(ctx.get("recent_window_eff_adj")) if isinstance(ctx, dict) else None
            recent_window_streak_adj = _n(ctx.get("recent_window_streak_adj")) if isinstance(ctx, dict) else None
            recent_window_w = _n(ctx.get("recent_window_w")) if isinstance(ctx, dict) else None
            recent_window_rate = _n(ctx.get("recent_window_rate")) if isinstance(ctx, dict) else None
            recent_window_scope_rate = _n(ctx.get("recent_window_scope_rate")) if isinstance(ctx, dict) else None
            recent_window_ppp = _n(ctx.get("recent_window_ppp")) if isinstance(ctx, dict) else None
            recent_window_scope_ppp = _n(ctx.get("recent_window_scope_ppp")) if isinstance(ctx, dict) else None
            recent_window_efg_pct = _n(ctx.get("recent_window_efg_pct")) if isinstance(ctx, dict) else None
            recent_window_ts_pct = _n(ctx.get("recent_window_ts_pct")) if isinstance(ctx, dict) else None
            recent_window_run_team = str(ctx.get("recent_window_run_team") or "").strip().upper() or None if isinstance(ctx, dict) else None
            recent_window_run_points = _n(ctx.get("recent_window_run_points")) if isinstance(ctx, dict) else None
            recent_window_seconds_since_score = _n(ctx.get("recent_window_seconds_since_score")) if isinstance(ctx, dict) else None

            scored.append(
                {
                    "date": str(obj.get("date") or ds),
                    "market": market,
                    "horizon": horizon,
                    "signal_key": str(obj.get("signal_key") or "") or None,
                    "game_id": gid,
                    "home": home,
                    "away": away,
                    "team_tri": None,
                    "player": None,
                    "name_key": None,
                    "stat": None,
                    "stat_key": None,
                    "side": side,
                    "klass": str(obj.get("klass") or "") or None,
                    "elapsed": _n(obj.get("elapsed")),
                    "live_line": live_line,
                    "edge": edge,
                    "strength": (abs(float(edge)) if edge is not None else None),
                    "pred": pred,
                    "act": act,
                    "result": result,
                    "err": err,
                    "has_context": (1 if isinstance(obj.get("context"), dict) else 0),
                    "tags": _derive_tags(
                        obj=obj,
                        market=market,
                        horizon=horizon,
                        klass=str(obj.get("klass") or "") or None,
                        stat_key=None,
                        interval_drift_on=interval_on,
                        recent_window_on=recent_on,
                        endgame_foul_on=foul_on,
                    ),
                    "w_pace": w_pace,
                    "edge_shrink_lambda": edge_shrink_lambda,
                    "edge_shrink_lambda_poss": edge_shrink_lambda_poss,
                    "edge_shrink_lambda_time": edge_shrink_lambda_time,
                    "shape_summary": shape_summary,
                    "shape_reasons": shape_reasons,
                    "recent_window_pace_adj": recent_window_pace_adj,
                    "recent_window_eff_adj": recent_window_eff_adj,
                    "recent_window_streak_adj": recent_window_streak_adj,
                    "recent_window_w": recent_window_w,
                    "recent_window_rate": recent_window_rate,
                    "recent_window_scope_rate": recent_window_scope_rate,
                    "recent_window_ppp": recent_window_ppp,
                    "recent_window_scope_ppp": recent_window_scope_ppp,
                    "recent_window_efg_pct": recent_window_efg_pct,
                    "recent_window_ts_pct": recent_window_ts_pct,
                    "recent_window_run_team": recent_window_run_team,
                    "recent_window_run_points": recent_window_run_points,
                    "recent_window_seconds_since_score": recent_window_seconds_since_score,
                    "missing_reason": missing_reason,
                    "has_recon_games": rg_ok,
                    "has_recon_quarters": rq_ok,
                    "has_recon_props": rp_ok,
                    "interval_drift_on": interval_on,
                    "recent_window_on": recent_on,
                    "endgame_foul_on": foul_on,
                }
            )
            continue

        if market == "ats":
            # Live ATS signals settle on final score vs the *picked-side spread*.
            gid = _resolve_gid_for_game(obj, gid_map)
            home = _safe_upper(obj.get("home"))
            away = _safe_upper(obj.get("away"))
            line = _n(obj.get("live_line"))
            side_raw = str(obj.get("side") or "").strip()
            side = _safe_upper(side_raw)
            klass = str(obj.get("klass") or "") or None
            # Actuals come from recon_games.
            hp = ap = None
            try:
                if not rg.empty:
                    hit = pd.DataFrame()
                    if gid and "_gid" in rg.columns:
                        hit = rg[rg.get("_gid") == gid]
                    if hit.empty and home and away:
                        hit = rg[(rg.get("home_tri") == home) & (rg.get("away_tri") == away)]
                    if not hit.empty:
                        hp = _n(hit.iloc[0].get("home_pts"))
                        ap = _n(hit.iloc[0].get("visitor_pts"))
            except Exception:
                hp = ap = None

            missing_reason = ""
            if hp is None or ap is None:
                missing_reason = "missing_recon_games" if rg_ok == 0 else "join_failed"
            if line is None:
                missing_reason = missing_reason or "missing_line"
            if not side:
                missing_reason = missing_reason or "missing_side"

            result = None
            act = None
            if hp is not None and ap is not None and side and line is not None and home and away:
                margin = float(hp) - float(ap)
                act = margin
                # Convention: line is the spread for the PICKED TEAM (positive for underdog).
                if side == home:
                    diff = margin + float(line)
                elif side == away:
                    diff = (float(ap) - float(hp)) + float(line)
                else:
                    diff = None
                if diff is not None:
                    if abs(diff) < 1e-9:
                        result = "push"
                    else:
                        result = "win" if diff > 0 else "loss"

            scored.append(
                {
                    "date": str(obj.get("date") or ds),
                    "market": market,
                    "horizon": str(obj.get("horizon") or "").strip().lower() or None,
                    "signal_key": str(obj.get("signal_key") or "") or None,
                    "game_id": gid,
                    "home": home,
                    "away": away,
                    "team_tri": None,
                    "player": None,
                    "name_key": None,
                    "stat": None,
                    "stat_key": None,
                    "side": (side_raw.strip().lower() or None),
                    "klass": klass,
                    "elapsed": _n(obj.get("elapsed")),
                    "live_line": line,
                    "edge": _n(obj.get("edge")),
                    "strength": (abs(float(_n(obj.get("edge")) or 0.0)) if _n(obj.get("edge")) is not None else None),
                    "pred": None,
                    "act": act,
                    "result": result,
                    "err": None,
                    "has_context": (1 if isinstance(obj.get("context"), dict) else 0),
                    "tags": _derive_tags(
                        obj=obj,
                        market=market,
                        horizon=str(obj.get("horizon") or "").strip().lower() or None,
                        klass=klass,
                        stat_key=None,
                        interval_drift_on=None,
                        recent_window_on=None,
                        endgame_foul_on=None,
                    ),
                    "missing_reason": missing_reason,
                    "has_recon_games": rg_ok,
                    "has_recon_quarters": rq_ok,
                    "has_recon_props": rp_ok,
                    "interval_drift_on": None,
                    "recent_window_on": None,
                    "endgame_foul_on": None,
                }
            )
            continue

        if market == "player_prop":
            # Align with ROI/tuning policy: model-fallback lines are diagnostic-only.
            # Exclude them from audit metrics by default (opt-in via CLI).
            try:
                if (not include_model_lines) and (str(obj.get("line_source") or "").strip().lower() == "model"):
                    continue
            except Exception:
                pass

            gid = _resolve_gid_for_props(obj, gid_map)
            horizon = str(obj.get("horizon") or "").strip().lower() or None
            if horizon is None:
                horizon = "live"
            stat = str(obj.get("stat") or "").strip()
            stat_key = _live_stat_key(stat)
            player = str(obj.get("player") or "").strip() or None
            name_key_raw = str(obj.get("name_key") or "").strip() or None
            name_key = _norm_player_name(name_key_raw or player or "") or None
            side = str(obj.get("side") or "").strip().lower() or None

            try:
                line_source = str(obj.get("line_source") or "").strip().lower() or None
            except Exception:
                line_source = None

            line = _n(obj.get("line"))
            if line is None:
                line = _n(obj.get("live_line"))

            edge = _n(obj.get("edge"))
            if edge is None:
                edge = _n(obj.get("pace_vs_line"))

            pred = (line + edge) if (line is not None and edge is not None) else None

            act = None
            r = None
            if gid and name_key:
                r = prop_index.get((gid, name_key))
                if r is not None:
                    act = _n(r.get(stat_key))
            if act is None:
                if rp_ok == 0:
                    missing_reason = "missing_recon_props"
                elif not gid:
                    missing_reason = "missing_gid"
                elif not name_key:
                    missing_reason = "missing_name_key"
                elif r is None:
                    missing_reason = "player_join_failed"
                else:
                    missing_reason = "stat_missing"
            else:
                missing_reason = ""
            result = _result_for_side(side, act, line)
            err = (pred - act) if (pred is not None and act is not None) else None

            # Carry forward useful diagnostics for clustering.
            ctx = obj.get("context")
            exp_min = exp_min_eff = proj_min_final = None
            usage_window_sec = None
            pace_mult = role_mult = foul_mult = None
            usg_recent = usg_game = team_usg_recent = team_usg_game = None
            fg3a_recent = fg3a_game = team_3a_recent = team_3a_game = None
            if isinstance(ctx, dict):
                exp_min = _n(ctx.get("exp_min"))
                exp_min_eff = _n(ctx.get("exp_min_eff"))
                proj_min_final = _n(ctx.get("proj_min_final"))
                usage_window_sec = _n(ctx.get("usage_window_sec"))
                pace_mult = _n(ctx.get("pace_mult"))
                role_mult = _n(ctx.get("role_mult"))
                foul_mult = _n(ctx.get("foul_mult"))
                usg_recent = _n(ctx.get("usg_recent"))
                usg_game = _n(ctx.get("usg_game"))
                team_usg_recent = _n(ctx.get("team_usg_recent"))
                team_usg_game = _n(ctx.get("team_usg_game"))
                fg3a_recent = _n(ctx.get("fg3a_recent"))
                fg3a_game = _n(ctx.get("fg3a_game"))
                team_3a_recent = _n(ctx.get("team_3a_recent"))
                team_3a_game = _n(ctx.get("team_3a_game"))

            scored.append(
                {
                    "date": str(obj.get("date") or ds),
                    "market": market,
                    "horizon": horizon,
                    "signal_key": str(obj.get("signal_key") or "") or None,
                    "game_id": gid,
                    "home": _safe_upper(obj.get("home")),
                    "away": _safe_upper(obj.get("away")),
                    "team_tri": _safe_upper(obj.get("team_tri")),
                    "player": player,
                    "name_key": name_key,
                    "stat": stat,
                    "stat_key": stat_key,
                    "side": side,
                    "klass": str(obj.get("klass") or "") or None,
                    "line_source": line_source,
                    "elapsed": _n(obj.get("elapsed")),
                    "live_line": line,
                    "edge": edge,
                    "strength": (abs(float(edge)) if edge is not None else None),
                    "pred": pred,
                    "act": act,
                    "result": result,
                    "err": err,
                    "has_context": (1 if isinstance(obj.get("context"), dict) else 0),
                    "tags": _derive_tags(
                        obj=obj,
                        market=market,
                        horizon=horizon,
                        klass=str(obj.get("klass") or "") or None,
                        stat_key=stat_key,
                        interval_drift_on=None,
                        recent_window_on=None,
                        endgame_foul_on=None,
                    ),
                    "exp_min": exp_min,
                    "exp_min_eff": exp_min_eff,
                    "proj_min_final": proj_min_final,
                    "usage_window_sec": usage_window_sec,
                    "pace_mult": pace_mult,
                    "role_mult": role_mult,
                    "foul_mult": foul_mult,
                    "usg_recent": usg_recent,
                    "usg_game": usg_game,
                    "team_usg_recent": team_usg_recent,
                    "team_usg_game": team_usg_game,
                    "fg3a_recent": fg3a_recent,
                    "fg3a_game": fg3a_game,
                    "team_3a_recent": team_3a_recent,
                    "team_3a_game": team_3a_game,
                    "missing_reason": missing_reason,
                    "has_recon_games": rg_ok,
                    "has_recon_quarters": rq_ok,
                    "has_recon_props": rp_ok,
                    "interval_drift_on": None,
                    "recent_window_on": None,
                    "endgame_foul_on": None,
                }
            )
            continue

    if not scored:
        return pd.DataFrame()
    for row in scored:
        try:
            row["driver_tags"] = _driver_tags_from_tags(row.get("tags"))
        except Exception:
            row["driver_tags"] = []
    df = pd.DataFrame(scored)
    if str(dedup_policy or "none").strip().lower() in {"first_bet", "first_bets"}:
        return _dedup_first_bets(df)
    return df


def _write_markdown(ds: str, df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _fmt_metrics(m: dict[str, float]) -> str:
        if not m or int(m.get("n", 0)) <= 0:
            return "n=0"
        return f"n={int(m['n'])}  mae={m.get('mae', float('nan')):.3f}  rmse={m.get('rmse', float('nan')):.3f}  bias={m.get('bias', float('nan')):.3f}"

    def _fmt_hits(h: dict[str, float]) -> str:
        if not h or int(h.get("n", 0)) <= 0:
            return "n=0"
        hr = h.get("hit_rate")
        hr_s = f"{float(hr):.3f}" if (hr is not None and not math.isnan(float(hr))) else "nan"
        return f"n={int(h['n'])}  W={int(h.get('wins', 0))}  L={int(h.get('losses', 0))}  P={int(h.get('pushes', 0))}  hit={hr_s}"

    lines: list[str] = []
    lines.append(f"# Live Lens Audit — {ds}")
    lines.append("")

    if df is None or df.empty:
        lines.append("No scored rows (missing logs or missing recon outputs).")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append("## Coverage")
    counts = df.groupby("market").size().to_dict()
    lines.append(
        f"- recon files loaded: games={int(df.get('has_recon_games', pd.Series([0])).max() if not df.empty else 0)} quarters={int(df.get('has_recon_quarters', pd.Series([0])).max() if not df.empty else 0)} props={int(df.get('has_recon_props', pd.Series([0])).max() if not df.empty else 0)}"
    )
    for k in sorted(counts.keys()):
        n0 = int(counts[k])
        if "has_context" in df.columns:
            try:
                n_ctx = int(pd.to_numeric(df[df["market"] == k].get("has_context"), errors="coerce").fillna(0).sum())
            except Exception:
                n_ctx = 0
            lines.append(f"- {k}: {n0}  ctx={n_ctx}/{n0}")
        else:
            lines.append(f"- {k}: {n0}")
    if "missing_reason" in df.columns:
        miss = df[df["missing_reason"].astype(str).str.len() > 0]
        if not miss.empty:
            mr = miss.groupby("missing_reason").size().sort_values(ascending=False).to_dict()
            parts = [f"{k}={int(v)}" for k, v in mr.items()]
            lines.append(f"- unscored (no actual): {', '.join(parts)}")
    lines.append("")

    totals = df[df["market"].isin(["total", "half_total", "quarter_total"])].copy()
    totals = totals.dropna(subset=["pred", "act"]).copy()

    lines.append("## Totals/halves/quarters")
    totals_all = df[df["market"].isin(["total", "half_total", "quarter_total"])].copy()
    for mkt in ["total", "half_total", "quarter_total"]:
        d_all = totals_all[totals_all["market"] == mkt].copy()
        n_all = int(len(d_all))
        n_act = int(d_all["act"].notna().sum()) if "act" in d_all.columns else 0
        n_pred = int(d_all["pred"].notna().sum()) if "pred" in d_all.columns else 0
        d_scored = d_all.dropna(subset=["pred", "act"]).copy()
        m = _metrics(d_scored)
        lines.append(f"- {mkt}: scored={_fmt_metrics(m)}  have_pred={n_pred}/{n_all}  have_act={n_act}/{n_all}")

    # Quick visibility into whether the new shrink diagnostics are making it into logs.
    try:
        if "edge_shrink_lambda" in totals_all.columns:
            n_lam = int(pd.to_numeric(totals_all.get("edge_shrink_lambda"), errors="coerce").notna().sum())
            lines.append(f"- edge_shrink_lambda present: {n_lam}/{int(len(totals_all))}")
    except Exception:
        pass
    lines.append("")

    if not totals.empty and "interval_drift_on" in totals.columns:
        lines.append("### Totals breakdown (adjustments on/off)")
        for flag, title in [
            ("interval_drift_on", "interval_drift"),
            ("recent_window_on", "recent_window"),
            ("endgame_foul_on", "endgame_foul"),
        ]:
            d_on = totals[totals[flag] == 1]
            d_off = totals[totals[flag] == 0]
            if len(d_on) + len(d_off) == 0:
                continue
            lines.append(f"- {title}: on({_fmt_metrics(_metrics(d_on))})  off({_fmt_metrics(_metrics(d_off))})")
        lines.append("")

    # New: breakdown by shrink confidence lambda.
    try:
        if not totals.empty and "edge_shrink_lambda" in totals.columns and totals["edge_shrink_lambda"].notna().any():
            t0 = totals.copy()
            lam = pd.to_numeric(t0.get("edge_shrink_lambda"), errors="coerce")
            # Bins in [0,1], plus a catchall for out-of-range.
            t0["shrink_bin"] = pd.cut(
                lam,
                bins=[-1e9, 0.25, 0.5, 0.75, 1.0, 1e9],
                labels=["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0", ">1"],
                include_lowest=True,
            )
            t0["shrink_bin"] = t0["shrink_bin"].astype(str).where(lam.notna(), other="missing")

            lines.append("### Totals breakdown (edge shrink lambda bins)")

            # Overall
            lines.append("- overall:")
            for k, g in t0.groupby("shrink_bin"):
                gg = g.dropna(subset=["pred", "act"]).copy()
                lines.append(f"  - lambda={k}: {_fmt_metrics(_metrics(gg))}")

            # Per market
            for mkt in ["total", "half_total", "quarter_total"]:
                mm = t0[t0["market"] == mkt]
                if mm.empty:
                    continue
                lines.append(f"- {mkt}:")
                for k, g in mm.groupby("shrink_bin"):
                    gg = g.dropna(subset=["pred", "act"]).copy()
                    lines.append(f"  - lambda={k}: {_fmt_metrics(_metrics(gg))}")

            lines.append("")
    except Exception:
        pass

    props = df[df["market"] == "player_prop"].copy()
    props_scored = props.dropna(subset=["act", "live_line"]).copy()
    lines.append("## Player props")
    n_props = int(len(props))
    n_props_act = int(props["act"].notna().sum()) if "act" in props.columns else 0
    n_props_pred = int(props["pred"].notna().sum()) if "pred" in props.columns else 0
    lines.append(f"- scored: {_fmt_hits(_hit_rate(props_scored))}  have_pred={n_props_pred}/{n_props}  have_act={n_props_act}/{n_props}")

    # Diagnostics: where are prop lines coming from?
    try:
        if "line_source" in props.columns and not props.empty:
            vc = (
                props["line_source"]
                .fillna("")
                .astype(str)
                .str.strip()
                .replace({"": "(missing)"})
                .value_counts()
                .to_dict()
            )
            if vc:
                parts = [f"{k}={int(v)}" for k, v in vc.items()]
                lines.append(f"- line_source: {', '.join(parts)}")
    except Exception:
        pass

    if not props_scored.empty:
        by_klass = props_scored.groupby(props_scored["klass"].fillna(""))
        lines.append("### Hit rate by klass")
        for k, g in by_klass:
            kk = k if k else "(blank)"
            lines.append(f"- {kk}: {_fmt_hits(_hit_rate(g))}")
        lines.append("")

        by_stat = props_scored.groupby(props_scored["stat_key"].fillna(""))
        lines.append("### Hit rate by stat")
        for k, g in by_stat:
            kk = k if k else "(blank)"
            lines.append(f"- {kk}: {_fmt_hits(_hit_rate(g))}")
        lines.append("")

        # Cluster diagnostics
        try:
            props_scored["min_bin"] = pd.cut(
                pd.to_numeric(props_scored.get("proj_min_final"), errors="coerce"),
                bins=[-1e9, 16, 24, 32, 1e9],
                labels=["<16", "16-24", "24-32", "32+"],
            )
        except Exception:
            props_scored["min_bin"] = None
        try:
            props_scored["edge_bin"] = pd.cut(
                pd.to_numeric(props_scored.get("strength"), errors="coerce"),
                bins=[-1e9, 2, 4, 6, 8, 1e9],
                labels=["<2", "2-4", "4-6", "6-8", "8+"],
            )
        except Exception:
            props_scored["edge_bin"] = None

        if props_scored["min_bin"].notna().any():
            lines.append("### Hit rate by projected minutes")
            for k, g in props_scored.groupby(props_scored["min_bin"].astype(str)):
                lines.append(f"- {k}: {_fmt_hits(_hit_rate(g))}")
            lines.append("")

        if props_scored["edge_bin"].notna().any():
            lines.append("### Hit rate by |edge|")
            for k, g in props_scored.groupby(props_scored["edge_bin"].astype(str)):
                lines.append(f"- {k}: {_fmt_hits(_hit_rate(g))}")
            lines.append("")

        # Loss clusters (largest volume, low hit rate)
        try:
            grp_cols = ["stat_key", "klass", "min_bin", "edge_bin"]
            gdf = props_scored.copy()
            for c in grp_cols:
                if c not in gdf.columns:
                    gdf[c] = ""
            rows = []
            for keys, g in gdf.groupby(grp_cols, dropna=False, observed=False):
                h = _hit_rate(g)
                n = int(h.get("wins", 0) + h.get("losses", 0) + h.get("pushes", 0))
                denom = int(h.get("wins", 0) + h.get("losses", 0))
                hr = float(h.get("hit_rate", float("nan")))
                if denom < 15:
                    continue
                rows.append({
                    "stat_key": keys[0] if keys[0] else "(blank)",
                    "klass": keys[1] if keys[1] else "(blank)",
                    "min_bin": str(keys[2]),
                    "edge_bin": str(keys[3]),
                    "wins": int(h.get("wins", 0)),
                    "losses": int(h.get("losses", 0)),
                    "pushes": int(h.get("pushes", 0)),
                    "hit": hr,
                    "denom": denom,
                })
            if rows:
                cdf = pd.DataFrame(rows)
                cdf = cdf.sort_values(["hit", "denom"], ascending=[True, False]).head(12)
                lines.append("### Loss clusters (low hit rate; denom>=15)")
                for _, r in cdf.iterrows():
                    lines.append(
                        f"- stat={r['stat_key']} klass={r['klass']} min={r['min_bin']} edge={r['edge_bin']}: W={int(r['wins'])} L={int(r['losses'])} P={int(r['pushes'])} hit={float(r['hit']):.3f}"
                    )
                lines.append("")
        except Exception:
            pass

        # Simple tuning suggestions (heuristics)
        suggestions: list[str] = []
        try:
            # If low-minute bin is clearly worse, suggest gating.
            if props_scored["min_bin"].notna().any():
                mins = []
                for k, g in props_scored.groupby(props_scored["min_bin"].astype(str)):
                    h = _hit_rate(g)
                    denom = int(h.get("wins", 0) + h.get("losses", 0))
                    if denom >= 30:
                        mins.append((k, float(h.get("hit_rate", float("nan"))), denom))
                if mins:
                    mins.sort(key=lambda x: x[1])
                    worst = mins[0]
                    if worst[0] in {"<16", "16-24"} and worst[1] < 0.48:
                        suggestions.append(f"Consider gating player-prop signals with proj_min_final {worst[0]} (hit={worst[1]:.3f} over {worst[2]} picks).")

            # If low-|edge| bin underperforms, suggest raising watch threshold.
            if props_scored["edge_bin"].notna().any():
                edges = []
                for k, g in props_scored.groupby(props_scored["edge_bin"].astype(str)):
                    h = _hit_rate(g)
                    denom = int(h.get("wins", 0) + h.get("losses", 0))
                    if denom >= 40:
                        edges.append((k, float(h.get("hit_rate", float("nan"))), denom))
                if edges:
                    edges.sort(key=lambda x: x[1])
                    worst = edges[0]
                    if worst[0] in {"<2", "2-4"} and worst[1] < 0.48:
                        suggestions.append(f"Consider raising player-prop WATCH threshold above {worst[0]} edges (hit={worst[1]:.3f} over {worst[2]} picks).")
        except Exception:
            suggestions = []

        if suggestions:
            lines.append("### Suggestions")
            for s in suggestions[:8]:
                lines.append(f"- {s}")
            lines.append("")

        # De-dup diagnostics: repeated ticks can inflate/deflate hit rates.
        # These views approximate "one decision per prop" policies.
        try:
            p0 = props_scored.copy()
            p0["_idx"] = range(len(p0))
            p0["_elapsed"] = pd.to_numeric(p0.get("elapsed"), errors="coerce")

            def _dedup_latest(df0: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
                d = df0.copy().sort_values(["_elapsed", "_idx"], ascending=[True, True], na_position="last")
                return d.drop_duplicates(subset=cols, keep="last")

            def _dedup_earliest(df0: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
                d = df0.copy().sort_values(["_elapsed", "_idx"], ascending=[True, True], na_position="last")
                return d.drop_duplicates(subset=cols, keep="first")

            def _dedup_max_strength(df0: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
                d = df0.copy()
                d["_abs_strength"] = pd.to_numeric(d.get("strength"), errors="coerce").abs()
                d = d.sort_values(["_abs_strength", "_idx"], ascending=[True, True], na_position="last")
                return d.drop_duplicates(subset=cols, keep="last")

            key_line = ["game_id", "name_key", "stat_key", "side", "live_line"]
            key_base = ["game_id", "name_key", "stat_key", "side"]

            a_last = _dedup_latest(p0, key_line)
            b_last = _dedup_latest(p0, key_base)
            b_first = _dedup_earliest(p0, key_base)
            b_max = _dedup_max_strength(p0, key_base)

            # Base-key: first time it becomes BET, else first seen.
            # Avoid groupby.apply (pandas deprecation) by using stable sort + head(1).
            p_sorted = p0.sort_values(["_elapsed", "_idx"], ascending=[True, True], na_position="last")
            first_any = p_sorted.groupby(key_base, dropna=False, sort=False).head(1).set_index(key_base)
            first_bet = (
                p_sorted[p_sorted.get("klass").astype(str) == "BET"]
                .groupby(key_base, dropna=False, sort=False)
                .head(1)
                .set_index(key_base)
            )
            b_first_bet = first_any.copy()
            if not first_bet.empty:
                idx = first_bet.index
                b_first_bet.loc[idx, :] = first_bet.loc[idx, :].values
            b_first_bet = b_first_bet.reset_index(drop=True)

            lines.append("### De-dup diagnostics")
            lines.append(f"- no dedup: {_fmt_hits(_hit_rate(p0))}")
            lines.append(f"- key=(game,player,stat,side,live_line) latest: {_fmt_hits(_hit_rate(a_last))}")
            lines.append(f"- key=(game,player,stat,side) latest: {_fmt_hits(_hit_rate(b_last))}")
            lines.append(f"- key=(game,player,stat,side) first seen: {_fmt_hits(_hit_rate(b_first))}")
            lines.append(f"- key=(game,player,stat,side) first BET (else first): {_fmt_hits(_hit_rate(b_first_bet))}")
            lines.append(f"- key=(game,player,stat,side) max |strength|: {_fmt_hits(_hit_rate(b_max))}")
            lines.append("")

            # Driver-tag breakdown on a decision-level view (avoids tick spam).
            try:
                tag_col = None
                if "driver_tags" in b_first_bet.columns and b_first_bet["driver_tags"].notna().any():
                    tag_col = "driver_tags"
                elif "tags" in b_first_bet.columns and b_first_bet["tags"].notna().any():
                    tag_col = "tags"
                if tag_col:
                    tdf = b_first_bet.copy()
                    tdf = tdf[tdf["result"].notna()].copy()
                    if not tdf.empty:
                        tdf["_tag"] = tdf[tag_col].apply(lambda v: v if isinstance(v, list) else ([] if pd.isna(v) else [str(v)]))
                        tdf = tdf.explode("_tag")
                        tdf = tdf[tdf["_tag"].notna()].copy()
                        tdf["_tag"] = tdf["_tag"].astype(str).str.strip()
                        tdf = tdf[tdf["_tag"].str.len() > 0].copy()

                        rows = []
                        for tag, g in tdf.groupby("_tag", dropna=False):
                            h = _hit_rate(g)
                            denom = int(h.get("wins", 0) + h.get("losses", 0))
                            if denom < 10:
                                continue
                            rows.append(
                                {
                                    "tag": str(tag),
                                    "wins": int(h.get("wins", 0)),
                                    "losses": int(h.get("losses", 0)),
                                    "pushes": int(h.get("pushes", 0)),
                                    "denom": denom,
                                    "hit": float(h.get("hit_rate", float("nan"))),
                                }
                            )

                        if rows:
                            tstats = pd.DataFrame(rows).sort_values(["denom", "hit"], ascending=[False, False]).head(18)
                            lines.append("### Driver tags (decision-level; denom>=10)")
                            for _, r in tstats.iterrows():
                                lines.append(
                                    f"- {r['tag']}: W={int(r['wins'])} L={int(r['losses'])} P={int(r['pushes'])} hit={float(r['hit']):.3f} (denom={int(r['denom'])})"
                                )
                            lines.append("")
            except Exception:
                pass
        except Exception:
            pass

        # Biggest misses (high |edge| but wrong side)
        props_scored["abs_edge"] = props_scored["edge"].abs().fillna(0.0)
        wrong = props_scored[props_scored["result"] == "loss"].sort_values("abs_edge", ascending=False).head(15)
        if not wrong.empty:
            lines.append("### Biggest misses (by |edge|)")
            for _, r in wrong.iterrows():
                lines.append(
                    f"- {r.get('game_id','')} {r.get('player','') or r.get('name_key','')} {r.get('stat_key','')} {r.get('side','')}: line={r.get('live_line')} act={r.get('act')} edge={r.get('edge')} klass={r.get('klass')}"
                )
            lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Daily Live Lens audit report")
    ap.add_argument(
        "--date",
        default=None,
        help="YYYY-MM-DD (default: yesterday, local time)",
    )
    ap.add_argument(
        "--out-dir",
        default=str(REPORTS),
        help="Output directory (default: data/processed/reports)",
    )
    ap.add_argument(
        "--include-model-lines",
        action="store_true",
        help="Include player_prop rows where line_source=model (default: excluded)",
    )
    args = ap.parse_args()

    if args.date:
        ds = _parse_date(args.date).isoformat()
    else:
        ds = ( _date.today() - timedelta(days=1) ).isoformat()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _score_day(ds, include_model_lines=bool(args.include_model_lines))

    scored_csv = out_dir / f"live_lens_scored_{ds}.csv"
    audit_md = out_dir / f"live_lens_audit_{ds}.md"

    if df is None or df.empty:
        # Still write a markdown stub so the task has a tangible artifact.
        _write_markdown(ds, pd.DataFrame(), audit_md)
        return 0

    df.to_csv(scored_csv, index=False)
    _write_markdown(ds, df, audit_md)
    print(f"Wrote: {audit_md}")
    print(f"Wrote: {scored_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
