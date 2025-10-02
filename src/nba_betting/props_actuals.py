from __future__ import annotations

import subprocess
import sys
import os
import shutil
from pathlib import Path
import pandas as pd
from .config import paths
from datetime import datetime as _dt
import time as _time
import requests


def fetch_prop_actuals_via_nba_cdn(date: str) -> pd.DataFrame:
    """Fetch player actuals using NBA's public liveData CDN boxscore endpoint.

    This avoids stats.nba.com and R dependencies. It uses our schedule fetcher to
    locate game_ids for the requested date, then pulls:
    https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gameId}.json

    Returns: DataFrame with columns [date, game_id, player_id, player_name, team_abbr, pts, reb, ast, threes, pra]
    """
    # Validate date
    try:
        d = _dt.strptime(date, "%Y-%m-%d").date()
    except Exception:
        raise ValueError("Invalid date; expected YYYY-MM-DD")
    # Fetch schedule and select games on this date (EST or UTC)
    try:
        from .schedule import fetch_schedule_2025_26 as _fetch_schedule
        sched = _fetch_schedule()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch schedule: {e}")
    if sched is None or sched.empty:
        return pd.DataFrame()
    sched = sched.copy()
    if "date_est" in sched.columns:
        sched["date_est"] = pd.to_datetime(sched["date_est"], errors="coerce").dt.date
    if "date_utc" in sched.columns:
        sched["date_utc"] = pd.to_datetime(sched["date_utc"], errors="coerce").dt.date
    mask = (sched.get("date_est").eq(d) if "date_est" in sched.columns else False) | (sched.get("date_utc").eq(d) if "date_utc" in sched.columns else False)
    day = sched[mask]
    if day.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for _, g in day.iterrows():
        gid = str(g.get("game_id"))
        if not gid or gid == "None":
            continue
        url = f"https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code != 200:
                continue
            js = r.json()
        except Exception:
            continue
        game = (js or {}).get("game") or {}
        # Teams
        for side in ("homeTeam", "awayTeam"):
            t = game.get(side) or {}
            tri = t.get("teamTricode")
            players = t.get("players") or []
            for p in players:
                pid = p.get("personId")
                name = p.get("name") or p.get("nameI")
                st = p.get("statistics") or {}
                # Capture basic totals
                pts = st.get("points")
                reb = st.get("reboundsTotal")
                ast = st.get("assists")
                threes = st.get("threePointersMade")
                # Some players may have no stats (DNP); skip if all null/zero
                vals = [pts, reb, ast, threes]
                if all(v in (None, 0) for v in vals):
                    continue
                # Coerce to float
                def _f(x):
                    try:
                        return float(x)
                    except Exception:
                        return None
                fpts, freb, fast, f3 = map(_f, (pts, reb, ast, threes))
                pra = sum(v for v in (fpts or 0.0, freb or 0.0, fast or 0.0))
                rows.append({
                    "date": date,
                    "game_id": gid,
                    "player_id": int(pid) if pid is not None and str(pid).isdigit() else None,
                    "player_name": name,
                    "team_abbr": tri,
                    "pts": fpts,
                    "reb": freb,
                    "ast": fast,
                    "threes": f3,
                    "pra": float(pra),
                })
    return pd.DataFrame(rows)


def fetch_prop_actuals_via_nbaapi(date: str) -> pd.DataFrame:
    """Fetch player actuals (PTS, REB, AST, 3PM, PRA) for a date using nba_api.

    Tries ScoreboardV2 for the given date, with cross-day tolerance (-1, +1) to
    account for timezone shifts in preseason/intl games. If a recon_games CSV for
    the date exists, restrict results to those teams to avoid picking up nearby slates.

    Returns a DataFrame with at least: date, game_id, player_id, player_name, team_abbr,
    pts, reb, ast, threes, pra
    """
    try:
        from nba_api.stats.endpoints import scoreboardv2 as _scoreboardv2  # type: ignore
        from nba_api.stats.endpoints import boxscoretraditionalv3 as _boxscoretraditionalv3  # type: ignore
        from nba_api.stats.library import http as _nba_http  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"nba_api not available: {e}")
    # Harden headers
    try:
        _nba_http.STATS_HEADERS.update({
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Origin': 'https://www.nba.com',
            'Referer': 'https://www.nba.com/stats/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
            'Connection': 'keep-alive',
        })
    except Exception:
        pass
    # Validate date
    try:
        _ = _dt.strptime(date, "%Y-%m-%d")
    except Exception:
        raise ValueError("Invalid date; expected YYYY-MM-DD")
    # Optional team restriction from recon file
    slate_teams: set[str] = set()
    try:
        recon_path = paths.data_processed / f"recon_games_{date}.csv"
        if recon_path.exists():
            _df_rg = pd.read_csv(recon_path)
            for col in ("home_tri", "away_tri", "visitor_tri", "visitor_team", "home_team"):
                if col in _df_rg.columns:
                    vals = _df_rg[col].dropna().astype(str).str.upper().str.strip().tolist()
                    # Normalize some common name->tri if needed
                    for v in vals:
                        if len(v) <= 4:
                            slate_teams.add(v)
            # Best-effort: also parse tri codes from team names if they look like TRI
    except Exception:
        pass

    # Get slate games and game IDs. Try date, then +/-1 day for preseason/intl quirks.
    gh = pd.DataFrame()
    for offset in (0, -1, 1):
        tries = 0
        while tries < 3:
            try:
                sb = _scoreboardv2.ScoreboardV2(game_date=date, day_offset=offset, timeout=65)
                nd = sb.get_normalized_dict()
                gh = pd.DataFrame(nd.get("GameHeader", []))
                if not gh.empty:
                    break
            except Exception:
                _time.sleep(2.5)
            finally:
                tries += 1
        if not gh.empty:
            break
    if gh is None or gh.empty:
        return pd.DataFrame()
    c = {x.upper(): x for x in gh.columns}
    if "GAME_ID" not in c:
        return pd.DataFrame()
    game_ids = [str(g[c["GAME_ID"]]) for _, g in gh.iterrows() if pd.notna(g.get(c["GAME_ID"]))]
    rows: list[dict] = []
    for gid in game_ids:
        # Fetch box score for each game
        tries = 0
        while tries < 3:
            try:
                bs = _boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=gid, timeout=65)
                nd = bs.get_normalized_dict()
                players = pd.DataFrame(nd.get("PlayerStats", []))
                if players is None or players.empty:
                    break
                pc = {x.upper(): x for x in players.columns}
                for _, r in players.iterrows():
                    try:
                        pid = int(r[pc["PLAYER_ID"]]) if "PLAYER_ID" in pc else None
                        name = str(r[pc["PLAYER_NAME"]]) if "PLAYER_NAME" in pc else None
                        tri = str(r[pc["TEAM_ABBREVIATION"]]) if "TEAM_ABBREVIATION" in pc else None
                        pts = pd.to_numeric(r.get(pc.get("PTS","PTS")), errors="coerce")
                        reb = pd.to_numeric(r.get(pc.get("REB","REB")), errors="coerce")
                        ast = pd.to_numeric(r.get(pc.get("AST","AST")), errors="coerce")
                        threes = pd.to_numeric(r.get(pc.get("FG3M","FG3M")), errors="coerce")
                        if pd.isna(pts) and pd.isna(reb) and pd.isna(ast) and pd.isna(threes):
                            continue
                        # If we know the slate teams, skip others
                        if slate_teams and tri and str(tri).upper() not in slate_teams:
                            continue
                        pra = (0 if pd.isna(pts) else float(pts)) + (0 if pd.isna(reb) else float(reb)) + (0 if pd.isna(ast) else float(ast))
                        rows.append({
                            "date": date,
                            "game_id": gid,
                            "player_id": pid,
                            "player_name": name,
                            "team_abbr": tri,
                            "pts": None if pd.isna(pts) else float(pts),
                            "reb": None if pd.isna(reb) else float(reb),
                            "ast": None if pd.isna(ast) else float(ast),
                            "threes": None if pd.isna(threes) else float(threes),
                            "pra": float(pra),
                        })
                    except Exception:
                        continue
                break
            except Exception:
                _time.sleep(2.5)
            finally:
                tries += 1
    return pd.DataFrame(rows)


def ensure_rscript() -> str:
    """Return path to Rscript executable or raise a helpful error.

    Checks RSCRIPT_PATH env, then PATH via shutil.which for Rscript/Rscript.exe.
    """
    # Env override
    env_path = os.environ.get("RSCRIPT_PATH") or os.environ.get("RSCRIPT")
    if env_path and Path(env_path).exists():
        return str(env_path)
    # PATH lookup
    cand = shutil.which("Rscript.exe" if sys.platform == "win32" else "Rscript")
    if cand:
        return cand
    raise FileNotFoundError(
        "Rscript not found. Install R and ensure Rscript is on PATH, or set RSCRIPT_PATH env to the full path to Rscript.exe.\n"
        "Windows example path: C\\\Program Files\\R\\R-4.x.x\\bin\\Rscript.exe"
    )


def fetch_prop_actuals_via_nbastatr(date: str | None = None, start: str | None = None, end: str | None = None, out: Path | None = None, verbose: bool = True) -> pd.DataFrame:
    """Call the R script to fetch player actuals and return a DataFrame.

    Either provide date (YYYY-MM-DD) or start/end.
    """
    if (date is None) == (start is None or end is None):
        raise ValueError("Provide either date or both start and end")
    script = paths.root / "scripts" / "nbastatr_fetch_prop_actuals.R"
    if not script.exists():
        raise FileNotFoundError(f"R script not found at {script}")
    exe = ensure_rscript()
    args = [exe, str(script)]
    if date:
        args += ["--date", date]
    else:
        args += ["--start", start, "--end", end]
    tmp_out = out if out else (paths.data_processed / (f"props_actuals_{date}.csv" if date else f"props_actuals_{start}_{end}.csv"))
    args += ["--out", str(tmp_out)]
    # Run
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        # If nbastatR missing, the script returns status 2 with message
        raise RuntimeError(f"Rscript failed (code {proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    # Load CSV if created
    if not Path(tmp_out).exists():
        return pd.DataFrame()
    df = pd.read_csv(tmp_out)
    return df


def upsert_props_actuals(df: pd.DataFrame) -> Path:
    """Append/dedupe into data/processed/props_actuals.parquet & CSV keyed by (date, game_id, player_id)."""
    out_csv = paths.data_processed / "props_actuals.csv"
    out_parq = paths.data_processed / "props_actuals.parquet"
    key_cols = ["date", "game_id", "player_id"]
    for c in key_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column {c}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    existing = None
    if out_parq.exists():
        try:
            existing = pd.read_parquet(out_parq)
        except Exception:
            existing = None
    if existing is None and out_csv.exists():
        try:
            existing = pd.read_csv(out_csv)
            existing["date"] = pd.to_datetime(existing["date"]).dt.date
        except Exception:
            existing = None
    if existing is not None and not existing.empty:
        # Deduplicate by key, prefer new rows
        existing["_key"] = existing[key_cols].astype(str).agg("|".join, axis=1)
        df["_key"] = df[key_cols].astype(str).agg("|".join, axis=1)
        keep_existing = existing[~existing["_key"].isin(df["_key"])]
        out = pd.concat([keep_existing.drop(columns=["_key"]), df.drop(columns=["_key"])], ignore_index=True)
    else:
        out = df
    out.to_csv(out_csv, index=False)
    try:
        out.to_parquet(out_parq, index=False)
    except Exception:
        pass
    return out_parq
