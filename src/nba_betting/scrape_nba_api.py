from __future__ import annotations

from datetime import datetime
import time
from typing import Optional

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, boxscoresummaryv2, scoreboardv2
from nba_api.stats.library import http as nba_http
from nba_api.stats.static import teams as static_teams
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type

from .config import paths


def season_to_str(season_end_year: int) -> str:
    # NBA API season format: "2023-24"
    start = season_end_year - 1
    return f"{start}-{str(season_end_year)[-2:]}"


def fetch_games_nba_api(last_n: int = 10, rate_delay: float = 0.6, with_periods: bool = True, verbose: bool = False, max_workers: int = 1) -> pd.DataFrame:
    # Ensure fresh, browser-like headers to reduce blocks/timeouts
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
    now_year = datetime.now().year
    seasons = list(range(now_year - last_n, now_year))
    records: list[dict] = []
    # Build abbreviation -> full name map
    team_list = static_teams.get_teams()
    abbr_to_full = {t['abbreviation']: t['full_name'] for t in team_list}

    # Helper: fallback via ScoreboardV2 per-date when LeagueGameLog is blocked
    @retry(
        retry=retry_if_exception_type((requests.exceptions.RequestException, Exception)),
        wait=wait_exponential_jitter(initial=1, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _scoreboard_for_date(dt: datetime):
        # NBA expects YYYY-MM-DD
        return scoreboardv2.ScoreboardV2(game_date=dt.strftime("%Y-%m-%d"), day_offset=0, timeout=30)

    def _fetch_season_via_scoreboard(season_end_year: int) -> list[dict]:
        """Fallback: iterate dates in the season window and use ScoreboardV2 to get games.

        We filter to regular season by GAME_ID prefix '002' and season match. We also parse
        per-period line scores directly from the Scoreboard LineScore table.
        """
        start = datetime(season_end_year - 1, 10, 1)
        # Extend end into October to capture irregular-season late games (e.g., 2020 bubble)
        end = datetime(season_end_year, 10, 15)
        rows: list[dict] = []
        cur = start
        while cur <= end:
            try:
                sb = _scoreboard_for_date(cur)
                nd = sb.get_normalized_dict()
                gh = pd.DataFrame(nd.get("GameHeader", []))
                ls = pd.DataFrame(nd.get("LineScore", []))
                if not gh.empty and not ls.empty:
                    # Normalize columns we rely on
                    # Ensure upper-cased matching
                    gh_cols = {c.upper(): c for c in gh.columns}
                    ls_cols = {c.upper(): c for c in ls.columns}
                    # Required columns
                    req_gh = ["GAME_ID", "SEASON", "HOME_TEAM_ID", "VISITOR_TEAM_ID", "GAME_DATE_EST"]
                    if not all(x in gh_cols for x in req_gh):
                        # Try alternative names
                        # Some scoreboards use GAME_DATE instead of GAME_DATE_EST
                        if "GAME_DATE" in gh_cols:
                            gh_cols["GAME_DATE_EST"] = gh_cols["GAME_DATE"]
                        if not all(x in gh_cols for x in req_gh):
                            raise ValueError("Scoreboard GameHeader missing required columns")
                    # LineScore must include per-quarter
                    if not {"GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PTS"}.issubset(ls_cols.keys()) or not {"PTS_QTR1", "PTS_QTR2", "PTS_QTR3", "PTS_QTR4"}.issubset(ls_cols.keys()):
                        # Skip this day if not complete
                        cur += pd.Timedelta(days=1)
                        time.sleep(rate_delay)
                        continue
                    # Keep only regular season games for this season
                    day_games = []
                    for _, g in gh.iterrows():
                        gid = str(g[gh_cols["GAME_ID"]])
                        # Regular season games use '002' prefix
                        if not gid.startswith("002"):
                            continue
                        # Season field should match season_end_year
                        try:
                            s_val = int(g[gh_cols["SEASON"]])
                        except Exception:
                            s_val = season_end_year
                        if s_val != season_end_year:
                            continue
                        day_games.append(g)
                    if not day_games:
                        cur += pd.Timedelta(days=1)
                        time.sleep(rate_delay)
                        continue

                    ls_map = ls
                    # Build rows per game by aligning LineScore to HOME/VISITOR ids
                    for g in day_games:
                        gid = str(g[gh_cols["GAME_ID"]])
                        home_id = g[gh_cols["HOME_TEAM_ID"]]
                        vis_id = g[gh_cols["VISITOR_TEAM_ID"]]
                        game_date = pd.to_datetime(g[gh_cols["GAME_DATE_EST"]]).date()
                        try:
                            lsg = ls_map[ls_map[ls_cols["GAME_ID"]] == gid]
                            if len(lsg) < 2:
                                continue
                            # Home row
                            hrow = lsg[lsg[ls_cols["TEAM_ID"]] == home_id].iloc[0]
                            vrow = lsg[lsg[ls_cols["TEAM_ID"]] == vis_id].iloc[0]
                            habbr = str(hrow[ls_cols["TEAM_ABBREVIATION"]])
                            vabbr = str(vrow[ls_cols["TEAM_ABBREVIATION"]])
                            home_full = abbr_to_full.get(habbr, habbr)
                            visitor_full = abbr_to_full.get(vabbr, vabbr)
                            rec = {
                                "season": season_end_year,
                                "date": game_date,
                                "home_team": home_full,
                                "visitor_team": visitor_full,
                                "home_pts": int(hrow[ls_cols["PTS"]]),
                                "visitor_pts": int(vrow[ls_cols["PTS"]]),
                                "game_id": gid,
                            }
                            # Periods (quarters + OT if present)
                            for i in range(1, 5):
                                qk = f"PTS_QTR{i}"
                                rec[f"home_q{i}"] = int(hrow[ls_cols[qk]]) if pd.notna(hrow[ls_cols[qk]]) else None
                                rec[f"visitor_q{i}"] = int(vrow[ls_cols[qk]]) if pd.notna(vrow[ls_cols[qk]]) else None
                            for i in range(1, 11):
                                otk = f"PTS_OT{i}"
                                if otk in ls_cols:
                                    hv = hrow.get(ls_cols[otk])
                                    vv = vrow.get(ls_cols[otk])
                                    rec[f"home_ot{i}"] = int(hv) if pd.notna(hv) else None
                                    rec[f"visitor_ot{i}"] = int(vv) if pd.notna(vv) else None
                            # Halves
                            if rec.get("home_q1") is not None and rec.get("home_q2") is not None:
                                rec["home_h1"] = (rec["home_q1"] + rec["home_q2"]) if rec["home_q1"] is not None and rec["home_q2"] is not None else None
                            if rec.get("home_q3") is not None and rec.get("home_q4") is not None:
                                rec["home_h2"] = (rec["home_q3"] + rec["home_q4"]) if rec["home_q3"] is not None and rec["home_q4"] is not None else None
                            if rec.get("visitor_q1") is not None and rec.get("visitor_q2") is not None:
                                rec["visitor_h1"] = (rec["visitor_q1"] + rec["visitor_q2"]) if rec["visitor_q1"] is not None and rec["visitor_q2"] is not None else None
                            if rec.get("visitor_q3") is not None and rec.get("visitor_q4") is not None:
                                rec["visitor_h2"] = (rec["visitor_q3"] + rec["visitor_q4"]) if rec["visitor_q3"] is not None and rec["visitor_q4"] is not None else None
                            rows.append(rec)
                        except Exception:
                            continue
            except Exception as e:
                if verbose:
                    print(f"[fetch][scoreboard] {cur.date()} failed: {e}")
            cur += pd.Timedelta(days=1)
            time.sleep(rate_delay)
        return rows

    # 1) Pull game logs and build per-game rows
    @retry(
        retry=retry_if_exception_type((requests.exceptions.RequestException, Exception)),
        wait=wait_exponential_jitter(initial=1, max=20),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _get_leaguegamelog(season_str: str):
        return leaguegamelog.LeagueGameLog(season=season_str, season_type_all_star="Regular Season", timeout=30)
    for s in seasons:
        season_str = season_to_str(s)
        if verbose:
            print(f"[fetch] Season {s} ({season_str}) - fetching logs...")
        try:
            gl = _get_leaguegamelog(season_str)
            df = gl.get_data_frames()[0]
            df = df[["GAME_ID", "GAME_DATE", "MATCHUP", "TEAM_ABBREVIATION", "PTS"]]
            df_home = df[df["MATCHUP"].str.contains(" vs. ")].copy()
            df_away = df[df["MATCHUP"].str.contains(" @ ")].copy()
            merged = pd.merge(df_home, df_away, on=["GAME_ID", "GAME_DATE"], suffixes=("_HOME", "_AWAY"))
            for _, r in merged.iterrows():
                home_abbr = r["TEAM_ABBREVIATION_HOME"]
                away_abbr = r["TEAM_ABBREVIATION_AWAY"]
                home_full = abbr_to_full.get(home_abbr, home_abbr)
                away_full = abbr_to_full.get(away_abbr, away_abbr)
                records.append({
                    "season": s,
                    "date": pd.to_datetime(r["GAME_DATE"]).date(),
                    "home_team": home_full,
                    "visitor_team": away_full,
                    "home_pts": int(r["PTS_HOME"]),
                    "visitor_pts": int(r["PTS_AWAY"]),
                    "game_id": r["GAME_ID"],
                })
        except Exception as e:
            if verbose:
                print(f"[fetch] Season {s} logs failed: {e}")
            # Fallback: try ScoreboardV2 by date
            if verbose:
                print(f"[fetch] Season {s} - falling back to ScoreboardV2 by date...")
            try:
                sb_rows = _fetch_season_via_scoreboard(s)
                if sb_rows:
                    records.extend(sb_rows)
            except Exception as ee:
                if verbose:
                    print(f"[fetch] Season {s} scoreboard fallback failed: {ee}")
        time.sleep(rate_delay)

    out = pd.DataFrame.from_records(records)
    if out.empty:
        return out

    # 2) Optional: fetch period line scores with concurrency + resume
    if with_periods:
        existing_path_parq = paths.data_raw / "games_nba_api.parquet"
        existing_path_csv = paths.data_raw / "games_nba_api.csv"
        prev = None
        if existing_path_parq.exists():
            try:
                prev = pd.read_parquet(existing_path_parq)
            except Exception:
                prev = None
        if prev is None and existing_path_csv.exists():
            try:
                prev = pd.read_csv(existing_path_csv)
            except Exception:
                prev = None

        period_cols = [
            *(f"home_q{i}" for i in range(1,5)), *(f"visitor_q{i}" for i in range(1,5)),
            *(f"home_ot{i}" for i in range(1,11)), *(f"visitor_ot{i}" for i in range(1,11)),
            "home_h1", "home_h2", "visitor_h1", "visitor_h2",
        ]
        to_fetch_ids = out["game_id"].unique().tolist()
        if prev is not None and "game_id" in prev.columns:
            prev_small = prev[[c for c in ["game_id","home_team","visitor_team"] + list(period_cols) if c in prev.columns]].copy()
            out = out.merge(prev_small, on=["game_id","home_team","visitor_team"], how="left", suffixes=("", "_prev"))
            if "home_q1" in out.columns:
                missing_mask = out["home_q1"].isna()
                to_fetch_ids = out.loc[missing_mask, "game_id"].unique().tolist()
            else:
                to_fetch_ids = out["game_id"].unique().tolist()

        if verbose:
            print(f"[fetch] Fetching line scores for {len(to_fetch_ids)} games with max_workers={max_workers}...", flush=True)
        qcols = ["PTS_QTR1", "PTS_QTR2", "PTS_QTR3", "PTS_QTR4"]
        otcols = [f"PTS_OT{i}" for i in range(1, 11)]
        id_to_lines: dict[str, pd.DataFrame] = {}

        def _extract_linescore_df(bs: boxscoresummaryv2.BoxScoreSummaryV2) -> Optional[pd.DataFrame]:
                """Return the LineScore dataframe from the endpoint, robustly.

                Prefer the normalized dict's "LineScore" table. As a fallback, scan
                all data frames and select the one that contains PTS_QTR1.. columns.
                """
                try:
                    nd = bs.get_normalized_dict()
                    if isinstance(nd, dict) and "LineScore" in nd and nd["LineScore"]:
                        df = pd.DataFrame(nd["LineScore"]).copy()
                    else:
                        # Fallback: search among frames
                        frames = bs.get_data_frames()
                        df = None
                        for f in frames:
                            cols = set(map(str.upper, f.columns))
                            if {"PTS_QTR1", "PTS_QTR2", "PTS_QTR3", "PTS_QTR4"}.issubset(cols):
                                df = f.copy()
                                break
                        if df is None:
                            return None

                    # Normalize expected columns and select
                    # Some envs may provide lowercase/uppercase variations; use .get
                    def col(df, name):
                        for c in df.columns:
                            if c.upper() == name:
                                return c
                        return None

                    needed = ["TEAM_ABBREVIATION", *qcols, *otcols, "PTS"]
                    mapping = {n: col(df, n) for n in needed}
                    # Filter to only the columns that exist; quarters must exist
                    if not all(mapping[k] for k in ["PTS_QTR1", "PTS_QTR2", "PTS_QTR3", "PTS_QTR4", "TEAM_ABBREVIATION"]):
                        return None
                    keep_cols = [mapping[k] for k in needed if mapping.get(k)]
                    out = df[keep_cols].copy()
                    # Rename to canonical names
                    rename_map = {mapping[k]: k for k in mapping if mapping[k]}
                    out.rename(columns=rename_map, inplace=True)
                    return out
                except Exception:
                    return None

        @retry(
            retry=retry_if_exception_type((requests.exceptions.RequestException, Exception)),
            wait=wait_exponential_jitter(initial=1, max=20),
            stop=stop_after_attempt(4),
            reraise=False,
        )
        def _get_boxscore(gid: str):
            return boxscoresummaryv2.BoxScoreSummaryV2(game_id=gid, timeout=30)

        def fetch_one(gid: str):
            try:
                bs = _get_boxscore(gid)
                ls = _extract_linescore_df(bs)
                if ls is None or ls.empty:
                    return gid, None
                ls["TEAM_FULL"] = ls["TEAM_ABBREVIATION"].map(lambda a: abbr_to_full.get(a, a))
                return gid, ls
            except Exception:
                return gid, None
        progress_file = paths.data_raw / "_period_fetch_progress.txt"
        success = 0
        failures = 0
        if max_workers and max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(fetch_one, gid): gid for gid in to_fetch_ids}
                done = 0
                for fut in as_completed(futures):
                    gid, ls = fut.result()
                    if ls is not None:
                        id_to_lines[gid] = ls
                        success += 1
                    else:
                        failures += 1
                    done += 1
                    if verbose and done % 100 == 0:
                        print(f"[fetch] Processed {done}/{len(to_fetch_ids)} games for line scores...", flush=True)
                        try:
                            progress_file.write_text(f"processed={done} total={len(to_fetch_ids)} last_gid={gid}\n")
                        except Exception:
                            pass
                    time.sleep(rate_delay)
        else:
            for idx, gid in enumerate(to_fetch_ids, start=1):
                gid, ls = fetch_one(gid)
                if ls is not None:
                    id_to_lines[gid] = ls
                    success += 1
                else:
                    failures += 1
                if verbose and idx % 100 == 0:
                    print(f"[fetch] Processed {idx}/{len(to_fetch_ids)} games for line scores...", flush=True)
                    try:
                        progress_file.write_text(f"processed={idx} total={len(to_fetch_ids)} last_gid={gid}\n")
                    except Exception:
                        pass
                time.sleep(rate_delay)

        if verbose:
            print(f"[fetch] LineScore fetch complete: success={success} failures={failures}")

        def attach_periods(row):
            gid = row["game_id"]
            ls = id_to_lines.get(gid)
            if ls is None or len(ls) < 2:
                return row
            try:
                home_row = ls[ls["TEAM_FULL"] == row["home_team"]].iloc[0]
                away_row = ls[ls["TEAM_FULL"] == row["visitor_team"]].iloc[0]
            except Exception:
                return row
            for idx, q in enumerate(["q1", "q2", "q3", "q4"], start=1):
                row[f"home_{q}"] = int(home_row.get(f"PTS_QTR{idx}")) if pd.notna(home_row.get(f"PTS_QTR{idx}")) else row.get(f"home_{q}")
                row[f"visitor_{q}"] = int(away_row.get(f"PTS_QTR{idx}")) if pd.notna(away_row.get(f"PTS_QTR{idx}")) else row.get(f"visitor_{q}")
            for i in range(1, 11):
                col = f"PTS_OT{i}"
                hv = home_row.get(col)
                av = away_row.get(col)
                if pd.notna(hv):
                    row[f"home_ot{i}"] = int(hv)
                if pd.notna(av):
                    row[f"visitor_ot{i}"] = int(av)
            return row

        if len(id_to_lines) > 0:
            out = out.apply(attach_periods, axis=1)

        # Compute halves if quarters present
        for side in ("home", "visitor"):
            if f"{side}_q1" in out.columns and f"{side}_q2" in out.columns:
                out[f"{side}_h1"] = out[[f"{side}_q1", f"{side}_q2"]].sum(axis=1, min_count=1)
            if f"{side}_q3" in out.columns and f"{side}_q4" in out.columns:
                out[f"{side}_h2"] = out[[f"{side}_q3", f"{side}_q4"]].sum(axis=1, min_count=1)

    # 3) Basic targets and totals
    out["total_points"] = out[["visitor_pts", "home_pts"]].sum(axis=1)
    out["home_win"] = (out["home_pts"] > out["visitor_pts"]).astype("Int64")
    out["margin"] = out["home_pts"] - out["visitor_pts"]

    # 4) Save, merging with any existing raw to preserve other seasons
    paths.data_raw.mkdir(parents=True, exist_ok=True)
    out_csv = paths.data_raw / "games_nba_api.csv"
    out_parq = paths.data_raw / "games_nba_api.parquet"

    base = None
    if out_parq.exists():
        try:
            base = pd.read_parquet(out_parq)
        except Exception:
            base = None
    if base is None and out_csv.exists():
        try:
            base = pd.read_csv(out_csv)
        except Exception:
            base = None

    if base is not None and "game_id" in base.columns:
        base = base[~base["game_id"].isin(out["game_id"])].copy()
        out_save = pd.concat([base, out], ignore_index=True)
    else:
        out_save = out

    out_save.to_csv(out_csv, index=False)
    try:
        out_save.to_parquet(out_parq, index=False)
    except Exception:
        pass
    return out_save


def enrich_periods_existing(rate_delay: float = 0.6, verbose: bool = False, max_workers: int = 1, limit: Optional[int] = None, seasons: Optional[list[int]] = None) -> pd.DataFrame:
    """Fetch quarter/OT line scores for games in existing raw that are missing them.

    This avoids LeagueGameLog (useful when that endpoint is blocked) and only calls
    BoxScoreSummaryV2 per missing game. Respects concurrency and writes incremental
    progress to data/raw/_period_fetch_progress.txt.
    """
    # Load existing raw
    existing_path_parq = paths.data_raw / "games_nba_api.parquet"
    existing_path_csv = paths.data_raw / "games_nba_api.csv"
    if existing_path_parq.exists():
        df = pd.read_parquet(existing_path_parq)
    elif existing_path_csv.exists():
        df = pd.read_csv(existing_path_csv)
    else:
        raise FileNotFoundError("No existing raw games file found.")

    if df.empty or "game_id" not in df.columns:
        if verbose:
            print("[periods] No games to enrich.")
        return df

    # Team map for full name
    team_list = static_teams.get_teams()
    abbr_to_full = {t['abbreviation']: t['full_name'] for t in team_list}

    # Optional season filter first
    if seasons:
        df = df[df.get("season").isin(seasons)].copy()

    # Determine which games need periods
    need_cols = ["home_q1", "visitor_q1"]
    missing_mask = ~df.columns.isin(need_cols).any() or df.get("home_q1").isna() if "home_q1" in df.columns else True
    if isinstance(missing_mask, bool):
        to_fetch_ids = df["game_id"].astype(str).unique().tolist()
    else:
        to_fetch_ids = df.loc[missing_mask, "game_id"].astype(str).unique().tolist()
    if limit is not None:
        to_fetch_ids = to_fetch_ids[:int(limit)]

    if not to_fetch_ids:
        if verbose:
            print("[periods] Nothing to fetch; all games have periods.")
        return df

    if verbose:
        print(f"[periods] Fetching line scores for {len(to_fetch_ids)} games (max_workers={max_workers})", flush=True)

    qcols = ["PTS_QTR1", "PTS_QTR2", "PTS_QTR3", "PTS_QTR4"]
    otcols = [f"PTS_OT{i}" for i in range(1, 11)]

    def _extract_linescore_df(bs: boxscoresummaryv2.BoxScoreSummaryV2) -> Optional[pd.DataFrame]:
        try:
            nd = bs.get_normalized_dict()
            if isinstance(nd, dict) and "LineScore" in nd and nd["LineScore"]:
                df_ = pd.DataFrame(nd["LineScore"]).copy()
            else:
                frames = bs.get_data_frames()
                df_ = None
                for f in frames:
                    cols = set(map(str.upper, f.columns))
                    if {"PTS_QTR1", "PTS_QTR2", "PTS_QTR3", "PTS_QTR4"}.issubset(cols):
                        df_ = f.copy()
                        break
                if df_ is None:
                    return None
            def col(df__, name):
                for c in df__.columns:
                    if c.upper() == name:
                        return c
                return None
            needed = ["TEAM_ABBREVIATION", *qcols, *otcols, "PTS"]
            mapping = {n: col(df_, n) for n in needed}
            if not all(mapping[k] for k in ["PTS_QTR1", "PTS_QTR2", "PTS_QTR3", "PTS_QTR4", "TEAM_ABBREVIATION"]):
                return None
            keep_cols = [mapping[k] for k in needed if mapping.get(k)]
            out_ = df_[keep_cols].copy()
            out_.rename(columns={mapping[k]: k for k in mapping if mapping[k]}, inplace=True)
            return out_
        except Exception:
            return None

    # Retries for boxscore requests
    @retry(
        retry=retry_if_exception_type((requests.exceptions.RequestException, Exception)),
        wait=wait_exponential_jitter(initial=1, max=20),
        stop=stop_after_attempt(4),
        reraise=False,
    )
    def _get_boxscore(gid: str):
        return boxscoresummaryv2.BoxScoreSummaryV2(game_id=gid, timeout=30)

    id_to_lines: dict[str, pd.DataFrame] = {}
    progress_file = paths.data_raw / "_period_fetch_progress.txt"

    def fetch_one(gid: str):
        try:
            bs = _get_boxscore(gid)
            ls = _extract_linescore_df(bs)
            if ls is None or ls.empty:
                return gid, None
            ls["TEAM_FULL"] = ls["TEAM_ABBREVIATION"].map(lambda a: abbr_to_full.get(a, a))
            return gid, ls
        except Exception:
            return gid, None

    if max_workers and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(fetch_one, gid): gid for gid in to_fetch_ids}
            done = 0
            for fut in as_completed(futures):
                gid, ls = fut.result()
                if ls is not None:
                    id_to_lines[gid] = ls
                done += 1
                if verbose and done % 100 == 0:
                    print(f"[periods] Processed {done}/{len(to_fetch_ids)}", flush=True)
                    try:
                        progress_file.write_text(f"processed={done} total={len(to_fetch_ids)} last_gid={gid}\n")
                    except Exception:
                        pass
                time.sleep(rate_delay)
    else:
        for idx, gid in enumerate(to_fetch_ids, start=1):
            gid, ls = fetch_one(gid)
            if ls is not None:
                id_to_lines[gid] = ls
            if verbose and idx % 100 == 0:
                print(f"[periods] Processed {idx}/{len(to_fetch_ids)}", flush=True)
                try:
                    progress_file.write_text(f"processed={idx} total={len(to_fetch_ids)} last_gid={gid}\n")
                except Exception:
                    pass
            time.sleep(rate_delay)

    if not id_to_lines:
        if verbose:
            print("[periods] No line scores fetched.")
        return df

    # Attach
    def attach_periods(row):
        gid = str(row["game_id"]) if not isinstance(row["game_id"], str) else row["game_id"]
        ls = id_to_lines.get(gid)
        if ls is None or len(ls) < 2:
            return row
        try:
            home_row = ls[ls["TEAM_FULL"] == row["home_team"]].iloc[0]
            away_row = ls[ls["TEAM_FULL"] == row["visitor_team"]].iloc[0]
        except Exception:
            return row
        for idx, q in enumerate(["q1", "q2", "q3", "q4"], start=1):
            row[f"home_{q}"] = int(home_row.get(f"PTS_QTR{idx}")) if pd.notna(home_row.get(f"PTS_QTR{idx}")) else row.get(f"home_{q}")
            row[f"visitor_{q}"] = int(away_row.get(f"PTS_QTR{idx}")) if pd.notna(away_row.get(f"PTS_QTR{idx}")) else row.get(f"visitor_{q}")
        for i in range(1, 11):
            col = f"PTS_OT{i}"
            hv = home_row.get(col)
            av = away_row.get(col)
            if pd.notna(hv):
                row[f"home_ot{i}"] = int(hv)
            if pd.notna(av):
                row[f"visitor_ot{i}"] = int(av)
        return row

    df = df.apply(attach_periods, axis=1)
    # Halves
    for side in ("home", "visitor"):
        if f"{side}_q1" in df.columns and f"{side}_q2" in df.columns:
            df[f"{side}_h1"] = df[[f"{side}_q1", f"{side}_q2"]].sum(axis=1, min_count=1)
        if f"{side}_q3" in df.columns and f"{side}_q4" in df.columns:
            df[f"{side}_h2"] = df[[f"{side}_q3", f"{side}_q4"]].sum(axis=1, min_count=1)

    # Save back
    out_csv = paths.data_raw / "games_nba_api.csv"
    out_parq = paths.data_raw / "games_nba_api.parquet"
    df.to_csv(out_csv, index=False)
    try:
        df.to_parquet(out_parq, index=False)
    except Exception:
        pass
    if verbose:
        print("[periods] Enrichment saved.")
    return df


def _scoreboard_iterate_dates_for_season(season_end_year: int, rate_delay: float, verbose: bool) -> list[dict]:
    """Internal: iterate dates in a season window and extract games via ScoreboardV2.

    Returns list of per-game dicts with periods and halves included.
    """
    # Build abbreviation -> full name map
    team_list = static_teams.get_teams()
    abbr_to_full = {t['abbreviation']: t['full_name'] for t in team_list}

    @retry(
        retry=retry_if_exception_type((requests.exceptions.RequestException, Exception)),
        wait=wait_exponential_jitter(initial=1, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _scoreboard_for_date(dt: datetime):
        return scoreboardv2.ScoreboardV2(game_date=dt.strftime("%Y-%m-%d"), day_offset=0, timeout=30)

    start = datetime(season_end_year - 1, 10, 1)
    end = datetime(season_end_year, 10, 15)
    rows: list[dict] = []
    cur = start
    while cur <= end:
        try:
            sb = _scoreboard_for_date(cur)
            nd = sb.get_normalized_dict()
            gh = pd.DataFrame(nd.get("GameHeader", []))
            ls = pd.DataFrame(nd.get("LineScore", []))
            if gh.empty or ls.empty:
                cur += pd.Timedelta(days=1); time.sleep(rate_delay); continue
            gh_cols = {c.upper(): c for c in gh.columns}
            ls_cols = {c.upper(): c for c in ls.columns}
            # Map or infer needed columns
            if "GAME_DATE_EST" not in gh_cols and "GAME_DATE" in gh_cols:
                gh_cols["GAME_DATE_EST"] = gh_cols["GAME_DATE"]
            req_gh = ["GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID", "GAME_DATE_EST"]
            if not all(x in gh_cols for x in req_gh):
                cur += pd.Timedelta(days=1); time.sleep(rate_delay); continue
            if not {"GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PTS"}.issubset(ls_cols.keys()) or not {"PTS_QTR1", "PTS_QTR2", "PTS_QTR3", "PTS_QTR4"}.issubset(ls_cols.keys()):
                cur += pd.Timedelta(days=1); time.sleep(rate_delay); continue
            # Filter regular season games by id prefix 002 and by year window
            day_games = []
            for _, g in gh.iterrows():
                gid = str(g[gh_cols["GAME_ID"]])
                if not gid.startswith("002"):
                    continue
                gdate = pd.to_datetime(g[gh_cols["GAME_DATE_EST"]])
                # ensure date is within season window (covers lockout/oddities)
                if not (start <= gdate.to_pydatetime() <= end):
                    continue
                day_games.append(g)
            if not day_games:
                cur += pd.Timedelta(days=1); time.sleep(rate_delay); continue

            for g in day_games:
                gid = str(g[gh_cols["GAME_ID"]])
                home_id = g[gh_cols["HOME_TEAM_ID"]]
                vis_id = g[gh_cols["VISITOR_TEAM_ID"]]
                game_date = pd.to_datetime(g[gh_cols["GAME_DATE_EST"]]).date()
                try:
                    lsg = ls[ls[ls_cols["GAME_ID"]] == gid]
                    if len(lsg) < 2:
                        continue
                    hrow = lsg[lsg[ls_cols["TEAM_ID"]] == home_id].iloc[0]
                    vrow = lsg[lsg[ls_cols["TEAM_ID"]] == vis_id].iloc[0]
                    habbr = str(hrow[ls_cols["TEAM_ABBREVIATION"]])
                    vabbr = str(vrow[ls_cols["TEAM_ABBREVIATION"]])
                    home_full = abbr_to_full.get(habbr, habbr)
                    visitor_full = abbr_to_full.get(vabbr, vabbr)
                    rec = {
                        "season": season_end_year,
                        "date": game_date,
                        "home_team": home_full,
                        "visitor_team": visitor_full,
                        "home_pts": int(hrow[ls_cols["PTS"]]),
                        "visitor_pts": int(vrow[ls_cols["PTS"]]),
                        "game_id": gid,
                    }
                    for i in range(1, 5):
                        qk = f"PTS_QTR{i}"
                        rec[f"home_q{i}"] = int(hrow[ls_cols[qk]]) if pd.notna(hrow[ls_cols[qk]]) else None
                        rec[f"visitor_q{i}"] = int(vrow[ls_cols[qk]]) if pd.notna(vrow[ls_cols[qk]]) else None
                    for i in range(1, 11):
                        otk = f"PTS_OT{i}"
                        if otk in ls_cols:
                            hv = hrow.get(ls_cols[otk]); vv = vrow.get(ls_cols[otk])
                            rec[f"home_ot{i}"] = int(hv) if pd.notna(hv) else None
                            rec[f"visitor_ot{i}"] = int(vv) if pd.notna(vv) else None
                    # Halves
                    if rec.get("home_q1") is not None and rec.get("home_q2") is not None:
                        rec["home_h1"] = rec["home_q1"] + rec["home_q2"]
                    if rec.get("home_q3") is not None and rec.get("home_q4") is not None:
                        rec["home_h2"] = rec["home_q3"] + rec["home_q4"]
                    if rec.get("visitor_q1") is not None and rec.get("visitor_q2") is not None:
                        rec["visitor_h1"] = rec["visitor_q1"] + rec["visitor_q2"]
                    if rec.get("visitor_q3") is not None and rec.get("visitor_q4") is not None:
                        rec["visitor_h2"] = rec["visitor_q3"] + rec["visitor_q4"]
                    rows.append(rec)
                except Exception:
                    continue
        except Exception as e:
            if verbose:
                try:
                    print(f"[scoreboard] {cur.date()} failed: {e}")
                except Exception:
                    pass
        cur += pd.Timedelta(days=1)
        time.sleep(rate_delay)
    return rows


def backfill_scoreboard(seasons: list[int], rate_delay: float = 0.8, verbose: bool = False, day_limit: Optional[int] = None, resume_file: Optional[str] = None) -> pd.DataFrame:
    """Backfill games (with per-periods) for given seasons via ScoreboardV2, day-by-day with resume support.

    - seasons: list of season end years (e.g., [2018,2019])
    - rate_delay: delay between days to moderate API calls
    - day_limit: if provided, process at most this many days per season (for smoke tests)
    - resume_file: path to a JSON storing last processed date per season
    """
    import json
    # Load base raw
    existing_path_parq = paths.data_raw / "games_nba_api.parquet"
    existing_path_csv = paths.data_raw / "games_nba_api.csv"
    if existing_path_parq.exists():
        base = pd.read_parquet(existing_path_parq)
    elif existing_path_csv.exists():
        base = pd.read_csv(existing_path_csv)
    else:
        base = pd.DataFrame()

    resume = {}
    if resume_file:
        try:
            with open(resume_file, "r", encoding="utf-8") as f:
                resume = json.load(f)
        except Exception:
            resume = {}

    total_added = 0
    total_updated = 0

    # Retry wrapper for per-day scoreboard calls
    @retry(
        retry=retry_if_exception_type((requests.exceptions.RequestException, Exception)),
        wait=wait_exponential_jitter(initial=1, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _scoreboard_for_date(dt: datetime):
        return scoreboardv2.ScoreboardV2(game_date=dt.strftime("%Y-%m-%d"), day_offset=0, timeout=30)
    for s in seasons:
        # Figure dates to iterate based on resume
        start = datetime(s - 1, 10, 1)
        end = datetime(s, 6, 30)
        if str(s) in resume:
            try:
                last = datetime.fromisoformat(resume[str(s)])
                # continue from next day
                start = max(start, last + pd.Timedelta(days=1))
            except Exception:
                pass
        # Iterate and fetch using helper, but allow day limit
        rows = []
        cur = start
        processed_days = 0
        while cur <= end:
            day_rows = _scoreboard_iterate_dates_for_season(s, rate_delay=rate_delay, verbose=verbose) if False else None
            # Above helper does a full season; for per-day we inline minimal fetch for performance
            try:
                if verbose:
                    try:
                        print(f"[backfill] Season {s} date {cur.date()} ...")
                    except Exception:
                        pass
                sb = _scoreboard_for_date(cur)
                nd = sb.get_normalized_dict()
                gh = pd.DataFrame(nd.get("GameHeader", []))
                ls = pd.DataFrame(nd.get("LineScore", []))
                if gh.empty or ls.empty:
                    cur += pd.Timedelta(days=1); processed_days += 1; resume[str(s)] = cur.strftime("%Y-%m-%d");
                    if resume_file:
                        try:
                            with open(resume_file, "w", encoding="utf-8") as f:
                                json.dump(resume, f)
                        except Exception:
                            pass
                    time.sleep(rate_delay); continue
                gh_cols = {c.upper(): c for c in gh.columns}
                ls_cols = {c.upper(): c for c in ls.columns}
                if "GAME_DATE_EST" not in gh_cols and "GAME_DATE" in gh_cols:
                    gh_cols["GAME_DATE_EST"] = gh_cols["GAME_DATE"]
                req_gh = ["GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID", "GAME_DATE_EST"]
                if not all(x in gh_cols for x in req_gh) or not {"GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PTS"}.issubset(ls_cols.keys()) or not {"PTS_QTR1", "PTS_QTR2", "PTS_QTR3", "PTS_QTR4"}.issubset(ls_cols.keys()):
                    cur += pd.Timedelta(days=1); processed_days += 1; resume[str(s)] = cur.strftime("%Y-%m-%d");
                    if resume_file:
                        try:
                            with open(resume_file, "w", encoding="utf-8") as f:
                                json.dump(resume, f)
                        except Exception:
                            pass
                    time.sleep(rate_delay); continue
                # Build abbreviation -> full name map once
                team_list = static_teams.get_teams(); abbr_to_full = {t['abbreviation']: t['full_name'] for t in team_list}
                # Filter to regular season and date in window
                day_games = []
                for _, g in gh.iterrows():
                    gid = str(g[gh_cols["GAME_ID"]])
                    if not gid.startswith("002"):
                        continue
                    gdate = pd.to_datetime(g[gh_cols["GAME_DATE_EST"]])
                    if not (datetime(s - 1, 10, 1) <= gdate.to_pydatetime() <= datetime(s, 6, 30)):
                        continue
                    day_games.append(g)
                new_rows = []
                for g in day_games:
                    gid = str(g[gh_cols["GAME_ID"]])
                    home_id = g[gh_cols["HOME_TEAM_ID"]]
                    vis_id = g[gh_cols["VISITOR_TEAM_ID"]]
                    game_date = pd.to_datetime(g[gh_cols["GAME_DATE_EST"]]).date()
                    try:
                        lsg = ls[ls[ls_cols["GAME_ID"]] == gid]
                        if len(lsg) < 2:
                            continue
                        hrow = lsg[lsg[ls_cols["TEAM_ID"]] == home_id].iloc[0]
                        vrow = lsg[lsg[ls_cols["TEAM_ID"]] == vis_id].iloc[0]
                        habbr = str(hrow[ls_cols["TEAM_ABBREVIATION"]]); vabbr = str(vrow[ls_cols["TEAM_ABBREVIATION"]])
                        home_full = abbr_to_full.get(habbr, habbr); visitor_full = abbr_to_full.get(vabbr, vabbr)
                        rec = {"season": s, "date": game_date, "home_team": home_full, "visitor_team": visitor_full, "home_pts": int(hrow[ls_cols["PTS"]]), "visitor_pts": int(vrow[ls_cols["PTS"]]), "game_id": gid}
                        for i in range(1, 5):
                            qk = f"PTS_QTR{i}"; rec[f"home_q{i}"] = int(hrow[ls_cols[qk]]) if pd.notna(hrow[ls_cols[qk]]) else None; rec[f"visitor_q{i}"] = int(vrow[ls_cols[qk]]) if pd.notna(vrow[ls_cols[qk]]) else None
                        for i in range(1, 11):
                            otk = f"PTS_OT{i}"; 
                            if otk in ls_cols:
                                hv = hrow.get(ls_cols[otk]); vv = vrow.get(ls_cols[otk]); rec[f"home_ot{i}"] = int(hv) if pd.notna(hv) else None; rec[f"visitor_ot{i}"] = int(vv) if pd.notna(vv) else None
                        if rec.get("home_q1") is not None and rec.get("home_q2") is not None:
                            rec["home_h1"] = rec["home_q1"] + rec["home_q2"]
                        if rec.get("home_q3") is not None and rec.get("home_q4") is not None:
                            rec["home_h2"] = rec["home_q3"] + rec["home_q4"]
                        if rec.get("visitor_q1") is not None and rec.get("visitor_q2") is not None:
                            rec["visitor_h1"] = rec["visitor_q1"] + rec["visitor_q2"]
                        if rec.get("visitor_q3") is not None and rec.get("visitor_q4") is not None:
                            rec["visitor_h2"] = rec["visitor_q3"] + rec["visitor_q4"]
                        new_rows.append(rec)
                    except Exception:
                        continue
                # Merge into base
                if new_rows:
                    day_df = pd.DataFrame(new_rows)
                    if base is None or base.empty:
                        base = day_df
                    else:
                        # Update existing rows by game_id; otherwise append
                        if "game_id" in base.columns:
                            base = base[~base["game_id"].isin(day_df["game_id"])].copy()
                        base = pd.concat([base, day_df], ignore_index=True)
                    total_added += len(new_rows)
                    if verbose:
                        print(f"[backfill] {s} {cur.date()}: added {len(new_rows)} games")
                # Save after each day
                paths.data_raw.mkdir(parents=True, exist_ok=True)
                out_csv = paths.data_raw / "games_nba_api.csv"
                out_parq = paths.data_raw / "games_nba_api.parquet"
                base.to_csv(out_csv, index=False)
                try:
                    base.to_parquet(out_parq, index=False)
                except Exception:
                    pass
            except Exception as e:
                if verbose:
                    try:
                        print(f"[backfill] {s} {cur.date()} error: {e}")
                    except Exception:
                        pass
            # Update resume and counters
            cur += pd.Timedelta(days=1)
            processed_days += 1
            resume[str(s)] = cur.strftime("%Y-%m-%d")
            if resume_file:
                try:
                    with open(resume_file, "w", encoding="utf-8") as f:
                        json.dump(resume, f)
                except Exception:
                    pass
            time.sleep(rate_delay)
            if day_limit is not None and processed_days >= int(day_limit):
                break
        if verbose:
            print(f"[backfill] Season {s}: added ~{total_added} rows so far")

    # Recompute derived columns for safety
    if base is not None and not base.empty:
        if "visitor_pts" in base.columns and "home_pts" in base.columns:
            base["total_points"] = base[["visitor_pts", "home_pts"]].sum(axis=1)
            base["home_win"] = (base["home_pts"] > base["visitor_pts"]).astype("Int64")
            base["margin"] = base["home_pts"] - base["visitor_pts"]
    return base
