"""Validate daily produced artifacts.

This is used by scripts/daily_update.ps1 to ensure key outputs exist. Some
artifacts, such as game odds and ESPN rotations coverage, are produced via
best-effort blocks and can be downgraded to warnings by configuration.

Exit codes:
  0: OK
  3: Missing required artifacts (when fail-on-missing enabled)
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
from pathlib import Path


def _props_team_coverage(proc: Path, date_str: str) -> tuple[list[str], list[str], list[str]]:
    expected: set[str] = set()
    present: set[str] = set()

    try:
        import pandas as pd
        from nba_betting.teams import to_tricode

        odds = proc / f"game_odds_{date_str}.csv"
        if odds.exists() and odds.stat().st_size > 0:
            odf = pd.read_csv(odds)
            if odf is not None and not odf.empty:
                home_col = "home_team" if "home_team" in odf.columns else None
                away_col = "visitor_team" if "visitor_team" in odf.columns else ("away_team" if "away_team" in odf.columns else None)
                if home_col and away_col:
                    for _, row in odf.iterrows():
                        home = str(to_tricode(row.get(home_col)) or "").strip().upper()
                        away = str(to_tricode(row.get(away_col)) or "").strip().upper()
                        if home:
                            expected.add(home)
                        if away:
                            expected.add(away)

        if not expected:
            ls = proc / f"league_status_{date_str}.csv"
            if ls.exists() and ls.stat().st_size > 0:
                ldf = pd.read_csv(ls)
                if ldf is not None and not ldf.empty:
                    team_col = "team" if "team" in ldf.columns else ("team_tri" if "team_tri" in ldf.columns else None)
                    if team_col:
                        tmp = ldf.copy()
                        if "team_on_slate" in tmp.columns:
                            team_on_slate = tmp["team_on_slate"].astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})
                            try:
                                team_on_slate = team_on_slate | (pd.to_numeric(tmp["team_on_slate"], errors="coerce").fillna(0.0) > 0.5)
                            except Exception:
                                pass
                            tmp = tmp[team_on_slate].copy()
                        expected = set(
                            str(to_tricode(v) or "").strip().upper()
                            for v in tmp[team_col].tolist()
                            if str(to_tricode(v) or "").strip().upper()
                        )

        props = proc / f"props_predictions_{date_str}.csv"
        if props.exists() and props.stat().st_size > 0:
            pdf = pd.read_csv(props)
            if pdf is not None and not pdf.empty and "team" in pdf.columns:
                present = set(
                    str(to_tricode(v) or str(v or "").strip().upper()).strip().upper()
                    for v in pdf["team"].tolist()
                    if str(to_tricode(v) or str(v or "").strip().upper()).strip().upper()
                )
    except Exception:
        pass

    missing = sorted(expected - present)
    return sorted(expected), sorted(present), missing


def _exists_nonempty(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except Exception:
        return bool(path.exists())


def _count_csv_data_rows(path: Path) -> int:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header:
                return 0
            return sum(1 for _ in reader)
    except Exception:
        return 0


def _count_cards_sim_detail_games(path: Path) -> int:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return 0
        payload = json.loads(path.read_text(encoding="utf-8"))
        games = payload.get("games") if isinstance(payload, dict) else None
        if not isinstance(games, list):
            return 0
        count = 0
        for game in games:
            if not isinstance(game, dict):
                continue
            home_tri = str(game.get("home_tri") or "").strip().upper()
            away_tri = str(game.get("away_tri") or "").strip().upper()
            if home_tri and away_tri:
                count += 1
        return count
    except Exception:
        return 0


def _parse_obj(value: object) -> object:
    if isinstance(value, (list, dict)):
        return value
    s = str(value or "").strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return None


def _count_props_recommendation_play_rows(path: Path) -> int:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return 0
            plays_col = next((name for name in reader.fieldnames if str(name).strip().lower() == "plays"), None)
            if not plays_col:
                return 0
            rows = 0
            for row in reader:
                plays = _parse_obj((row or {}).get(plays_col))
                if isinstance(plays, list) and plays:
                    rows += 1
            return rows
    except Exception:
        return 0


def _detect_no_slate_day(proc: Path, date_str: str) -> tuple[bool, str | None]:
    league_status = proc / f"league_status_{date_str}.csv"
    try:
        import pandas as pd

        if league_status.exists() and league_status.stat().st_size > 0:
            ldf = pd.read_csv(league_status)
            if ldf is not None and not ldf.empty and "team_on_slate" in ldf.columns:
                slate_mask = ldf["team_on_slate"].astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})
                try:
                    slate_mask = slate_mask | (pd.to_numeric(ldf["team_on_slate"], errors="coerce").fillna(0.0) > 0.5)
                except Exception:
                    pass
                if int(slate_mask.sum()) <= 0:
                    return True, league_status.name
    except Exception:
        pass
    return False, None


def _parse_bool(v: object, default: bool = False) -> bool:
    if v is None:
        return default
    s = str(v).strip().lower()
    if s == "":
        return default
    return s in {"1", "true", "yes", "y"}


def _rotations_status(proc: Path, date_yesterday: str | None) -> tuple[int | None, int, list[str], list[str], str | None]:
    rot_dir = proc / "rotations_espn"
    rot_expected: int | None = None
    rot_have = 0
    rot_missing_gids: list[str] = []
    rot_excused_no_event_gids: list[str] = []
    rot_error: str | None = None

    if not date_yesterday:
        return rot_expected, rot_have, rot_missing_gids, rot_excused_no_event_gids, rot_error

    try:
        from nba_betting.boxscores import _nba_gid_to_tricodes

        gid_map = _nba_gid_to_tricodes(str(date_yesterday)) or {}
        gids = sorted([str(g).strip() for g in gid_map.keys() if str(g).strip()])

        # If ESPN returns no event for a game, rotations cannot be fetched.
        excused: set[str] = set()
        try:
            import csv

            failures_fp = rot_dir / f"rotations_failures_{str(date_yesterday)}.csv"
            if failures_fp.exists() and failures_fp.stat().st_size > 0:
                with failures_fp.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        gid = str((row or {}).get("game_id") or "").strip()
                        err = str((row or {}).get("error") or "").strip().lower()
                        if gid and err == "no_event":
                            excused.add(gid)
        except Exception:
            excused = set()

        rot_excused_no_event_gids = sorted(excused)
        rot_expected = len([g for g in gids if g not in excused])

        for gid in gids:
            if gid in excused:
                continue
            hp = rot_dir / f"stints_home_{gid}.csv"
            ap = rot_dir / f"stints_away_{gid}.csv"
            if _exists_nonempty(hp) and _exists_nonempty(ap):
                rot_have += 1
            else:
                rot_missing_gids.append(gid)

    except Exception as e:
        rot_error = str(e)

    return rot_expected, rot_have, rot_missing_gids, rot_excused_no_event_gids, rot_error


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=os.environ.get("REPO_ROOT", "."), help="Repo root path")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--yesterday", default=None, help="YYYY-MM-DD")

    ap.add_argument("--fail-on-missing", action="store_true", default=_parse_bool(os.environ.get("FAIL_ON_MISSING"), True))
    ap.add_argument("--require-odds", action="store_true", default=_parse_bool(os.environ.get("REQUIRE_ODDS"), True))
    ap.add_argument("--require-smartsim", action="store_true", default=_parse_bool(os.environ.get("REQUIRE_SMARTSIM"), True))
    ap.add_argument(
        "--require-props-lines",
        action="store_true",
        default=_parse_bool(os.environ.get("REQUIRE_PROPS_LINES"), False),
        help="Require props_edges and props_recommendations artifacts when props predictions are present.",
    )
    ap.add_argument(
        "--require-rotations",
        action="store_true",
        default=_parse_bool(os.environ.get("REQUIRE_ROTATIONS"), True),
        help="Require ESPN rotations stints coverage (uses a minimum coverage threshold)",
    )
    ap.add_argument(
        "--rotations-min-coverage",
        type=float,
        default=float(os.environ.get("ROTATIONS_MIN_COVERAGE", "0.70") or "0.70"),
        help="Minimum fraction of yesterday's games that must have rotations stints present",
    )

    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    proc = repo_root / "data" / "processed"

    date_str = str(args.date)
    date_yesterday = str(args.yesterday) if args.yesterday else None

    pred = proc / f"predictions_{date_str}.csv"
    props = proc / f"props_predictions_{date_str}.csv"
    odds = proc / f"game_odds_{date_str}.csv"
    props_edges = proc / f"props_edges_{date_str}.csv"
    props_recs = proc / f"props_recommendations_{date_str}.csv"
    cards_sim_detail = proc / f"cards_sim_detail_{date_str}.json"
    props_snapshot = repo_root / "data" / "raw" / f"odds_nba_player_props_{date_str}.csv"

    pred_rows = _count_csv_data_rows(pred)
    props_rows = _count_csv_data_rows(props)
    props_snapshot_rows = _count_csv_data_rows(props_snapshot)
    props_edges_rows = _count_csv_data_rows(props_edges)
    props_recs_rows = _count_csv_data_rows(props_recs)
    props_recs_play_rows = _count_props_recommendation_play_rows(props_recs)
    cards_sim_detail_games = _count_cards_sim_detail_games(cards_sim_detail)
    props_expected_teams, props_present_teams, props_missing_teams = _props_team_coverage(proc, date_str)
    no_slate_day, no_slate_source = _detect_no_slate_day(proc, date_str)

    slate_games: int | None = None
    try:
        import pandas as pd

        if _exists_nonempty(pred):
            df = pd.read_csv(pred)
            if df is not None and not df.empty:
                if {"home_team", "visitor_team"}.issubset(set(df.columns)):
                    slate_games = int(df[["home_team", "visitor_team"]].drop_duplicates().shape[0])
                else:
                    slate_games = int(len(df))
    except Exception:
        slate_games = None

    smart = sorted(proc.glob(f"smart_sim_{date_str}_*.json"))
    smart_count = len(smart)

    rot_expected, rot_have, rot_missing_gids, rot_excused_no_event_gids, rot_error = _rotations_status(proc, date_yesterday)

    missing: list[str] = []
    warnings: list[str] = []

    # --- Sanity checks: minutes vs player stats (props predictions) ---
    try:
        import pandas as pd
        import numpy as np

        if _exists_nonempty(props):
            dfp = pd.read_csv(props)
            if dfp is not None and not dfp.empty:
                # Choose the best available minutes signal.
                min_candidates = [c for c in ["pred_min", "roll10_min", "roll5_min", "roll3_min", "lag1_min"] if c in dfp.columns]
                min_col = min_candidates[0] if min_candidates else None
                # Prefer the most-populated column.
                if min_candidates:
                    best = None
                    best_n = -1
                    best_sum = -1.0
                    for c in min_candidates:
                        v = pd.to_numeric(dfp[c], errors="coerce")
                        ok = v[np.isfinite(v) & (v > 0)]
                        n_ok = int(ok.shape[0])
                        s_ok = float(ok.sum()) if n_ok > 0 else 0.0
                        if (n_ok > best_n) or (n_ok == best_n and s_ok > best_sum):
                            best = c
                            best_n = n_ok
                            best_sum = s_ok
                    min_col = best or min_col

                if min_col:
                    mins = pd.to_numeric(dfp[min_col], errors="coerce").fillna(0.0).astype(float)
                    dfp = dfp.copy()
                    dfp["__min__"] = mins.clip(lower=0.0, upper=48.0)

                    # Restrict to players expected on slate if present.
                    if "team_on_slate" in dfp.columns:
                        try:
                            tos = dfp["team_on_slate"].astype(str).str.lower().str.strip()
                            dfp = dfp[tos.isin(["true", "1", "yes", "y"])].copy()
                        except Exception:
                            pass
                    if "playing_today" in dfp.columns:
                        try:
                            pt = dfp["playing_today"].astype(str).str.lower().str.strip()
                            dfp = dfp[~pt.isin(["false", "0", "no", "n"])].copy()
                        except Exception:
                            pass

                    # Team-level minutes coverage: top-10 should roughly cover a full game.
                    if "team" in dfp.columns and not dfp.empty:
                        tmp = dfp.copy()
                        tmp["team"] = tmp["team"].astype(str).str.upper().str.strip()
                        grp = tmp.groupby("team", as_index=False)
                        for _, g in grp:
                            team = str(g["team"].iloc[0] or "").strip()
                            top = g.sort_values("__min__", ascending=False).head(10)
                            top_sum = float(top["__min__"].sum())
                            starters20 = int((top["__min__"] >= 20.0).sum())
                            if top_sum < 180.0:
                                warnings.append(f"minutes_sanity: team {team} top10 minutes sum low ({top_sum:.1f}) using {min_col}")
                            if starters20 < 5:
                                warnings.append(f"minutes_sanity: team {team} has <5 players at 20+ min (found {starters20}) using {min_col}")

                    # Player-level: suspicious combos.
                    if "pred_pts" in dfp.columns:
                        pts = pd.to_numeric(dfp["pred_pts"], errors="coerce").fillna(0.0).astype(float)
                        dfp["__pts__"] = pts
                        bad = dfp[(dfp["__min__"] > 0) & (dfp["__min__"] < 8.0) & (dfp["__pts__"] >= 10.0)]
                        if not bad.empty:
                            # only report a small sample
                            names = []
                            for _, r in bad.head(6).iterrows():
                                nm = str(r.get("player_name") or "").strip()
                                team = str(r.get("team") or "").strip()
                                names.append(f"{nm}({team})")
                            warnings.append(f"minutes_sanity: {len(bad)} players have <8 min but >=10 pred_pts (sample: {', '.join(names)})")
    except Exception:
        # Best-effort: do not fail daily artifacts validation.
        pass

    props_team_gap_is_publish_blocking = bool(
        props_missing_teams
        and not (props_snapshot_rows > 0 and props_edges_rows > 0 and props_recs_rows > 0)
    )

    if pred_rows <= 0 and not no_slate_day:
        missing.append(pred.name)
    elif pred_rows <= 0 and no_slate_day:
        warnings.append(f"no-slate day detected from {no_slate_source}; allowing missing predictions artifact: {pred.name}")
    if props_rows <= 0:
        missing.append(props.name)
    elif props_team_gap_is_publish_blocking:
        missing.append(f"props_predictions missing slate teams: {', '.join(props_missing_teams)}")
    elif props_missing_teams:
        warnings.append(
            f"props_predictions missing slate teams but downstream props artifacts are present: {', '.join(props_missing_teams)}"
        )

    if args.require_props_lines and props_rows > 0 and props_snapshot_rows > 0:
        if props_edges_rows <= 0:
            missing.append(props_edges.name)
        if props_recs_rows <= 0:
            missing.append(props_recs.name)
    elif props_edges_rows > 0 and props_recs_rows <= 0:
        warnings.append(f"props edges present but recommendations missing: {props_recs.name}")

    if props_recs_rows > 0 and props_recs_play_rows <= 0:
        if props_edges_rows > 0:
            warnings.append(f"props_recommendations has zero playable plays: {props_recs.name}")
        else:
            warnings.append(f"props_recommendations exists without line-bearing plays while props_edges is missing: {props_recs.name}")

    odds_ok = _exists_nonempty(odds)
    if not odds_ok:
        if args.require_odds:
            missing.append(odds.name)
        else:
            warnings.append(f"optional artifact missing: {odds.name}")

    if args.require_smartsim and slate_games is not None and slate_games > 0 and smart_count < max(1, slate_games):
        missing.append(f"smart_sim_{date_str}_*.json ({smart_count}/{slate_games})")
    if args.require_smartsim and slate_games is not None and slate_games > 0 and cards_sim_detail_games < max(1, slate_games):
        missing.append(f"cards_sim_detail_{date_str}.json ({cards_sim_detail_games}/{slate_games})")
    elif smart_count > 0 and cards_sim_detail_games <= 0:
        warnings.append(f"cards sim detail snapshot missing or empty: {cards_sim_detail.name}")

    cov = None
    if rot_expected is None:
        msg = "rotations_espn stints (could not determine game ids)"
        if args.require_rotations:
            missing.append(msg)
        elif date_yesterday:
            warnings.append(f"optional artifact unavailable: {msg}")
    elif rot_expected > 0:
        cov = float(rot_have) / float(rot_expected)
        thr = float(args.rotations_min_coverage)
        thr = max(0.0, min(1.0, thr))
        if cov + 1e-12 < thr:
            msg = f"rotations_espn stints ({rot_have}/{rot_expected})"
            if args.require_rotations:
                missing.append(msg)
            else:
                warnings.append(f"optional artifact partial: {msg}")
        elif rot_have < rot_expected:
            warnings.append(f"rotations_espn stints partial ({rot_have}/{rot_expected})")

    out = {
        "date": date_str,
        "yesterday": date_yesterday,
        "predictions_ok": _exists_nonempty(pred),
        "props_predictions_ok": _exists_nonempty(props),
        "game_odds_ok": _exists_nonempty(odds),
        "props_snapshot_ok": props_snapshot_rows > 0,
        "props_snapshot_rows": props_snapshot_rows,
        "props_edges_ok": props_edges_rows > 0,
        "props_edges_rows": props_edges_rows,
        "props_recommendations_ok": props_recs_rows > 0,
        "props_recommendations_rows": props_recs_rows,
        "props_recommendations_play_rows": props_recs_play_rows,
        "cards_sim_detail_ok": cards_sim_detail_games > 0,
        "cards_sim_detail_games": cards_sim_detail_games,
        "props_expected_teams": props_expected_teams,
        "props_present_teams": props_present_teams,
        "props_missing_teams": props_missing_teams,
        "no_slate_day": no_slate_day,
        "no_slate_source": no_slate_source,
        "slate_games": slate_games,
        "smart_sim_count": smart_count,
        "rotations_dir_exists": bool((proc / "rotations_espn").exists()),
        "rotations_expected_games_yesterday": rot_expected,
        "rotations_have_games_yesterday": rot_have,
        "rotations_min_coverage": float(args.rotations_min_coverage),
        "rotations_coverage_yesterday": cov,
        "rotations_missing_game_ids_yesterday": rot_missing_gids[:50],
        "rotations_excused_no_event_game_ids_yesterday": rot_excused_no_event_gids[:50],
        "rotations_error": rot_error,
        "require_odds": bool(args.require_odds),
        "require_smartsim": bool(args.require_smartsim),
        "require_props_lines": bool(args.require_props_lines),
        "require_rotations_espn": bool(args.require_rotations),
        "warnings": warnings,
        "missing": missing,
        "ok": (len(missing) == 0),
    }

    try:
        fp = proc / f"daily_artifacts_{date_str}.json"
        fp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    except Exception:
        pass

    print(json.dumps(out, indent=2))

    if args.fail_on_missing and missing:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
