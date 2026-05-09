from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import date as dt_date
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app as app_module


PROCESSED = ROOT / "data" / "processed"
DEFAULT_BASE_URL = "https://nba-betting-5qgf.onrender.com"


def _parse_date(value: str) -> dt_date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        normalized = text.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is not None:
            return parsed.astimezone().replace(tzinfo=None)
        return parsed
    except Exception:
        return None


def _daterange(start: dt_date, end: dt_date) -> Iterable[dt_date]:
    current = start
    while current <= end:
        yield current
        current = current + timedelta(days=1)


def _american_profit(price: float, win: bool) -> float:
    if not win:
        return -1.0
    if price > 0:
        return float(price) / 100.0
    return 100.0 / abs(float(price)) if float(price) != 0 else 0.0


def _settle_over_under(actual: float, line: float, side: str) -> tuple[str, bool | None]:
    pick_side = str(side or "").strip().upper()
    if pick_side not in {"OVER", "UNDER"}:
        return "", None
    if float(actual) == float(line):
        return "PUSH", None
    if pick_side == "OVER":
        return ("WIN", True) if float(actual) > float(line) else ("LOSS", False)
    return ("WIN", True) if float(actual) < float(line) else ("LOSS", False)


def _norm_player_name(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if "(" in text:
        text = text.split("(", 1)[0]
    text = text.replace("-", " ")
    text = text.replace(".", "").replace("'", "").replace(",", " ").strip()
    for suffix in [" JR", " SR", " II", " III", " IV"]:
        if text.upper().endswith(suffix):
            text = text[: -len(suffix)]
    try:
        import unicodedata as _ud

        text = _ud.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("ascii")
    except Exception:
        pass
    return text.upper().strip()


def _prep_recon_props(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "player_name" in out.columns:
        out["name_key"] = out["player_name"].map(_norm_player_name)
    else:
        out["name_key"] = ""
    team_col = "team_abbr" if "team_abbr" in out.columns else ("team" if "team" in out.columns else None)
    if team_col is not None:
        out["team_tri"] = out[team_col].astype(str).str.upper().str.strip()
    else:
        out["team_tri"] = ""
    for column in ("pts", "reb", "ast", "threes", "stl", "blk", "tov", "pra"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    if all(column in out.columns for column in ("pts", "reb")) and "pr" not in out.columns:
        out["pr"] = out["pts"] + out["reb"]
    if all(column in out.columns for column in ("pts", "ast")) and "pa" not in out.columns:
        out["pa"] = out["pts"] + out["ast"]
    if all(column in out.columns for column in ("reb", "ast")) and "ra" not in out.columns:
        out["ra"] = out["reb"] + out["ast"]
    return out


def _load_recon_lookup(date_str: str) -> dict[tuple[str, str], dict[str, Any]]:
    candidates = [
        PROCESSED / f"recon_props_{date_str}.csv",
        PROCESSED / f"props_actuals_{date_str}.csv",
    ]
    prepared = pd.DataFrame()
    for path in candidates:
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        prepared = _prep_recon_props(frame)
        if not prepared.empty:
            break
    if prepared.empty:
        return {}
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in prepared.iterrows():
        team_tri = str(row.get("team_tri") or "").upper().strip()
        name_key = str(row.get("name_key") or "").upper().strip()
        if not team_tri or not name_key:
            continue
        lookup[(team_tri, name_key)] = row.to_dict()
    return lookup


def _load_pregame_lines(date_str: str) -> dict[tuple[str, str, str], float]:
    path = PROCESSED / f"props_edges_{date_str}.csv"
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except Exception:
        return {}
    required = {"team", "player_name", "stat", "line"}
    if frame.empty or not required.issubset(frame.columns):
        return {}
    working = frame.copy()
    working["team_tri"] = working["team"].astype(str).str.upper().str.strip()
    working["name_key"] = working["player_name"].map(_norm_player_name)
    working["stat_key"] = working["stat"].map(app_module._live_stat_key)
    working["line"] = pd.to_numeric(working["line"], errors="coerce")
    working = working.dropna(subset=["line"])
    if working.empty:
        return {}
    grouped = (
        working.groupby(["team_tri", "name_key", "stat_key"], as_index=False)
        .agg(line=("line", "median"))
    )
    return {
        (str(row["team_tri"]), str(row["name_key"]), str(row["stat_key"])): float(row["line"])
        for _, row in grouped.iterrows()
    }


def _fetch_signal_rows(base_url: str, date_str: str, timeout: float) -> list[dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/api/download_live_lens_signals?date={date_str}"
    response = requests.get(url, timeout=timeout)
    if response.status_code == 404:
        return []
    response.raise_for_status()
    rows: list[dict[str, Any]] = []
    for raw_line in response.text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _elapsed_minutes(row: dict[str, Any]) -> float | None:
    elapsed = app_module._safe_float(row.get("elapsed"))
    if elapsed is not None:
        return float(elapsed)
    period = app_module._safe_float(row.get("period"))
    sec_left = app_module._safe_float(row.get("sec_left_period"))
    if period is None or sec_left is None:
        return None
    try:
        period_index = max(1, int(period)) - 1
        elapsed_prior = 12.0 * float(period_index)
        elapsed_this_period = 12.0 - (float(sec_left) / 60.0)
        return float(max(0.0, min(48.0, elapsed_prior + elapsed_this_period)))
    except Exception:
        return None


def _price_for_side(row: dict[str, Any], side: str) -> float | None:
    context = row.get("context") if isinstance(row.get("context"), dict) else {}
    if str(side).upper() == "OVER":
        return app_module._safe_float(context.get("price_over"))
    return app_module._safe_float(context.get("price_under"))


@dataclass
class SweepRow:
    date: str
    signal_group: str
    player: str
    team_tri: str
    stat: str
    side: str
    klass: str
    line: float
    actual_final: float
    price: float
    outcome: str
    profit_u: float
    base_score: float
    shape_score: float
    win_prob: float | None
    implied_prob: float | None
    received_at: datetime | None
    received_at_text: str | None
    line_live_age_sec: float | None
    line_live_n: float | None
    minutes_since_first_bet: float | None = None


def _signal_group_key(obj: dict[str, Any], team_tri: str, name_key: str, stat_key: str, side: str) -> str:
    game_id = str(obj.get("game_id_canon") or obj.get("game_id") or "").strip()
    return "|".join([
        str(obj.get("date") or "").strip(),
        game_id,
        team_tri,
        name_key,
        stat_key,
        side,
    ])


def _row_sort_key(row: SweepRow) -> tuple[Any, ...]:
    return (
        row.received_at or datetime.max,
        float(row.line_live_n) if row.line_live_n is not None else math.inf,
        float(row.line_live_age_sec) if row.line_live_age_sec is not None else math.inf,
        row.player,
        row.stat,
        row.side,
    )


def _dedupe_signal_groups(rows: list[SweepRow], *, min_minutes_since_first_bet: float | None = None) -> list[SweepRow]:
    by_group: dict[str, SweepRow] = {}
    for row in sorted(rows, key=_row_sort_key):
        if row.klass != "BET":
            continue
        if min_minutes_since_first_bet is not None:
            minutes = row.minutes_since_first_bet
            if minutes is None or float(minutes) < float(min_minutes_since_first_bet):
                continue
        if row.signal_group not in by_group:
            by_group[row.signal_group] = row
    return list(by_group.values())


def _select_rows_near_target_minutes(rows: list[SweepRow], *, target_minutes: float) -> list[SweepRow]:
    by_group: dict[str, list[SweepRow]] = {}
    for row in rows:
        if row.klass != "BET":
            continue
        minutes = row.minutes_since_first_bet
        if minutes is None:
            continue
        by_group.setdefault(row.signal_group, []).append(row)

    selected: list[SweepRow] = []
    for group_rows in by_group.values():
        eligible = [row for row in group_rows if row.minutes_since_first_bet is not None and float(row.minutes_since_first_bet) >= float(target_minutes)]
        if not eligible:
            continue
        chosen = min(
            eligible,
            key=lambda row: (
                abs(float(row.minutes_since_first_bet or 0.0) - float(target_minutes)),
                _row_sort_key(row),
            ),
        )
        selected.append(chosen)
    return selected


def _annotate_first_bet_timing(rows: list[SweepRow]) -> list[SweepRow]:
    first_bet_at: dict[str, datetime] = {}
    for row in sorted(rows, key=_row_sort_key):
        if row.klass != "BET" or row.received_at is None:
            continue
        first_bet_at.setdefault(row.signal_group, row.received_at)

    annotated: list[SweepRow] = []
    for row in rows:
        first_dt = first_bet_at.get(row.signal_group)
        minutes_since_first_bet = None
        if first_dt is not None and row.received_at is not None:
            minutes_since_first_bet = max(0.0, (row.received_at - first_dt).total_seconds() / 60.0)
        annotated.append(
            SweepRow(
                **{
                    **row.__dict__,
                    "minutes_since_first_bet": minutes_since_first_bet,
                }
            )
        )
    return annotated


def _build_sweep_rows(date_str: str, raw_rows: list[dict[str, Any]]) -> list[SweepRow]:
    recon_lookup = _load_recon_lookup(date_str)
    pregame_lines = _load_pregame_lines(date_str)
    if not recon_lookup:
        return []
    rows: list[SweepRow] = []
    for obj in raw_rows:
        if str(obj.get("market") or "").strip().lower() != "player_prop":
            continue
        if str(obj.get("horizon") or "").strip().lower() not in {"", "live"}:
            continue
        if str(obj.get("klass") or "").strip().upper() not in {"BET", "WATCH"}:
            continue
        team_tri = str(obj.get("team_tri") or "").upper().strip()
        name_key = str(obj.get("name_key") or _norm_player_name(obj.get("player"))).upper().strip()
        if not team_tri or not name_key:
            continue
        recon = recon_lookup.get((team_tri, name_key))
        if not recon:
            continue
        stat_key = app_module._live_stat_key(obj.get("stat"))
        if stat_key not in recon or pd.isna(recon.get(stat_key)):
            continue
        line = app_module._safe_float(obj.get("line"))
        if line is None:
            continue
        actual_final = app_module._safe_float(recon.get(stat_key))
        if actual_final is None:
            continue
        side = str(obj.get("side") or "").upper().strip()
        klass = str(obj.get("klass") or "").upper().strip()
        price = _price_for_side(obj, side)
        if price is None:
            price = -110.0
        outcome, win = _settle_over_under(float(actual_final), float(line), side)
        if outcome not in {"WIN", "LOSS", "PUSH"}:
            continue
        profit_u = 0.0 if outcome == "PUSH" else _american_profit(float(price), bool(win))

        context = obj.get("context") if isinstance(obj.get("context"), dict) else {}
        sim_gap = app_module._safe_float(context.get("sim_vs_line_adjusted"))
        if sim_gap is None:
            sim_gap = app_module._safe_float(context.get("sim_vs_line"))
        pace_proj = app_module._safe_float(obj.get("pace_proj"))
        pace_gap = (float(pace_proj) - float(line)) if pace_proj is not None else None
        current_gap = float(app_module._safe_float(obj.get("actual")) or 0.0) - float(line)
        pregame_line = pregame_lines.get((team_tri, name_key, stat_key))
        sim_mu_adjusted = app_module._safe_float(obj.get("sim_mu_adjusted"))
        pregame_gap = None
        if pregame_line is not None and sim_mu_adjusted is not None:
            pregame_gap = float(sim_mu_adjusted) - float(pregame_line)

        elapsed_minutes = _elapsed_minutes(obj)
        progress_fraction = app_module._live_prop_progress_fraction(elapsed_minutes)
        win_prob = app_module._safe_float(context.get("win_prob"))
        implied_prob = app_module._safe_float(context.get("implied_prob"))
        line_live_age_sec = app_module._safe_float(obj.get("line_live_age_sec"))
        line_live_n = app_module._safe_float(obj.get("line_live_n"))
        bettable_score = app_module._safe_float(obj.get("bettable_score"))
        edge_sigma = app_module._safe_float(obj.get("edge_sigma"))
        received_at_text = str(obj.get("received_at") or "").strip() or None
        received_at = _parse_datetime(received_at_text)
        first_seen_age_sec = app_module._safe_float(obj.get("first_seen_age_sec"))
        if first_seen_age_sec is None:
            first_seen_age_sec = line_live_age_sec

        base_prob = app_module._live_prop_rank_probability(
            selected_side=side,
            selected_prob=win_prob,
            selected_implied_prob=implied_prob,
            pace_gap=pace_gap,
            sim_gap=sim_gap,
            pregame_gap=pregame_gap,
            current_gap=current_gap,
            progress_fraction=progress_fraction,
            score_diff_team=None,
            bettable_score=bettable_score,
            edge_sigma=edge_sigma,
            line_live_age_sec=line_live_age_sec,
            first_seen_age_sec=first_seen_age_sec,
            seen_observations=line_live_n,
        )
        if base_prob is None:
            continue
        shape_payload = app_module._live_prop_shape_payload(
            market_key=stat_key,
            selected_side=side,
            selected_prob=win_prob,
            pace_gap=pace_gap,
            sim_gap=sim_gap,
            pregame_gap=pregame_gap,
            current_gap=current_gap,
            progress_fraction=progress_fraction,
            score_diff_team=None,
            bettable_score=bettable_score,
            proj_min_final=context.get("proj_min_final"),
            exp_min_eff=context.get("exp_min_eff"),
            starter=obj.get("starter"),
            pf=obj.get("pf"),
            injury_flag=context.get("injury_flag"),
            pregame_team_total_ratio=context.get("pregame_team_total_ratio"),
            pregame_stat_multiplier=context.get("pregame_stat_multiplier"),
        )
        shape_score = app_module._safe_float(shape_payload.get("shape_score"))
        if shape_score is None:
            shape_score = 0.0

        rows.append(
            SweepRow(
                date=date_str,
                signal_group=_signal_group_key(obj, team_tri, name_key, stat_key, side),
                player=str(obj.get("player") or ""),
                team_tri=team_tri,
                stat=stat_key,
                side=side,
                klass=klass,
                line=float(line),
                actual_final=float(actual_final),
                price=float(price),
                outcome=outcome,
                profit_u=float(profit_u),
                base_score=round(float(base_prob) * 100.0, 4),
                shape_score=round(float(shape_score), 4),
                win_prob=win_prob,
                implied_prob=implied_prob,
                received_at=received_at,
                received_at_text=received_at_text,
                line_live_age_sec=line_live_age_sec,
                line_live_n=line_live_n,
            )
        )
    return rows


def _evaluate_weight(rows: list[SweepRow], weight: float, top_k: int, *, evaluation: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_date: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        blended_score = ((1.0 - float(weight)) * float(row.base_score)) + (float(weight) * float(row.shape_score))
        by_date.setdefault(row.date, []).append(
            {
                **row.__dict__,
                "blended_score": round(float(blended_score), 4),
            }
        )

    picks: list[dict[str, Any]] = []
    for date_str, items in by_date.items():
        ranked = sorted(items, key=lambda item: item["blended_score"], reverse=True)
        picks.extend(ranked[:top_k])

    if not picks:
        return ({
            "evaluation": evaluation,
            "weight": float(weight),
            "top_k": int(top_k),
            "dates": 0,
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "profit_u": 0.0,
            "roi_u_per_bet": 0.0,
            "win_rate": 0.0,
        }, [])

    wins = sum(1 for item in picks if item["outcome"] == "WIN")
    losses = sum(1 for item in picks if item["outcome"] == "LOSS")
    pushes = sum(1 for item in picks if item["outcome"] == "PUSH")
    profit = float(sum(float(item["profit_u"]) for item in picks))
    bet_count = len(picks)
    graded = max(1, wins + losses)
    summary = {
        "evaluation": evaluation,
        "weight": round(float(weight), 4),
        "top_k": int(top_k),
        "dates": len(by_date),
        "bets": int(bet_count),
        "wins": int(wins),
        "losses": int(losses),
        "pushes": int(pushes),
        "profit_u": round(profit, 4),
        "roi_u_per_bet": round(profit / float(max(1, bet_count)), 4),
        "win_rate": round(float(wins) / float(graded), 4),
    }
    return summary, picks


def _build_evaluation_sets(rows: list[SweepRow], follow_minutes: list[float]) -> list[tuple[str, list[SweepRow]]]:
    annotated_rows = _annotate_first_bet_timing(rows)
    evaluations: list[tuple[str, list[SweepRow]]] = []
    evaluations.append(("all_snapshots", annotated_rows))
    evaluations.append(("first_bet", _dedupe_signal_groups(annotated_rows)))
    for minutes in follow_minutes:
        label = f"bet_after_{int(minutes)}m"
        evaluations.append((label, _select_rows_near_target_minutes(annotated_rows, target_minutes=float(minutes))))
    return evaluations


def _picks_frame(picks: list[dict[str, Any]]) -> pd.DataFrame:
    if not picks:
        return pd.DataFrame()
    frame = pd.DataFrame(picks)
    preferred_columns = [
        "date",
        "player",
        "team_tri",
        "stat",
        "side",
        "klass",
        "line",
        "actual_final",
        "outcome",
        "profit_u",
        "blended_score",
        "base_score",
        "shape_score",
        "minutes_since_first_bet",
        "received_at_text",
        "line_live_age_sec",
        "line_live_n",
        "signal_group",
    ]
    columns = [column for column in preferred_columns if column in frame.columns]
    if columns:
        frame = frame[columns]
    return frame.sort_values([column for column in ["date", "blended_score", "player"] if column in frame.columns], ascending=[True, False, True])


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune live prop shape blend against playoff Render logs")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL for Render artifact downloads")
    parser.add_argument("--top-k", type=int, default=3, help="Picks per date to evaluate (default: 3)")
    parser.add_argument(
        "--weights",
        default="0.00,0.08,0.12,0.16,0.20,0.24,0.28,0.32,0.36,0.40",
        help="Comma-separated shape weights to test",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional output CSV path (default: data/processed/live_prop_shape_playoff_sweep_<start>_<end>.csv)",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds")
    parser.add_argument(
        "--follow-minutes",
        default="5,10,15",
        help="Comma-separated minutes-after-first-bet buckets for follow-through analysis",
    )
    parser.add_argument(
        "--export-picks",
        action="store_true",
        help="Also write the selected top-k picks for each evaluation/weight combination",
    )
    args = parser.parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    weights = [float(part.strip()) for part in str(args.weights).split(",") if part.strip()]
    follow_minutes = [float(part.strip()) for part in str(args.follow_minutes).split(",") if part.strip()]

    all_rows: list[SweepRow] = []
    for day in _daterange(start, end):
        date_str = day.isoformat()
        raw_rows = _fetch_signal_rows(args.base_url, date_str, timeout=float(args.timeout))
        if not raw_rows:
            continue
        all_rows.extend(_build_sweep_rows(date_str, raw_rows))

    if not all_rows:
        raise SystemExit("No settled playoff live prop rows available for the requested window")

    summary_rows: list[dict[str, Any]] = []
    export_pick_frames: list[pd.DataFrame] = []
    for evaluation, evaluation_rows in _build_evaluation_sets(all_rows, follow_minutes):
        for weight in weights:
            summary_row, picks = _evaluate_weight(evaluation_rows, weight, int(args.top_k), evaluation=evaluation)
            summary_rows.append(summary_row)
            if args.export_picks:
                picks_frame = _picks_frame(picks)
                if not picks_frame.empty:
                    picks_frame.insert(0, "weight", summary_row["weight"])
                    picks_frame.insert(0, "evaluation", evaluation)
                    export_pick_frames.append(picks_frame)

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(["evaluation", "roi_u_per_bet", "profit_u", "win_rate", "weight"], ascending=[True, False, False, False, True])

    out_path = Path(args.out) if args.out else (PROCESSED / f"live_prop_shape_playoff_sweep_{start.isoformat()}_{end.isoformat()}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)

    if args.export_picks and export_pick_frames:
        picks_out_path = out_path.with_name(f"{out_path.stem}_picks{out_path.suffix}")
        pd.concat(export_pick_frames, ignore_index=True).to_csv(picks_out_path, index=False)
        print(f"Picks: {picks_out_path}")

    print(summary.to_string(index=False))
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
