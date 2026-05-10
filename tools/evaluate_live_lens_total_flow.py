#!/usr/bin/env python3
"""Evaluate scored Live Lens totals rows by recent-window flow drivers.

Reads:
- data/processed/reports/live_lens_scored_<date>.csv

Purpose:
- summarize full-game / half / quarter total accuracy by the new
  recent_window_* flow fields now carried through the scored audit CSV
- fail soft on older reports that predate those columns
"""

from __future__ import annotations

import argparse
import math
from datetime import date as _date
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "data" / "processed" / "reports"


def _parse_date(s: str) -> _date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _daterange(start: _date, end: _date) -> Iterable[_date]:
    cur = start
    while cur <= end:
        yield cur
        cur = cur + timedelta(days=1)


def _num(x: Any) -> float | None:
    try:
        if x is None:
            return None
        value = float(x)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


def _metrics(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {"n": 0}
    err = df["pred"].astype(float) - df["act"].astype(float)
    mae = float(err.abs().mean())
    rmse = float(math.sqrt(float((err**2).mean())))
    bias = float(err.mean())
    return {"n": int(len(df)), "mae": mae, "rmse": rmse, "bias": bias}


def _fmt_metrics(df: pd.DataFrame) -> str:
    m = _metrics(df)
    if int(m.get("n", 0)) <= 0:
        return "n=0"
    return f"n={int(m['n'])} mae={m['mae']:.2f} rmse={m['rmse']:.2f} bias={m['bias']:.2f}"


def _bucket_signed(series: pd.Series, threshold: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = pd.Series("flat", index=series.index, dtype="object")
    out.loc[values >= float(threshold)] = "positive"
    out.loc[values <= -float(threshold)] = "negative"
    out.loc[values.isna()] = "missing"
    return out


def _bucket_run_points(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = pd.Series("missing", index=series.index, dtype="object")
    out.loc[values.notna() & (values < 8)] = "<8"
    out.loc[values >= 8] = "8+"
    out.loc[values >= 12] = "12+"
    return out


def _bucket_drought(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = pd.Series("missing", index=series.index, dtype="object")
    out.loc[values.notna() & (values < 30)] = "<30s"
    out.loc[values >= 30] = "30s+"
    out.loc[values >= 75] = "75s+"
    return out


def _load_range(start: str, end: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for day in _daterange(_parse_date(start), _parse_date(end)):
        ds = day.isoformat()
        fp = REPORTS / f"live_lens_scored_{ds}.csv"
        if not fp.exists():
            continue
        try:
            frame = pd.read_csv(fp)
        except Exception:
            continue
        if frame.empty:
            continue
        frame["source_date"] = ds
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate scored Live Lens totals by recent flow fields")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", help="Single date YYYY-MM-DD")
    group.add_argument("--start", help="Start date YYYY-MM-DD")
    ap.add_argument("--end", help="End date YYYY-MM-DD (inclusive; default=start)")
    ap.add_argument("--min-n", type=int, default=3, help="Minimum rows to print a bucket")
    args = ap.parse_args()

    if args.date:
        start = end = str(args.date)
    else:
        start = str(args.start)
        end = str(args.end or args.start)

    df = _load_range(start, end)
    if df.empty:
        print(f"No scored Live Lens CSVs found for {start}..{end} under {REPORTS}")
        return 2

    totals = df[df["market"].isin(["total", "half_total", "quarter_total"])].copy()
    totals = totals.dropna(subset=["pred", "act"]).copy()
    if totals.empty:
        print(f"No scored totals rows found for {start}..{end}")
        return 2

    print(f"Rows loaded: {len(totals)} from {start}..{end}")
    print("\nOverall by market:")
    for market in ["total", "half_total", "quarter_total"]:
        part = totals[totals["market"] == market].copy()
        if part.empty:
            continue
        print(f"- {market}: {_fmt_metrics(part)}")

    needed_cols = [
        "recent_window_pace_adj",
        "recent_window_eff_adj",
        "recent_window_streak_adj",
        "recent_window_w",
    ]
    present_cols = [col for col in needed_cols if col in totals.columns]
    if not present_cols:
        print("\nFlow columns are absent in these scored CSVs. They were generated before the flow-logging enrichment landed.")
        return 0

    populated = False
    for col in present_cols:
        if pd.to_numeric(totals.get(col), errors="coerce").notna().any():
            populated = True
            break
    if not populated:
        print("\nFlow columns exist but have no populated values in this range. These logs predate the new recent_window_* totals logging.")
        return 0

    print("\nRecent window on/off:")
    if "recent_window_on" in totals.columns:
        for state_value, part in totals.groupby("recent_window_on"):
            label = "on" if int(_num(state_value) or 0) == 1 else "off"
            if len(part) >= int(args.min_n):
                print(f"- recent_window {label}: {_fmt_metrics(part)}")

    totals["pace_adj_bucket"] = _bucket_signed(totals.get("recent_window_pace_adj"), 0.25)
    totals["eff_adj_bucket"] = _bucket_signed(totals.get("recent_window_eff_adj"), 0.20)
    totals["streak_adj_bucket"] = _bucket_signed(totals.get("recent_window_streak_adj"), 0.15)
    totals["run_points_bucket"] = _bucket_run_points(totals.get("recent_window_run_points")) if "recent_window_run_points" in totals.columns else "missing"
    totals["drought_bucket"] = _bucket_drought(totals.get("recent_window_seconds_since_score")) if "recent_window_seconds_since_score" in totals.columns else "missing"

    for label, bucket_col in [
        ("pace adj", "pace_adj_bucket"),
        ("eff adj", "eff_adj_bucket"),
        ("streak adj", "streak_adj_bucket"),
        ("run points", "run_points_bucket"),
        ("scoring drought", "drought_bucket"),
    ]:
        print(f"\nBy {label}:")
        for bucket_value, part in totals.groupby(bucket_col):
            if len(part) < int(args.min_n):
                continue
            print(f"- {bucket_value}: {_fmt_metrics(part)}")

    if "shape_summary" in totals.columns:
        summaries = (
            totals["shape_summary"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        summaries = summaries[summaries.str.len() > 0]
        if not summaries.empty:
            print("\nTop shape summaries:")
            for summary, count in summaries.value_counts().head(10).items():
                print(f"- n={int(count)} {summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())