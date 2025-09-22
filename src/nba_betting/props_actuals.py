from __future__ import annotations

import subprocess
import sys
import os
import shutil
from pathlib import Path
import pandas as pd
from .config import paths


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
