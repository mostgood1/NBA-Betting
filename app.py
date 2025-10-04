from __future__ import annotations

import base64
import os
from pathlib import Path
import sys

from flask import Flask, jsonify, redirect, request, send_from_directory
import threading
import subprocess
import shlex
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import subprocess as _subp

import pandas as pd
import numpy as np

# Ensure local package in src/ is importable (for odds, schedule, CLI)
from pathlib import Path as _PathEarly
BASE_DIR = _PathEarly(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
import sys as _sys_early
if str(SRC_DIR) not in _sys_early.path:
    _sys_early.path.insert(0, str(SRC_DIR))

try:
    # Optional (for scoreboard)
    from nba_api.stats.endpoints import scoreboardv2 as _scoreboardv2
    from nba_api.stats.library import http as _nba_http
except Exception:  # pragma: no cover
    _scoreboardv2 = None  # type: ignore
    _nba_http = None  # type: ignore

try:
    # local package for odds fetching (now importable due to early sys.path insert)
    from nba_betting.odds_bovada import (
        fetch_bovada_odds_current as _fetch_bovada_odds_current,
        probe_bovada as _probe_bovada,
    )  # type: ignore
except Exception:  # pragma: no cover
    _fetch_bovada_odds_current = None  # type: ignore
    _probe_bovada = None  # type: ignore

# Optional: load environment variables from a .env file if present
try:  # lightweight optional dependency
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

WEB_DIR = BASE_DIR / "web"
CRON_META_PATH = BASE_DIR / "data" / "processed" / ".cron_meta.json"

# Serve the static frontend under /web, and serve the cards at '/'
app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="/web")


@app.route("/")
def root():
    # Serve the NBA slate homepage directly at '/'
    return send_from_directory(str(WEB_DIR), "index.html")


@app.route("/web/")
@app.route("/web/index.html")
def web_index():
    # Redirect legacy URL to root for a single canonical entrypoint
    return redirect("/")


@app.route("/web/<path:path>")
def web_static(path: str):
    # Serve any static asset in web/
    return send_from_directory(str(WEB_DIR), path)


@app.route("/data/<path:path>")
def data_static(path: str):
    # Serve data files (JSON/CSV) referenced by the frontend, e.g., /data/processed/*.json
    data_dir = BASE_DIR / "data"
    return send_from_directory(str(data_dir), path)

# Friendly routes to mirror NHL site paths (serve static HTML files)
@app.route("/recommendations")
def route_recommendations():
    return send_from_directory(str(WEB_DIR), "recommendations.html")

@app.route("/props")
def route_props():
    return send_from_directory(str(WEB_DIR), "props.html")

@app.route("/reconciliation")
def route_reconciliation():
    return send_from_directory(str(WEB_DIR), "reconciliation.html")

@app.route("/odds-coverage")
def route_odds_coverage():
    return send_from_directory(str(WEB_DIR), "odds_coverage.html")


@app.route("/predictions_<date>.csv")
def serve_predictions_csv(date: str):
    """Serve predictions_YYYY-MM-DD.csv from data/processed (fallback root)."""
    processed = BASE_DIR / "data" / "processed" / f"predictions_{date}.csv"
    if processed.exists():
        return send_from_directory(str(processed.parent), processed.name)
    legacy = BASE_DIR / f"predictions_{date}.csv"
    if legacy.exists():
        return send_from_directory(str(BASE_DIR), legacy.name)
    from flask import abort
    abort(404)


@app.route("/props_predictions_<date>.csv")
def serve_props_predictions_csv(date: str):
    """Serve props_predictions_YYYY-MM-DD.csv from data/processed (fallback root)."""
    processed = BASE_DIR / "data" / "processed" / f"props_predictions_{date}.csv"
    if processed.exists():
        return send_from_directory(str(processed.parent), processed.name)
    legacy = BASE_DIR / f"props_predictions_{date}.csv"
    if legacy.exists():
        return send_from_directory(str(BASE_DIR), legacy.name)
    from flask import abort
    abort(404)


@app.route("/health")
def health():
    # Lightweight health/status
    try:
        exists = (WEB_DIR / "index.html").exists()
        return jsonify({"status": "ok", "have_index": bool(exists)}), 200
    except Exception as e:  # noqa: BLE001
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/version")
def api_version():
    """Return app version info (git SHA, branch) to verify deploy state."""
    try:
        sha = _subp.check_output(["git", "rev-parse", "HEAD"], cwd=str(BASE_DIR), text=True).strip()
    except Exception:
        sha = None
    try:
        branch = _subp.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(BASE_DIR), text=True).strip()
    except Exception:
        branch = None
    return jsonify({
        "sha": sha,
        "branch": branch,
        "time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    })


@app.route("/favicon.ico")
def favicon():
    # 1x1 transparent PNG to avoid 404s
    png_b64 = (
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y6r/RwAAAAASUVORK5CYII="
    )
    png = base64.b64decode(png_b64)
    from flask import Response  # local import to keep module import light

    return Response(png, mimetype="image/png")


# ---------------- Shared helpers ---------------- #

def _parse_date_param(req, default_to_today: bool = True) -> str:
    val = (req.args.get("date") or req.args.get("d") or "").strip()
    if not val and default_to_today:
        try:
            return datetime.utcnow().date().isoformat()
        except Exception:
            return ""
    try:
        # normalize YYYY-MM-DD
        return pd.to_datetime(val).date().isoformat()
    except Exception:
        return val


def _read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        return None
    return None


def _number(x):
    try:
        if pd.isna(x):
            return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _normalize_team_str(x: str) -> str:
    # Best effort normalization: uppercase tricode if given, else strip
    try:
        return str(x).strip()
    except Exception:
        return str(x)


def _implied_prob_american(o: Any) -> Optional[float]:
    try:
        if o is None or (isinstance(o, float) and not np.isfinite(o)):
            return None
        o = float(o)
        if o == 0:
            return None
        if o > 0:
            return 100.0 / (o + 100.0)
        return (-o) / ((-o) + 100.0)
    except Exception:
        return None


def _american_to_b(o: Any) -> Optional[float]:
    try:
        o = float(o)
        return (o / 100.0) if o > 0 else (100.0 / abs(o))
    except Exception:
        return None


def _ev_from_prob_and_american(p: Optional[float], odds: Any) -> Optional[float]:
    if p is None:
        return None
    b = _american_to_b(odds)
    if b is None:
        return None
    try:
        return p * b - (1 - p)
    except Exception:
        return None


def _has_games_for_date(date_str: str, verbose: bool = False) -> bool:
    """Return True if there are NBA games on the given date.

    Preference order:
    1) nba_api ScoreboardV2 (if available)
    2) Bovada odds feed (if available)
    """
    # First try nba_api ScoreboardV2
    try:
        if _scoreboardv2 is not None:
            sb = _scoreboardv2.ScoreboardV2(game_date=date_str, day_offset=0, timeout=30)
            nd = sb.get_normalized_dict()
            gh = pd.DataFrame(nd.get("GameHeader", []))
            if not gh.empty and len(gh) > 0:
                return True
    except Exception as e:
        if verbose:
            print(f"[_has_games_for_date] scoreboard error: {e}")
    # Fallback to Bovada odds
    try:
        if _fetch_bovada_odds_current is not None:
            dt = pd.to_datetime(date_str).to_pydatetime()
            o = _fetch_bovada_odds_current(dt, verbose=False)
            if isinstance(o, pd.DataFrame) and not o.empty:
                return True
    except Exception as e:
        if verbose:
            print(f"[_has_games_for_date] bovada error: {e}")
    return False


# ---------------- Admin: daily update (mirrors NFL-Betting shape) ---------------- #
_job_state = {
    "running": False,
    "started_at": None,
    "ended_at": None,
    "ok": None,
    "logs": [],
    "log_file": None,
}


def _append_log(line: str) -> None:
    try:
        ts = datetime.utcnow().isoformat(timespec="seconds")
        msg = f"[{ts}] {line.rstrip()}"
        _job_state["logs"].append(msg)
        if len(_job_state["logs"]) > 1000:
            del _job_state["logs"][:-500]
        try:
            lf = _job_state.get("log_file")
            if lf:
                with open(lf, "a", encoding="utf-8", errors="ignore") as f:
                    f.write(msg + "\n")
        except Exception:
            pass
    except Exception:
        pass


def _ensure_logs_dir() -> Path:
    p = BASE_DIR / "logs"
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p


def _find_fallback_odds_for_date(date_str: str) -> Optional[Path]:
    """Return best available non-Bovada odds file for the given date, if any.

    Preference order:
      1) closing_lines_{date}.csv
      2) odds_{date}.csv
      3) market_{date}.csv
    """
    candidates = [
        BASE_DIR / "data" / "processed" / f"closing_lines_{date_str}.csv",
        BASE_DIR / "data" / "processed" / f"odds_{date_str}.csv",
        BASE_DIR / "data" / "processed" / f"market_{date_str}.csv",
    ]
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return p
        except Exception:
            continue
    return None


def _git_commit_and_push(msg: str) -> tuple[bool, str]:
    """Commit and push changes to the current branch.

    Auth methods:
    - GH_TOKEN or GIT_PAT env: used to set a push URL for 'origin' without altering fetch URL.
    - GH_NAME/GH_EMAIL env: configure git user identity.
    Returns (ok, detail).
    """
    try:
        env = {**os.environ}
        # Configure identity if provided
        name = os.environ.get("GH_NAME") or os.environ.get("GIT_NAME")
        email = os.environ.get("GH_EMAIL") or os.environ.get("GIT_EMAIL")
        if name:
            subprocess.run(["git", "config", "user.name", name], cwd=str(BASE_DIR), check=False)
        if email:
            subprocess.run(["git", "config", "user.email", email], cwd=str(BASE_DIR), check=False)
        # Determine current branch
        try:
            branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(BASE_DIR), text=True).strip()
            if not branch or branch == "HEAD":
                branch = os.environ.get("GIT_BRANCH", "main")
        except Exception:
            branch = os.environ.get("GIT_BRANCH", "main")
        # Set push URL with token if present
        token = os.environ.get("GH_TOKEN") or os.environ.get("GIT_PAT")
        push_url_set = False
        if token:
            try:
                origin = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=str(BASE_DIR), text=True).strip()
                # Expect https URL; fall back to origin if parsing fails
                url = origin
                if origin.startswith("https://") and "@" not in origin:
                    # Insert token; use x-access-token as username to avoid leaking real usernames
                    url = origin.replace("https://", f"https://x-access-token:{token}@")
                # Set push URL only
                subprocess.run(["git", "remote", "set-url", "--push", "origin", url], cwd=str(BASE_DIR), check=False)
                push_url_set = True
            except Exception:
                push_url_set = False
        # Stage and commit (allow empty to create a heartbeat commit if needed)
        subprocess.run(["git", "add", "-A"], cwd=str(BASE_DIR), check=False)
        subprocess.run(["git", "commit", "-m", msg, "--allow-empty"], cwd=str(BASE_DIR), check=False)
        # Rebase then push
        subprocess.run(["git", "pull", "--rebase"], cwd=str(BASE_DIR), check=False)
        rc = subprocess.run(["git", "push", "origin", f"HEAD:{branch}"], cwd=str(BASE_DIR), check=False)
        ok = (rc.returncode == 0)
        detail = f"pushed to {branch}; push_url={'set' if push_url_set else 'default'}"
        return ok, detail
    except Exception as e:
        return False, f"git push error: {e}"


def _run_to_file(cmd: list[str] | str, log_fp: Path, cwd: Path | None = None, env: dict | None = None) -> int:
    if isinstance(cmd, list):
        popen_cmd = cmd
    else:
        popen_cmd = shlex.split(cmd)
    with log_fp.open("a", encoding="utf-8", errors="ignore") as out:
        out.write(f"[{datetime.utcnow().isoformat(timespec='seconds')}] Starting: {' '.join(popen_cmd)}\n")
        out.flush()
        proc = subprocess.Popen(
            popen_cmd,
            cwd=str(cwd) if cwd else None,
            env={**os.environ, **(env or {})},
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        proc.wait()
        out.write(f"[{datetime.utcnow().isoformat(timespec='seconds')}] Exited with code {proc.returncode}\n")
        out.flush()
        return int(proc.returncode)


def _admin_auth_ok(req) -> bool:
    key = os.environ.get("ADMIN_KEY")
    if not key:
        # If no key configured, allow only local (127.0.0.1) requests
        try:
            host = (req.remote_addr or "").strip()
            if host in {"127.0.0.1", "::1", "::ffff:127.0.0.1"}:
                return True
            # Also allow private LAN ranges for local development convenience
            if host.startswith("192.168.") or host.startswith("10."):
                return True
            if host.startswith("172."):
                try:
                    parts = host.split(".")
                    if len(parts) >= 2:
                        second = int(parts[1])
                        if 16 <= second <= 31:
                            return True
                except Exception:
                    pass
            return False
        except Exception:
            return False
    return (req.args.get("key") == key) or (req.headers.get("X-Admin-Key") == key)


def _cron_auth_ok(req) -> bool:
    """Cron auth using CRON_TOKEN. Accepts:
    - Authorization: Bearer <token>
    - X-Cron-Token: <token>
    - ?token=<token>
    If CRON_TOKEN is unset, fall back to admin auth policy for local/dev.
    """
    token = os.environ.get("CRON_TOKEN")
    if not token:
        # fall back to admin auth (local dev convenience)
        return _admin_auth_ok(req)
    auth = (req.headers.get("Authorization") or "").strip()
    if auth.lower().startswith("bearer ") and auth.split(" ", 1)[1] == token:
        return True
    if (req.headers.get("X-Cron-Token") == token) or (req.args.get("token") == token):
        return True
    return False


def _daily_update_job(do_push: bool) -> None:
    _job_state["running"] = True
    _job_state["started_at"] = datetime.utcnow().isoformat()
    _job_state["ended_at"] = None
    _job_state["ok"] = None
    _job_state["logs"] = []
    logs_dir = _ensure_logs_dir()
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"web_daily_update_{stamp}.log"
    _job_state["log_file"] = str(log_file)
    try:
        _append_log("Starting daily update...")
        py = os.environ.get("PYTHON", (os.environ.get("VIRTUAL_ENV") or "") + "/bin/python")
        if not py or not Path(str(py)).exists():
            py = (Path(os.environ.get("VIRTUAL_ENV") or "") / "Scripts" / "python.exe")
            if not py.exists():
                py = "python"
        env = {"PYTHONPATH": str(SRC_DIR)}
        cmds: list[list[str]] = []
        # Light, safe steps for static UI refresh; customize as needed.
        # Example: rebuild closing lines snapshot if CLI supports it.
        # cmds.append([str(py), "-m", "nba_betting.cli", "export-closing-lines-csv", "--date", "2025-04-13"])  # sample
        # Example: re-run predictions for a sample matchups CSV (if present)
        sample_csv = BASE_DIR / "samples" / "matchups.csv"
        if sample_csv.exists():
            cmds.append([str(py), "-m", "nba_betting.cli", "predict", "--input", str(sample_csv)])
        rc_total = 0
        for c in cmds:
            _append_log(f"Running: {' '.join(c)}")
            rc = _run_to_file(c, log_file, cwd=BASE_DIR, env=env)
            rc_total += int(rc)
            _append_log(f"Exit code: {rc}")
            if rc != 0:
                break
        ok = (rc_total == 0)
        _append_log(f"Daily update finished. ok={ok}")
        # Optional: push updates back to Git if requested and configured
        if ok and do_push:
            try:
                _append_log("Pushing changes (if any) to Git...")
                # minimal push (requires git configured on Render and token/permissions)
                subprocess.run(["git", "add", "-A"], cwd=str(BASE_DIR), check=False)
                subprocess.run(["git", "commit", "-m", "chore: daily update"], cwd=str(BASE_DIR), check=False)
                subprocess.run(["git", "pull", "--rebase"], cwd=str(BASE_DIR), check=False)
                subprocess.run(["git", "push"], cwd=str(BASE_DIR), check=False)
                _append_log("Git push attempted.")
            except Exception as e:  # noqa: BLE001
                _append_log(f"Git push error: {e}")
        _job_state["ok"] = ok
    except Exception as e:  # noqa: BLE001
        _append_log(f"Daily update exception: {e}")
        _job_state["ok"] = False
    finally:
        _job_state["ended_at"] = datetime.utcnow().isoformat()
        _job_state["running"] = False


# ---------------- Data APIs (parity with NHL web) ---------------- #

def _find_predictions_for_date(date_str: str) -> Optional[Path]:
    # Prefer processed path
    p = BASE_DIR / "data" / "processed" / f"predictions_{date_str}.csv"
    if p.exists():
        return p
    # Legacy root fallback
    legacy = BASE_DIR / f"predictions_{date_str}.csv"
    if legacy.exists():
        return legacy
    return None


def _find_game_odds_for_date(date_str: str) -> Optional[Path]:
    # Processed standardized game odds
    p = BASE_DIR / "data" / "processed" / f"game_odds_{date_str}.csv"
    if p.exists():
        return p
    # Alternate names we search for
    for name in [
        f"closing_lines_{date_str}.csv",
        f"odds_{date_str}.csv",
        f"market_{date_str}.csv",
    ]:
        q = BASE_DIR / "data" / "processed" / name
        if q.exists():
            return q
    return None


@app.route("/api/status")
def api_status():
    try:
        dproc = BASE_DIR / "data" / "processed"
        files = [str(x.name) for x in dproc.glob("*.csv")] if dproc.exists() else []
        return jsonify({
            "status": "ok",
            "processed_files": files,
            "have_index": (WEB_DIR / "index.html").exists(),
        })
    except Exception as e:  # noqa: BLE001
        return jsonify({"status": "error", "error": str(e)}), 500


def _cron_meta_update(kind: str, payload: Dict[str, Any]) -> None:
    try:
        CRON_META_PATH.parent.mkdir(parents=True, exist_ok=True)
        base = {}
        if CRON_META_PATH.exists():
            try:
                import json as _json
                base = _json.loads(CRON_META_PATH.read_text(encoding="utf-8", errors="ignore"))
                if not isinstance(base, dict):
                    base = {}
            except Exception:
                base = {}
        now_iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        entry = dict(payload)
        entry["timestamp"] = now_iso
        base[f"last_{kind}"] = entry
        try:
            import json as _json
            CRON_META_PATH.write_text(_json.dumps(base, indent=2), encoding="utf-8")
        except Exception:
            pass
    except Exception:
        pass


@app.route("/api/cron/meta")
def api_cron_meta():
    """Return lightweight information about last cron runs (best-effort).

    Includes last_refresh_bovada if recorded, and the latest odds file mtime as fallback.
    """
    try:
        out: Dict[str, Any] = {}
        # Load recorded meta if present
        if CRON_META_PATH.exists():
            try:
                import json as _json
                meta = _json.loads(CRON_META_PATH.read_text(encoding="utf-8", errors="ignore"))
                if isinstance(meta, dict):
                    out.update(meta)
            except Exception:
                pass
        # Fallback: scan for the newest odds CSV mtime
        try:
            dproc = BASE_DIR / "data" / "processed"
            newest_ts = None
            newest_file = None
            if dproc.exists():
                for p in dproc.glob("*.csv"):
                    if not any(s in p.name for s in ("odds_", "game_odds_", "market_", "closing_lines_")):
                        continue
                    st = p.stat().st_mtime
                    if newest_ts is None or st > newest_ts:
                        newest_ts = st
                        newest_file = p
            if newest_ts is not None:
                out.setdefault("fallback_odds_latest", {})
                out["fallback_odds_latest"] = {
                    "path": str(newest_file),
                    "mtime": datetime.utcfromtimestamp(newest_ts).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
        except Exception:
            pass
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/last-updated")
def api_last_updated():
    d = _parse_date_param(request)
    pred = _find_predictions_for_date(d) if d else None
    odds = _find_game_odds_for_date(d) if d else None
    try:
        def mtime(p: Optional[Path]) -> Optional[str]:
            if p and p.exists():
                return datetime.utcfromtimestamp(p.stat().st_mtime).isoformat()
            return None
        return jsonify({
            "date": d,
            "predictions": str(pred) if pred else None,
            "predictions_mtime": mtime(pred),
            "odds": str(odds) if odds else None,
            "odds_mtime": mtime(odds),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predictions")
def api_predictions():
    d = _parse_date_param(request)
    if not d:
        return jsonify({"error": "missing date"}), 400
    p = _find_predictions_for_date(d)
    if not p:
        return jsonify({"date": d, "rows": []})
    try:
        df = pd.read_csv(p)
        # Try to merge odds if available
        q = _find_game_odds_for_date(d)
        if q is not None:
            try:
                o = pd.read_csv(q)
                if "date" in o.columns:
                    o["date"] = pd.to_datetime(o["date"], errors="coerce").dt.date
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
                on = ["date", "home_team", "visitor_team"]
                if all(c in df.columns for c in on) and all(c in o.columns for c in on):
                    df = df.merge(o, on=on, how="left", suffixes=("", "_odds"))
                    # Compute implied and edges if possible
                    if "home_ml" in df.columns and "home_win_prob" in df.columns:
                        df["home_implied_prob"] = df["home_ml"].apply(_implied_prob_american)
                        df["edge_win"] = df["home_win_prob"].astype(float) - df["home_implied_prob"].astype(float)
                    if "home_spread" in df.columns and "pred_margin" in df.columns:
                        df["market_home_margin"] = -pd.to_numeric(df["home_spread"], errors="coerce")
                        df["edge_spread"] = pd.to_numeric(df["pred_margin"], errors="coerce") - df["market_home_margin"]
                    if "total" in df.columns and "pred_total" in df.columns:
                        df["edge_total"] = pd.to_numeric(df["pred_total"], errors="coerce") - pd.to_numeric(df["total"], errors="coerce")
            except Exception:
                pass
        # Return compact JSON
        rows = df.fillna("").to_dict(orient="records")
        return jsonify({"date": d, "rows": rows})
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


@app.route("/api/recommendations")
def api_recommendations():
    """Derive simple recommendations from predictions and odds for the date.

    - Winner: pick side with higher EV if EV > 0
    - Spread: pick side of model margin if abs(edge_spread) >= threshold
    - Total: pick Over/Under if abs(edge_total) >= threshold
    """
    d = _parse_date_param(request)
    if not d:
        return jsonify({"error": "missing date"}), 400
    try:
        pred_path = _find_predictions_for_date(d)
        if not pred_path:
            return jsonify({"date": d, "rows": [], "summary": {}})
        df = pd.read_csv(pred_path)
        # Merge odds if present
        q = _find_game_odds_for_date(d)
        if q is not None:
            try:
                o = pd.read_csv(q)
                for col in ("date",):
                    if col in o.columns:
                        o[col] = pd.to_datetime(o[col], errors="coerce").dt.date
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
                on = ["date", "home_team", "visitor_team"]
                if all(c in df.columns for c in on) and all(c in o.columns for c in on):
                    df = df.merge(o, on=on, how="left", suffixes=("", "_odds"))
            except Exception:
                pass
        # Build recs
        recs: List[Dict[str, Any]] = []
        th_spread = float(request.args.get("spread_edge", 1.0))
        th_total = float(request.args.get("total_edge", 1.5))
        for _, r in df.iterrows():
            try:
                home = r.get("home_team"); away = r.get("visitor_team")
                # Winner EV
                p_home = _number(r.get("home_win_prob"))
                ev_h = _ev_from_prob_and_american(p_home, r.get("home_ml")) if p_home is not None else None
                ev_a = _ev_from_prob_and_american(None if p_home is None else (1 - p_home), r.get("away_ml"))
                if ev_h is not None or ev_a is not None:
                    side = home if (ev_h or -1) >= (ev_a or -1) else away
                    ev = ev_h if side == home else ev_a
                    if ev is not None and ev > 0:
                        recs.append({
                            "market": "ML", "side": side, "home": home, "away": away,
                            "ev": float(ev), "date": d,
                        })
                # Spread rec
                pred_m = _number(r.get("pred_margin"))
                home_spread = _number(r.get("home_spread"))
                if pred_m is not None and home_spread is not None:
                    market_home_margin = -home_spread
                    edge = pred_m - market_home_margin
                    if abs(edge) >= th_spread:
                        side = home if edge > 0 else away
                        recs.append({
                            "market": "ATS", "side": side, "home": home, "away": away,
                            "edge": float(edge), "date": d,
                        })
                # Total rec
                pred_t = _number(r.get("pred_total"))
                total = _number(r.get("total"))
                if pred_t is not None and total is not None:
                    edge_t = pred_t - total
                    if abs(edge_t) >= th_total:
                        side = "Over" if edge_t > 0 else "Under"
                        recs.append({
                            "market": "TOTAL", "side": side, "home": home, "away": away,
                            "edge": float(edge_t), "date": d,
                        })
            except Exception:
                continue
        # Simple summary
        summary = {
            "n": len(recs),
            "by_market": {
                k: int(sum(1 for x in recs if x["market"] == k)) for k in ("ML","ATS","TOTAL")
            },
        }
        return jsonify({"date": d, "rows": recs, "summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/props")
def api_props():
    d = _parse_date_param(request)
    if not d:
        return jsonify({"error": "missing date"}), 400
    try:
        # Prefer edges if available, fall back to predictions
        edges_p = BASE_DIR / "data" / "processed" / f"props_edges_{d}.csv"
        preds_p = BASE_DIR / "data" / "processed" / f"props_predictions_{d}.csv"
        df = _read_csv_if_exists(edges_p)
        src = "edges"
        if df is None or df.empty:
            df = _read_csv_if_exists(preds_p)
            src = "predictions"
        if df is None:
            return jsonify({"date": d, "rows": [], "source": None})
        # Optional filters
        min_edge = float(request.args.get("min_edge", "0"))
        min_ev = float(request.args.get("min_ev", "0"))
        if "edge" in df.columns:
            df = df[pd.to_numeric(df["edge"], errors="coerce").fillna(0) >= min_edge]
        if "ev" in df.columns:
            df = df[pd.to_numeric(df["ev"], errors="coerce").fillna(0) >= min_ev]
        rows = df.fillna("").to_dict(orient="records")
        return jsonify({"date": d, "source": src, "rows": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/reconciliation")
def api_reconciliation():
    d = _parse_date_param(request)
    if not d:
        return jsonify({"error": "missing date"}), 400
    try:
        # Use recon files if present
        gpath = BASE_DIR / "data" / "processed" / f"recon_games_{d}.csv"
        ppath = BASE_DIR / "data" / "processed" / f"recon_props_{d}.csv"
        gdf = _read_csv_if_exists(gpath)
        pdf = _read_csv_if_exists(ppath)
        out: Dict[str, Any] = {"date": d}
        if gdf is not None and not gdf.empty:
            # Compute simple errors summary
            for col in ("margin_error","total_error"):
                if col in gdf.columns:
                    s = pd.to_numeric(gdf[col], errors="coerce").dropna()
                    if not s.empty:
                        out[f"{col}_mae"] = float(s.abs().mean())
                        out[f"{col}_rmse"] = float(np.sqrt((s**2).mean()))
            out["games"] = int(len(gdf))
        else:
            out["games"] = 0
        out["props_rows"] = int(0 if pdf is None else len(pdf))
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/odds-coverage")
def api_odds_coverage():
    d = _parse_date_param(request)
    if not d:
        return jsonify({"error": "missing date"}), 400
    try:
        # Import team normalizer (maps aliases/abbreviations to full names)
        try:
            from nba_betting.teams import normalize_team as _norm_team  # type: ignore
        except Exception:  # pragma: no cover
            def _norm_team(x: str) -> str:
                return str(x or "").strip()
        pred_p = _find_predictions_for_date(d)
        odds_p = _find_game_odds_for_date(d)
        df = pd.read_csv(pred_p) if pred_p else pd.DataFrame()
        o = pd.read_csv(odds_p) if odds_p else pd.DataFrame()
        rows = []
        if not df.empty:
            # Normalize keys
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
            if not o.empty and "date" in o.columns:
                o["date"] = pd.to_datetime(o["date"], errors="coerce").dt.date
            # Normalize team names to common canonical form to ensure merge matches
            for frame in (df, o):
                if not frame.empty:
                    if "home_team" in frame.columns:
                        frame["home_team_norm"] = frame["home_team"].map(lambda x: _norm_team(str(x)))
                    if "visitor_team" in frame.columns:
                        frame["visitor_team_norm"] = frame["visitor_team"].map(lambda x: _norm_team(str(x)))
            on_cols = ["date", "home_team_norm", "visitor_team_norm"] if ("home_team_norm" in df.columns and "home_team_norm" in o.columns) else ["date","home_team","visitor_team"]
            merged = df.merge(o, left_on=on_cols, right_on=on_cols, how="left", suffixes=("","_odds")) if not o.empty else df
            for _, r in merged.iterrows():
                rows.append({
                    "home_team": r.get("home_team"),
                    "visitor_team": r.get("visitor_team"),
                    "have_ml": bool(pd.notna(r.get("home_ml")) and pd.notna(r.get("away_ml"))),
                    "have_spread": bool(pd.notna(r.get("home_spread"))),
                    "have_total": bool(pd.notna(r.get("total"))),
                })
        return jsonify({"date": d, "rows": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Simple in-memory cache for scoreboard (to limit API calls)
_scoreboard_cache: Dict[str, Tuple[float, Any]] = {}


@app.route("/api/scoreboard")
def api_scoreboard():
    d = _parse_date_param(request)
    if not d:
        return jsonify({"error": "missing date"}), 400
    # Serve from cache within 30 seconds
    now = time.time()
    ent = _scoreboard_cache.get(d)
    if ent and now - ent[0] < 30:
        return jsonify(ent[1])
    if _scoreboardv2 is None:
        return jsonify({"date": d, "error": "nba_api not installed"}), 500
    try:
        # Harden headers
        try:
            if _nba_http is not None:
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
        sb = _scoreboardv2.ScoreboardV2(game_date=d, day_offset=0, timeout=30)
        nd = sb.get_normalized_dict()
        gh = pd.DataFrame(nd.get("GameHeader", []))
        ls = pd.DataFrame(nd.get("LineScore", []))
        games = []
        if not gh.empty and not ls.empty:
            cgh = {c.upper(): c for c in gh.columns}
            cls = {c.upper(): c for c in ls.columns}
            # Map TEAM_ID -> ABBR
            abbr = {}
            for _, r in ls.iterrows():
                try:
                    abbr[int(r[cls["TEAM_ID"]])] = str(r[cls["TEAM_ABBREVIATION"]]).upper()
                except Exception:
                    pass
            for _, g in gh.iterrows():
                try:
                    hid = int(g[cgh["HOME_TEAM_ID"]]); vid = int(g[cgh["VISITOR_TEAM_ID"]])
                    games.append({
                        "home": abbr.get(hid),
                        "away": abbr.get(vid),
                        "status": g.get(cgh.get("GAME_STATUS_TEXT", "GAME_STATUS_TEXT")),
                        "game_id": g.get(cgh.get("GAME_ID", "GAME_ID")),
                    })
                except Exception:
                    continue
        payload = {"date": d, "games": games}
        _scoreboard_cache[d] = (now, payload)
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/schedule")
def api_schedule():
    """Serve the NBA schedule as JSON. If the processed JSON is missing, attempt to generate it.

    Query params:
      - season (optional): currently defaults to '2025-26'
      - date (optional): if provided, filter to that YYYY-MM-DD
    """
    season = (request.args.get("season") or "2025-26").strip()
    date_str = _parse_date_param(request, default_to_today=False)
    out_dir = BASE_DIR / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "schedule_2025_26.json"  # schema tied to function name for now
    try:
        df = None
        if json_path.exists():
            try:
                df = pd.read_json(json_path)
            except Exception:
                df = None
        if df is None or df.empty:
            # Try to fetch via python module
            try:
                from nba_betting.schedule import fetch_schedule_2025_26  # type: ignore
                df = fetch_schedule_2025_26()
                # Save in compact list form for frontend
                df.to_json(json_path, orient="records", date_format="iso")
            except Exception as e:
                return jsonify({"error": f"Failed to load or build schedule: {e}"}), 500
        # Optional filter by date
        if date_str:
            try:
                df = df.copy()
                if "date_utc" in df.columns:
                    df["date_utc"] = pd.to_datetime(df["date_utc"], errors="coerce").dt.date
                mask = df["date_utc"].astype(str) == date_str
                df = df[mask]
            except Exception:
                pass
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cron/refresh-bovada", methods=["POST", "GET"])
def api_cron_refresh_bovada():
    """Fetch current Bovada odds for a specific date and save a standardized CSV under data/processed.

    Query params:
      - date (required): YYYY-MM-DD
    Auth: CRON_TOKEN (preferred) or ADMIN_KEY (fallback if CRON_TOKEN unset).
    """
    if not (_cron_auth_ok(request) or _admin_auth_ok(request)):
        return jsonify({"error": "unauthorized"}), 401
    d = _parse_date_param(request, default_to_today=False)
    if not d:
        # Default to yesterday (UTC) for closing lines capture
        d = (datetime.utcnow().date() - timedelta(days=1)).isoformat()
    if _fetch_bovada_odds_current is None:
        return jsonify({"error": "bovada fetcher not available"}), 500
    # Pass date string directly; function treats as ET calendar date
    try:
        _ = pd.to_datetime(d)
    except Exception:
        return jsonify({"error": "invalid date"}), 400
    try:
        df = _fetch_bovada_odds_current(d)
        rows = 0 if df is None else len(df)
        out = BASE_DIR / "data" / "processed" / f"game_odds_{d}.csv"
        used_fallback = False
        fallback_path = None
        if df is not None and not df.empty:
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
        else:
            # Try fallback to consensus/closing lines if Bovada has nothing
            fb = _find_fallback_odds_for_date(d)
            if fb is not None:
                try:
                    o = pd.read_csv(fb)
                    if not o.empty:
                        out.parent.mkdir(parents=True, exist_ok=True)
                        o.to_csv(out, index=False)
                        rows = int(len(o))
                        used_fallback = True
                        fallback_path = str(fb)
                except Exception:
                    pass
        # Record cron meta best-effort
        try:
            meta = {
                "date": d,
                "rows": int(rows),
                "output": str(out),
            }
            if used_fallback:
                meta.update({"used_fallback": True, "fallback_path": fallback_path})
            _cron_meta_update("refresh_bovada", meta)
        except Exception:
            pass
        # Optional push
        pushed = None; push_detail = None
        if str(request.args.get("push", "0")).lower() in {"1","true","yes"}:
            ok, detail = _git_commit_and_push(msg=f"refresh-bovada {d}")
            pushed = bool(ok); push_detail = detail
        return jsonify({
            "date": d, "rows": int(rows), "output": str(out),
            "used_fallback": used_fallback, "fallback_path": fallback_path,
            "pushed": pushed, "push_detail": push_detail,
        })
    except Exception as e:
        return jsonify({"error": f"bovada fetch failed: {e}"}), 500


@app.route("/api/cron/probe-bovada")
def api_cron_probe_bovada():
    """Debug endpoint: probe Bovada endpoints and report event counts for a date.

    Auth: CRON_TOKEN or ADMIN_KEY. Query: date=YYYY-MM-DD
    """
    if not (_cron_auth_ok(request) or _admin_auth_ok(request)):
        return jsonify({"error": "unauthorized"}), 401
    d = _parse_date_param(request, default_to_today=True)
    if _probe_bovada is None:
        return jsonify({"error": "probe not available"}), 500
    try:
        dt = pd.to_datetime(d)
        out = _probe_bovada(dt, verbose=True)
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cron/capture-closing", methods=["POST", "GET"])
def api_cron_capture_closing():
    """Export consensus closing lines CSV for a given date under data/processed.

    Query params:
      - date (required): YYYY-MM-DD
    """
    if not _cron_auth_ok(request):
        return jsonify({"error": "unauthorized"}), 401
    d = _parse_date_param(request, default_to_today=False)
    if not d:
        return jsonify({"error": "missing date"}), 400
    # Choose python executable
    py = os.environ.get("PYTHON", (os.environ.get("VIRTUAL_ENV") or "") + "/bin/python")
    if not py or not Path(str(py)).exists():
        py_win = (Path(os.environ.get("VIRTUAL_ENV") or "") / "Scripts" / "python.exe")
        py = str(py_win) if py_win.exists() else "python"
    logs_dir = _ensure_logs_dir(); stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"cron_capture_closing_{d}_{stamp}.log"
    try:
        env = {"PYTHONPATH": str(SRC_DIR)}
        rc = _run_to_file([str(py), "-m", "nba_betting.cli", "export-closing-lines-csv", "--date", d], log_file, cwd=BASE_DIR, env=env)
        out = BASE_DIR / "data" / "processed" / f"closing_lines_{d}.csv"
        rows = 0
        if out.exists():
            try:
                rows = int(sum(1 for _ in out.open("r", encoding="utf-8", errors="ignore")) - 1)
            except Exception:
                try:
                    rows = len(pd.read_csv(out))
                except Exception:
                    rows = 0
        # Optional push
        pushed = None; push_detail = None
        if str(request.args.get("push", "0")).lower() in {"1","true","yes"}:
            ok, detail = _git_commit_and_push(msg=f"capture closing lines {d}")
            pushed = bool(ok); push_detail = detail
        return jsonify({"date": d, "rc": int(rc), "output": str(out), "rows": int(rows), "log_file": str(log_file), "pushed": pushed, "push_detail": push_detail})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cron/predict-date", methods=["POST", "GET"])
def api_cron_predict_date():
    """Run predict-date for the given date to refresh predictions_YYYY-MM-DD.csv and odds CSV.

    Auth: CRON_TOKEN (preferred) or ADMIN_KEY (fallback/manual).
    """
    if not (_cron_auth_ok(request) or _admin_auth_ok(request)):
        return jsonify({"error": "unauthorized"}), 401
    d = _parse_date_param(request, default_to_today=True)
    do_push = (str(request.args.get("push", "0")).lower() in {"1","true","yes"})
    do_async = (str(request.args.get("async", "0")).lower() in {"1","true","yes"})
    skip_if_no_games = (str(request.args.get("skip_if_no_games", "1")).lower() in {"1","true","yes"})
    # If there are no games today and skipping is allowed, fast-exit.
    if skip_if_no_games and d:
        try:
            if not _has_games_for_date(d):
                return jsonify({
                    "status": "skipped",
                    "reason": "no games for date",
                    "date": d,
                    "push": do_push,
                })
        except Exception:
            # On check failure, proceed to run to be safe
            pass
    # Choose python executable
    py = os.environ.get("PYTHON", (os.environ.get("VIRTUAL_ENV") or "") + "/bin/python")
    if not py or not Path(str(py)).exists():
        py_win = (Path(os.environ.get("VIRTUAL_ENV") or "") / "Scripts" / "python.exe")
        py = str(py_win) if py_win.exists() else "python"
    logs_dir = _ensure_logs_dir(); stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"cron_predict_date_{d}_{stamp}.log"
    try:
        env = {"PYTHONPATH": str(SRC_DIR)}
        if do_async:
            # Background job to avoid Render timeouts
            def _job():
                try:
                    _run_to_file([str(py), "-m", "nba_betting.cli", "predict-date", "--date", d], log_file, cwd=BASE_DIR, env=env)
                    if do_push:
                        _git_commit_and_push(msg=f"predict-date {d}")
                except Exception:
                    pass
            t = threading.Thread(target=_job, daemon=True)
            t.start()
            return jsonify({
                "status": "started",
                "date": d,
                "log_file": str(log_file),
                "push": do_push,
            }), 202
        # Synchronous mode (original behavior)
        rc = _run_to_file([str(py), "-m", "nba_betting.cli", "predict-date", "--date", d], log_file, cwd=BASE_DIR, env=env)
        # Locate predictions from either processed/ or legacy root
        pred_path = _find_predictions_for_date(d)
        pred = pred_path if pred_path is not None else (BASE_DIR / f"predictions_{d}.csv")
        odds = BASE_DIR / "data" / "processed" / f"game_odds_{d}.csv"
        n_pred = int(len(pd.read_csv(pred))) if (hasattr(pred, 'exists') and pred.exists()) else (int(len(pd.read_csv(pred))) if pred_path is not None else 0)
        n_odds = int(len(pd.read_csv(odds))) if odds.exists() else 0
        pushed = None; push_detail = None
        if do_push:
            ok, detail = _git_commit_and_push(msg=f"predict-date {d}")
            pushed = bool(ok); push_detail = detail
        return jsonify({
            "date": d,
            "rc": int(rc),
            "predictions": (str(pred) if (hasattr(pred, 'exists') and pred.exists()) else (str(pred) if pred_path is not None else None)),
            "pred_rows": n_pred,
            "odds": str(odds) if odds.exists() else None,
            "odds_rows": n_odds,
            "log_file": str(log_file),
            "pushed": pushed,
            "push_detail": push_detail,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cron/reconcile-games", methods=["POST", "GET"])
def api_cron_reconcile_games():
    """Build reconciliation CSV for a date by joining predictions with final scores.

    - Input date defaults to yesterday (UTC) if omitted.
    - Output: data/processed/recon_games_YYYY-MM-DD.csv
    - Columns: date, home_team, visitor_team, home_pts, visitor_pts, pred_margin, pred_total, actual_margin, total_actual, margin_error, total_error
    """
    if not _cron_auth_ok(request):
        return jsonify({"error": "unauthorized"}), 401
    # Date defaults to yesterday
    d = _parse_date_param(request, default_to_today=False)
    if not d:
        d = (datetime.utcnow().date() - timedelta(days=1)).isoformat()
    # Load predictions for that date
    pred_path = BASE_DIR / f"predictions_{d}.csv"
    if not pred_path.exists():
        # Gracefully return rows=0 so cron runs don't fail on off days or missed predictions
        out = BASE_DIR / "data" / "processed" / f"recon_games_{d}.csv"
        return jsonify({"date": d, "rows": 0, "output": None, "reason": "predictions missing"})
    try:
        preds = pd.read_csv(pred_path)
    except Exception as e:
        return jsonify({"error": f"failed to read predictions: {e}"}), 500
    # Normalize prediction keys to tricodes using nba_api static teams map
    try:
        from nba_api.stats.static import teams as _static_teams  # type: ignore
        team_list = _static_teams.get_teams()
        full_to_abbr = {str(t.get('full_name')).upper(): str(t.get('abbreviation')).upper() for t in team_list}
        # Some common alternate name aliases
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
            # Already a tri?
            if len(s) <= 4:
                return s
            return s
        preds = preds.copy()
        preds["home_tri"] = preds.get("home_team").apply(to_tri)
        preds["away_tri"] = preds.get("visitor_team").apply(to_tri)
    except Exception:
        preds = preds.copy()
        preds["home_tri"] = preds.get("home_team").astype(str).str.upper()
        preds["away_tri"] = preds.get("visitor_team").astype(str).str.upper()
    # Fetch finals from ScoreboardV2
    if _scoreboardv2 is None:
        return jsonify({"error": "nba_api not installed"}), 500
    try:
        # Harden headers for reliability
        try:
            if _nba_http is not None:
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
        # Simple retry/backoff on ScoreboardV2 due to occasional timeouts
        tries = 0
        last_err: Optional[Exception] = None
        gh = pd.DataFrame(); ls = pd.DataFrame()
        while tries < 2:
            try:
                sb = _scoreboardv2.ScoreboardV2(game_date=d, day_offset=0, timeout=45)
                nd = sb.get_normalized_dict()
                gh = pd.DataFrame(nd.get("GameHeader", []))
                ls = pd.DataFrame(nd.get("LineScore", []))
                last_err = None
                break
            except Exception as e:
                last_err = e
                tries += 1
                time.sleep(3)
        if last_err is not None and (gh.empty or ls.empty):
            return jsonify({"error": f"scoreboard fetch failed: {last_err}"}), 502
        out_rows = []
        if not gh.empty and not ls.empty:
            cgh = {c.upper(): c for c in gh.columns}
            cls = {c.upper(): c for c in ls.columns}
            # Build TEAM_ID -> (TRI, PTS)
            team_rows = {}
            for _, r in ls.iterrows():
                try:
                    tid = int(r[cls["TEAM_ID"]])
                    tri = str(r[cls["TEAM_ABBREVIATION"]]).upper()
                    pts = None
                    # PTS sometimes available directly
                    if "PTS" in cls:
                        try:
                            pts = int(r[cls["PTS"]])
                        except Exception:
                            pts = None
                    # Try sum of quarters/OT if needed (optional)
                    team_rows[tid] = {"tri": tri, "pts": pts}
                except Exception:
                    continue
            # For each game, find home/away IDs and map to tri/pts
            for _, g in gh.iterrows():
                try:
                    hid = int(g[cgh["HOME_TEAM_ID"]]); vid = int(g[cgh["VISITOR_TEAM_ID"]])
                    h = team_rows.get(hid, {}); v = team_rows.get(vid, {})
                    htri = str(h.get("tri") or "").upper(); vtri = str(v.get("tri") or "").upper()
                    hpts = h.get("pts"); vpts = v.get("pts")
                    out_rows.append({"home_tri": htri, "away_tri": vtri, "home_pts": hpts, "visitor_pts": vpts})
                except Exception:
                    continue
        finals = pd.DataFrame(out_rows)
        # Join predictions to finals by tri pairs
        if finals.empty:
            return jsonify({"date": d, "rows": 0, "output": None})
        merged = preds.merge(finals, on=["home_tri","away_tri"], how="left")
        # Compute errors
        def to_float(x):
            try:
                return float(x)
            except Exception:
                return None
        merged["pred_margin"] = pd.to_numeric(merged.get("pred_margin"), errors="coerce")
        merged["pred_total"] = pd.to_numeric(merged.get("pred_total"), errors="coerce")
        merged["home_pts"] = pd.to_numeric(merged.get("home_pts"), errors="coerce")
        merged["visitor_pts"] = pd.to_numeric(merged.get("visitor_pts"), errors="coerce")
        merged["actual_margin"] = merged["home_pts"] - merged["visitor_pts"]
        merged["total_actual"] = merged[["home_pts","visitor_pts"]].sum(axis=1)
        merged["margin_error"] = merged["pred_margin"] - merged["actual_margin"]
        merged["total_error"] = merged["pred_total"] - merged["total_actual"]
        # Output tidy columns
        keep = [
            "date","home_team","visitor_team","home_tri","away_tri",
            "home_pts","visitor_pts","pred_margin","pred_total",
            "actual_margin","total_actual","margin_error","total_error"
        ]
        # Ensure date column present
        if "date" not in merged.columns:
            merged["date"] = d
        out_df = merged[keep]
        out = BASE_DIR / "data" / "processed" / f"recon_games_{d}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out, index=False)
        # Optional push
        pushed = None; push_detail = None
        if str(request.args.get("push", "0")).lower() in {"1","true","yes"}:
            ok, detail = _git_commit_and_push(msg=f"reconcile games {d}")
            pushed = bool(ok); push_detail = detail
        try:
            _cron_meta_update("reconcile_games", {"date": d, "rows": int(len(out_df)), "output": str(out), "pushed": pushed})
        except Exception:
            pass
        return jsonify({"date": d, "rows": int(len(out_df)), "output": str(out), "pushed": pushed, "push_detail": push_detail})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/cron/daily-update", methods=["POST", "GET"])
def api_cron_daily_update():
    """Trigger the daily update job via cron token. Git push is disabled by default for cron."""
    if not _cron_auth_ok(request):
        return jsonify({"error": "unauthorized"}), 401
    if _job_state["running"]:
        return jsonify({"status": "already-running", "started_at": _job_state["started_at"]}), 409
    do_push = (str(request.args.get("push", "0")).lower() in {"1", "true", "yes"})
    t = threading.Thread(target=_daily_update_job, args=(do_push,), daemon=True)
    t.start()
    return jsonify({"status": "started", "push": do_push, "started_at": datetime.utcnow().isoformat()}), 202


@app.route("/api/cron/run-all", methods=["POST", "GET"])
def api_cron_run_all():
    """Composite daily cron mirroring NHL behavior: predict, fetch odds, reconcile, props actuals/edges.

    Steps (best-effort, continue on failures):
      1) Predict for today's slate
      2) Refresh Bovada odds (or fallback to any available odds file)
      3) Reconcile yesterday's games (to post finals)
      4) Props actuals upsert for yesterday
      5) Compute props edges for today (if supported)
    """
    if not _cron_auth_ok(request):
        return jsonify({"error": "unauthorized"}), 401
    today = datetime.utcnow().date()
    yesterday = today - timedelta(days=1)
    d_today = request.args.get("date") or today.isoformat()
    d_yest = request.args.get("yesterday") or yesterday.isoformat()
    push = (str(request.args.get("push", "0")).lower() in {"1","true","yes"})
    # Choose python exe
    py = os.environ.get("PYTHON", (os.environ.get("VIRTUAL_ENV") or "") + "/bin/python")
    if not py or not Path(str(py)).exists():
        py_win = (Path(os.environ.get("VIRTUAL_ENV") or "") / "Scripts" / "python.exe")
        py = str(py_win) if py_win.exists() else "python"
    env = {"PYTHONPATH": str(SRC_DIR)}
    logdir = _ensure_logs_dir(); stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_file = logdir / f"web_daily_update_{stamp}.log"
    results: Dict[str, Any] = {"date": d_today, "yesterday": d_yest, "log_file": str(log_file)}
    try:
        # 1) predict-date today
        rc1 = _run_to_file([str(py), "-m", "nba_betting.cli", "predict-date", "--date", d_today], log_file, cwd=BASE_DIR, env=env)
        results["predict_date"] = int(rc1)
        # 2) refresh-bovada for today (HTTP to our own endpoint, to centralize logic/meta)
        try:
            import requests as _rq
            token = os.environ.get("CRON_TOKEN")
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            r = _rq.get(f"http://127.0.0.1:5000/api/cron/refresh-bovada?date={d_today}", headers=headers, timeout=60)
            results["refresh_bovada"] = (r.status_code, r.text[:200])
        except Exception as e:
            results["refresh_bovada_error"] = str(e)
        # 3) reconcile-games for yesterday
        try:
            import requests as _rq
            token = os.environ.get("CRON_TOKEN")
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            r = _rq.get(f"http://127.0.0.1:5000/api/cron/reconcile-games?date={d_yest}", headers=headers, timeout=60)
            results["reconcile_games"] = (r.status_code, r.text[:200])
        except Exception as e:
            results["reconcile_games_error"] = str(e)
        # 4) props actuals upsert for yesterday via CLI
        try:
            rc4 = _run_to_file([str(py), "-m", "nba_betting.cli", "props-actuals", "--date", d_yest], log_file, cwd=BASE_DIR, env=env)
            results["props_actuals"] = int(rc4)
        except Exception as e:
            results["props_actuals_error"] = str(e)
        # 5) compute props edges for today via CLI if available
        try:
            rc5 = _run_to_file([str(py), "-m", "nba_betting.cli", "props-edges", "--date", d_today, "--source", "auto"], log_file, cwd=BASE_DIR, env=env)
            results["props_edges"] = int(rc5)
        except Exception as e:
            results["props_edges_error"] = str(e)
        if push:
            ok, detail = _git_commit_and_push(msg=f"daily-run-all {d_today}")
            results["pushed"] = bool(ok)
            results["push_detail"] = detail
        _cron_meta_update("run_all", {"date": d_today, "yesterday": d_yest, "log_file": str(log_file)})
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e), **results}), 500


@app.route("/api/cron/config")
def api_cron_config():
    """Introspection for cron/admin configuration (safe to expose booleans only)."""
    try:
        return jsonify({
            "have_cron_token": bool(os.environ.get("CRON_TOKEN")),
            "have_admin_key": bool(os.environ.get("ADMIN_KEY")),
            "endpoints": [
                "/api/cron/refresh-bovada",
                "/api/cron/predict-date",
                "/api/cron/capture-closing",
                "/api/cron/reconcile-games",
                "/api/cron/daily-update",
            ],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/daily-update", methods=["POST", "GET"])
def api_admin_daily_update():
    if not _admin_auth_ok(request):
        return jsonify({"error": "unauthorized"}), 401
    if _job_state["running"]:
        return jsonify({"status": "already-running", "started_at": _job_state["started_at"]}), 409
    do_push = (str(request.args.get("push", "1")).lower() in {"1", "true", "yes"})
    t = threading.Thread(target=_daily_update_job, args=(do_push,), daemon=True)
    t.start()
    return jsonify({"status": "started", "push": do_push, "started_at": datetime.utcnow().isoformat()}), 202


@app.route("/api/admin/daily-update/status")
def api_admin_daily_update_status():
    if not _admin_auth_ok(request):
        return jsonify({"error": "unauthorized"}), 401
    try:
        tail = int(request.args.get("tail", "200"))
    except Exception:
        tail = 200
    logs = _job_state.get("logs", [])
    if tail > 0:
        logs = logs[-tail:]
    return jsonify({
        "running": _job_state["running"],
        "started_at": _job_state["started_at"],
        "ended_at": _job_state["ended_at"],
        "ok": _job_state["ok"],
        "log_file": _job_state.get("log_file"),
        "logs": logs,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    app.run(host="0.0.0.0", port=port, debug=False)
