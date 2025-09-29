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
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

try:
    # Optional (for scoreboard)
    from nba_api.stats.endpoints import scoreboardv2 as _scoreboardv2
    from nba_api.stats.library import http as _nba_http
except Exception:  # pragma: no cover
    _scoreboardv2 = None  # type: ignore
    _nba_http = None  # type: ignore

try:
    # local package for odds fetching
    from nba_betting.odds_bovada import fetch_bovada_odds_current as _fetch_bovada_odds_current  # type: ignore
except Exception:  # pragma: no cover
    _fetch_bovada_odds_current = None  # type: ignore

# Optional: load environment variables from a .env file if present
try:  # lightweight optional dependency
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

# Ensure local package in src/ is importable (for schedule and CLI imports)
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Serve the static frontend under /web and redirect / -> /web/
app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="/web")


@app.route("/")
def root():
    # Make the NBA slate the homepage (explicitly target index.html under /web)
    return redirect("/web/index.html")


@app.route("/web/")
def web_index():
    return send_from_directory(str(WEB_DIR), "index.html")


@app.route("/web/<path:path>")
def web_static(path: str):
    # Serve any static asset in web/
    return send_from_directory(str(WEB_DIR), path)


@app.route("/data/<path:path>")
def data_static(path: str):
    # Serve data files (JSON/CSV) referenced by the frontend, e.g., /data/processed/*.json
    data_dir = BASE_DIR / "data"
    return send_from_directory(str(data_dir), path)


@app.route("/health")
def health():
    # Lightweight health/status
    try:
        exists = (WEB_DIR / "index.html").exists()
        return jsonify({"status": "ok", "have_index": bool(exists)}), 200
    except Exception as e:  # noqa: BLE001
        return jsonify({"status": "error", "error": str(e)}), 500


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
        env = {"PYTHONPATH": str(BASE_DIR)}
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
            rc = _run_to_file(c, log_file, cwd=BASE_DIR)
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
    # Prefer root predictions_YYYY-MM-DD.csv
    p = BASE_DIR / f"predictions_{date_str}.csv"
    if p.exists():
        return p
    # Fallbacks
    for cand in sorted(BASE_DIR.glob("predictions_*.csv")):
        if date_str in cand.name:
            return cand
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
        preds_p = BASE_DIR / f"props_predictions_{d}.csv"
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
            merged = df.merge(o, on=["date","home_team","visitor_team"], how="left", suffixes=("","_odds")) if not o.empty else df
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
    Auth: same as admin (uses ADMIN_KEY if set; otherwise allows local/LAN).
    """
    if not _admin_auth_ok(request):
        return jsonify({"error": "unauthorized"}), 401
    d = _parse_date_param(request, default_to_today=False)
    if not d:
        return jsonify({"error": "missing date"}), 400
    if _fetch_bovada_odds_current is None:
        return jsonify({"error": "bovada fetcher not available"}), 500
    try:
        dt = pd.to_datetime(d).date()
    except Exception:
        return jsonify({"error": "invalid date"}), 400
    try:
        df = _fetch_bovada_odds_current(pd.to_datetime(dt))
        rows = 0 if df is None else len(df)
        out = BASE_DIR / "data" / "processed" / f"game_odds_{d}.csv"
        if df is not None and not df.empty:
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
        return jsonify({"date": d, "rows": int(rows), "output": str(out)})
    except Exception as e:
        return jsonify({"error": f"bovada fetch failed: {e}"}), 500


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
