from __future__ import annotations

import base64
import os
from pathlib import Path

from flask import Flask, jsonify, redirect, request, send_from_directory
import threading
import subprocess
import shlex
import time
from datetime import datetime

# Optional: load environment variables from a .env file if present
try:  # lightweight optional dependency
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

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
