from __future__ import annotations

import base64
import os
from pathlib import Path

from flask import Flask, jsonify, redirect, send_from_directory


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

# Serve the static frontend under /web and redirect / -> /web/
app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="/web")


@app.route("/")
def root():
    # Make the NBA slate the homepage (keeps relative paths working)
    return redirect("/web/")


@app.route("/web/")
def web_index():
    return send_from_directory(str(WEB_DIR), "index.html")


@app.route("/web/<path:path>")
def web_static(path: str):
    # Serve any static asset in web/
    return send_from_directory(str(WEB_DIR), path)


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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    app.run(host="0.0.0.0", port=port, debug=False)
