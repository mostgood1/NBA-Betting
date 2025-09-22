from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Iterable

import requests

ROOT = Path(__file__).resolve().parents[1]
TEAMS_JSON = ROOT / 'web' / 'assets' / 'teams_nba.json'
OUT_DIR = ROOT / 'web' / 'assets' / 'logos'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36',
    'Accept': 'image/svg+xml,image/*;q=0.9,*/*;q=0.8',
    'Referer': 'https://www.nba.com/',
}

URL_TEMPLATES: list[str] = [
    # Primary marks
    'https://cdn.nba.com/logos/nba/{id}/primary/L/logo.svg',
    'https://cdn.nba.com/logos/nba/{id}/primary/L/logo.png',
    # Global fallback
    'https://cdn.nba.com/logos/nba/{id}/global/L/logo.svg',
    'https://cdn.nba.com/logos/nba/{id}/global/L/logo.png',
]


def try_download(urls: Iterable[str]) -> tuple[bytes | None, str | None]:
    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200 and r.content:
                return r.content, url
        except Exception:
            pass
    return None, None


def main() -> int:
    if not TEAMS_JSON.exists():
        print(f"Teams JSON not found: {TEAMS_JSON}", file=sys.stderr)
        return 1
    data = json.loads(TEAMS_JSON.read_text(encoding='utf-8'))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ok = 0
    miss = 0
    for t in data:
        tri = t.get('tricode')
        tid = t.get('id')
        if not tri or not tid:
            continue
        # Try download
        urls = [u.format(id=tid) for u in URL_TEMPLATES]
        content, used = try_download(urls)
        if content is None:
            print(f"MISS {tri} ({tid})")
            miss += 1
            continue
        # Determine extension from URL
        ext = '.svg' if used.endswith('.svg') else '.png'
        out = OUT_DIR / f"{tri}{ext}"
        out.write_bytes(content)
        print(f"OK   {tri} -> {out.name}")
        ok += 1
        time.sleep(0.2)  # be nice

    print(f"\nDone. Downloaded={ok}, Missing={miss}. Saved to {OUT_DIR}")
    print("Note: Keep these assets local unless you have rights to distribute.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
