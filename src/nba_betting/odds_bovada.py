from __future__ import annotations

import pandas as pd
import requests
from datetime import datetime
from typing import Any

from .teams import normalize_team


ENDPOINTS = [
    # Region A (Americas), description feed; NBA main
    "https://www.bovada.lv/services/sports/event/v2/events/A/description/basketball/nba",
    # Some slates may also appear under basketball/usa/nba; keep as fallback
    "https://www.bovada.lv/services/sports/event/v2/events/A/description/basketball/usa/nba",
    # Preseason sometimes appears here
    "https://www.bovada.lv/services/sports/event/v2/events/A/description/basketball/nba-preseason",
]


def _safe_get(d: dict, *keys: str, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is default:
            return default
    return cur


def _extract_markets(ev: dict) -> dict:
    """Extract Moneyline, Spread, and Total from Bovada event JSON.

    Returns dict with keys: home_ml, away_ml, home_spread, away_spread, total.
    Values may be None if not present.
    """
    out = {"home_ml": None, "away_ml": None, "home_spread": None, "away_spread": None, "total": None}
    dgs = ev.get("displayGroups", []) or []
    # Flatten markets
    for dg in dgs:
        for m in dg.get("markets", []) or []:
            mtype = (m.get("description") or m.get("marketType") or "").lower()
            # Moneyline
            if "moneyline" in mtype:
                for oc in m.get("outcomes", []) or []:
                    typ = (oc.get("type") or oc.get("description") or "").lower()
                    price = _safe_get(oc, "price", "american")
                    if price is None:
                        continue
                    try:
                        price = int(str(price).replace("+", "")) if str(price).startswith("+") else int(price)
                    except Exception:
                        continue
                    if typ == "home":
                        out["home_ml"] = price
                    elif typ == "away":
                        out["away_ml"] = price
            # Point spread
            elif "spread" in mtype or "point spread" in mtype:
                # Home/Away outcomes with handicap
                for oc in m.get("outcomes", []) or []:
                    typ = (oc.get("type") or oc.get("description") or "").lower()
                    handicap = _safe_get(oc, "price", "handicap") or oc.get("handicap")
                    try:
                        hval = float(handicap) if handicap is not None else None
                    except Exception:
                        hval = None
                    if hval is None:
                        continue
                    if typ == "home":
                        out["home_spread"] = hval
                    elif typ == "away":
                        out["away_spread"] = hval
            # Game Total
            elif "total" in mtype:
                # Total outcomes with over/under; handicap is the total line
                for oc in m.get("outcomes", []) or []:
                    handicap = _safe_get(oc, "price", "handicap") or oc.get("handicap")
                    try:
                        hval = float(handicap) if handicap is not None else None
                    except Exception:
                        hval = None
                    if hval is None:
                        continue
                    out["total"] = hval
    return out


def fetch_bovada_odds_current(date: datetime, verbose: bool = False) -> pd.DataFrame:
    """Fetch current game odds from Bovada for events on the given calendar date (UTC date match).

    Returns a normalized DataFrame with columns:
      - date, commence_time, home_team, away_team, home_ml, away_ml, home_spread, away_spread, total, bookmaker
    """
    target = pd.to_datetime(date).date()
    rows: list[dict] = []
    payloads = []
    for url in ENDPOINTS:
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "nba-betting/1.0"})
            if r.ok:
                payloads.append(r.json())
        except Exception as e:
            if verbose:
                print(f"[bovada] {url} failed: {e}")
            continue
    # payloads are lists of category arrays; each entry may contain "events"
    for p in payloads:
        try:
            cats = p if isinstance(p, list) else []
            for cat in cats:
                for sport in cat or []:
                    events = sport.get("events", []) if isinstance(sport, dict) else []
                    for ev in events:
                        try:
                            ct = pd.to_datetime(ev.get("startTime")).date() if ev.get("startTime") else None
                        except Exception:
                            ct = None
                        if ct != target:
                            continue
                        comps = ev.get("competitors", []) or []
                        home_name = None; away_name = None
                        for c in comps:
                            nm = c.get("name") or c.get("team") or c.get("abbreviation")
                            if c.get("home") is True:
                                home_name = nm
                            else:
                                # Bovada marks non-home as away (there should be 2 comps)
                                away_name = nm if away_name is None else away_name
                        if not home_name or not away_name:
                            # fallback from titles
                            title = ev.get("description") or ev.get("name") or ""
                            if " @ " in title:
                                a, h = title.split(" @ ", 1)
                                away_name = away_name or a
                                home_name = home_name or h
                        home = normalize_team(str(home_name or "").strip())
                        away = normalize_team(str(away_name or "").strip())
                        mk = _extract_markets(ev)
                        rows.append({
                            "date": str(target),
                            "commence_time": ev.get("startTime"),
                            "home_team": home,
                            "visitor_team": away,
                            "home_ml": mk.get("home_ml"),
                            "away_ml": mk.get("away_ml"),
                            "home_spread": mk.get("home_spread"),
                            "away_spread": mk.get("away_spread"),
                            "total": mk.get("total"),
                            "bookmaker": "bovada",
                        })
        except Exception:
            continue
    return pd.DataFrame(rows)
