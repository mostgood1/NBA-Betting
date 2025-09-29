from __future__ import annotations

import pandas as pd
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
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
    out = {
        "home_ml": None, "away_ml": None,
        "home_spread": None, "away_spread": None,
        "home_spread_price": None, "away_spread_price": None,
        "total": None, "total_over_price": None, "total_under_price": None,
    }
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
                    price_obj = oc.get("price") or {}
                    handicap = _safe_get(oc, "price", "handicap") or oc.get("handicap")
                    try:
                        hval = float(handicap) if handicap is not None else None
                    except Exception:
                        hval = None
                    if hval is None:
                        continue
                    # capture juice for EV
                    spr_price = None
                    try:
                        s = price_obj.get("american")
                        if s is not None:
                            spr_price = int(str(s).replace("+", "")) if str(s).startswith("+") else int(s)
                    except Exception:
                        spr_price = None
                    if typ == "home":
                        out["home_spread"] = hval
                        if spr_price is not None:
                            out["home_spread_price"] = spr_price
                    elif typ == "away":
                        out["away_spread"] = hval
                        if spr_price is not None:
                            out["away_spread_price"] = spr_price
            # Game Total
            elif "total" in mtype:
                # Total outcomes with over/under; handicap is the total line
                for oc in m.get("outcomes", []) or []:
                    price_obj = oc.get("price") or {}
                    typ = (oc.get("type") or oc.get("description") or "").lower()
                    handicap = _safe_get(oc, "price", "handicap") or oc.get("handicap")
                    try:
                        hval = float(handicap) if handicap is not None else None
                    except Exception:
                        hval = None
                    if hval is None:
                        continue
                    out["total"] = hval
                    # capture over/under prices
                    try:
                        s = price_obj.get("american")
                        if s is not None:
                            pr = int(str(s).replace("+", "")) if str(s).startswith("+") else int(s)
                            if "over" in typ:
                                out["total_over_price"] = pr
                            elif "under" in typ:
                                out["total_under_price"] = pr
                    except Exception:
                        pass
    return out


def _to_dt_utc(val) -> pd.Timestamp | None:
    try:
        # Bovada often uses epoch millis; handle both ms and ISO strings
        if isinstance(val, (int, float)) or (isinstance(val, str) and val.isdigit()):
            return pd.to_datetime(int(val), unit="ms", utc=True)
        return pd.to_datetime(val, utc=True)
    except Exception:
        return None


def _walk_event_lists(payload: Any):
    """Yield lists of events found anywhere under the payload.

    Bovada responses can be arrays of category dicts; each may contain an 'events' array directly
    or nested under additional arrays. This walker finds any dict with an 'events' key.
    """
    try:
        if isinstance(payload, dict):
            if isinstance(payload.get("events"), list):
                yield payload.get("events")
            # Recurse into values
            for v in payload.values():
                yield from _walk_event_lists(v)
        elif isinstance(payload, list):
            for item in payload:
                yield from _walk_event_lists(item)
    except Exception:
        return


def fetch_bovada_odds_current(date: datetime, verbose: bool = False) -> pd.DataFrame:
    """Fetch current game odds from Bovada for events on the given calendar date (UTC date match).

    Returns a normalized DataFrame with columns:
      - date, commence_time, home_team, away_team, home_ml, away_ml, home_spread, away_spread, total, bookmaker
    """
    # Match by US/Eastern calendar day to align with slate dates
    et = ZoneInfo("US/Eastern")
    ts = pd.to_datetime(date)
    if ts.tzinfo is None:
        target_et = ts.tz_localize(et).date()
    else:
        target_et = ts.tz_convert(et).date()
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
    # Traverse payloads to locate events lists regardless of depth
    for p in payloads:
        try:
            for events in _walk_event_lists(p):
                for ev in (events or []):
                    try:
                        dt = _to_dt_utc(ev.get("startTime"))
                        ct = dt.tz_convert(et).date() if dt is not None else None
                    except Exception:
                        ct = None
                    if ct != target_et:
                        continue
                    comps = ev.get("competitors", []) or []
                    home_name = None; away_name = None
                    for c in comps:
                        nm = c.get("name") or c.get("team") or c.get("abbreviation")
                        # Bovada sometimes uses "home": True or a "position": "H"/"A"
                        is_home = bool(c.get("home") is True or str(c.get("position")).upper() == "H")
                        if is_home:
                            home_name = nm
                        else:
                            away_name = away_name or nm
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
                        "date": str(target_et),
                        "commence_time": dt.isoformat() if dt is not None else ev.get("startTime"),
                        "home_team": home,
                        "visitor_team": away,
                        "home_ml": mk.get("home_ml"),
                        "away_ml": mk.get("away_ml"),
                        "home_spread": mk.get("home_spread"),
                        "away_spread": mk.get("away_spread"),
                        "home_spread_price": mk.get("home_spread_price"),
                        "away_spread_price": mk.get("away_spread_price"),
                        "total": mk.get("total"),
                        "total_over_price": mk.get("total_over_price"),
                        "total_under_price": mk.get("total_under_price"),
                        "bookmaker": "bovada",
                    })
        except Exception:
            continue
    return pd.DataFrame(rows)
