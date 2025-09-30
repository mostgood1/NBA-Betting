from __future__ import annotations

import pandas as pd
import requests
from datetime import datetime
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore
from typing import Any

from .teams import normalize_team


_BOVADA_REGIONS = ["A", "B", "C"]  # A: Americas, others as fallback
_BOVADA_BASES = [
    "https://www.bovada.lv/services/sports/event/v2/events/{region}/description/basketball/nba",
    "https://www.bovada.lv/services/sports/event/v2/events/{region}/description/basketball/usa/nba",
    "https://www.bovada.lv/services/sports/event/v2/events/{region}/description/basketball/nba-preseason",
]
_BOVADA_PARAMS = [
    "",
    "?lang=en",
    "?lang=en&preMatchOnly=true&marketFilterId=def",
    "?lang=en&marketFilterId=all",
]
ENDPOINTS = [
    base.format(region=r) + q
    for r in _BOVADA_REGIONS
    for base in _BOVADA_BASES
    for q in _BOVADA_PARAMS
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.bovada.lv",
    "Referer": "https://www.bovada.lv/sports/basketball/nba",
    "Connection": "keep-alive",
}


def _safe_get(d: dict, *keys: str, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is default:
            return default
    return cur


def _extract_markets(ev: dict, *, home_norm: str | None = None, away_norm: str | None = None) -> dict:
    """Extract Moneyline, Spread, and Total from Bovada event JSON for full-game markets.

    Strategy:
    - Prefer markets in display group "Game Lines" or period indicating a full game.
    - First pass: keyword-based detection (broad synonyms) for ML/ATS/TOTAL.
    - Second pass: heuristic fallback by inspecting outcomes shape (e.g., two team outcomes w/o handicap -> ML).
    - Filters out quarters/halves and alternate-line/team totals to avoid bad values.

    Returns dict with keys: home_ml, away_ml, home_spread, away_spread, total, and price fields.
    """
    out = {
        "home_ml": None, "away_ml": None,
        "home_spread": None, "away_spread": None,
        "home_spread_price": None, "away_spread_price": None,
        "total": None, "total_over_price": None, "total_under_price": None,
    }
    
    def _to_american(val) -> int | None:
        try:
            s = str(val)
            if s.startswith("+"):
                s = s[1:]
            return int(s)
        except Exception:
            return None
    
    def _is_full_game(period_desc: str, dg_desc: str) -> bool:
        if "game lines" in dg_desc:
            return True
        pd = period_desc
        return any(k in pd for k in ["game", "match", "full", "regular time", "regulation"]) or (pd.strip() == "")
    
    def _looks_team_total(mtype: str, dg_desc: str) -> bool:
        mt = mtype
        return ("team" in mt) or ("team" in dg_desc)

    dgs = ev.get("displayGroups", []) or []
    # Candidates collected across markets, then reduced to best pick
    def _dg_priority(desc: str) -> int:
        d = desc.lower()
        if ("game lines" in d) or ("game" in d and "lines" in d) or ("main" in d):
            return 0
        return 1
    ml_cands: list[dict] = []  # each: {"home": int, "away": int, "prio": int}
    spread_cands: dict[float, dict] = {}  # abs(handicap) -> {"home_h": float, "away_h": float, "home_price": int, "away_price": int, "prio": int}
    total_cands: dict[float, dict] = {}   # total -> {"over_price": int, "under_price": int, "prio": int}
    # Flatten markets
    for dg in dgs:
        dg_desc = str(dg.get("description") or dg.get("name") or "").lower()
        # Prefer "game lines"/full game groups; still allow others if period filter passes
        for m in dg.get("markets", []) or []:
            mtype = (m.get("description") or m.get("marketType") or "").lower()
            # Skip obvious non-full-game markets
            period = m.get("period") or {}
            period_desc = str(period.get("description") or period.get("abbreviation") or "").lower()
            if any(k in period_desc for k in ["quarter", "1st quarter", "2nd quarter", "3rd quarter", "4th quarter", "half", "1st half", "2nd half"]):
                continue
            if "alternate" in mtype:
                # Avoid alternate lines to keep a single canonical line
                continue
            # If display group doesn't look like full game, require period_desc that looks like game/match
            if ("game lines" not in dg_desc) and not any(k in period_desc for k in ["game", "match", "full", "regular time", "regulation"]):
                # Allow totals/spreads named as game types as a fallback
                if not any(k in mtype for k in [
                    "moneyline", "money line", "match odds", "match result", "to win",
                    "head to head", "h2h", "winner", "game winner", "match winner",
                    "spread", "point spread", "handicap", "line", "against the spread", "ats",
                    "total", "totals", "total points", "over/under", "o/u", "game total", "points total"
                ]):
                    continue
            # Moneyline (keep names tight to avoid picking race/winner props)
            is_ml = any(k in mtype for k in [
                "moneyline", "money line", "match odds", "match result", "to win",
                "head to head", "h2h"
            ]) and not any(k in mtype for k in ["race", "first to", "margin", "exact", "by "])
            if is_ml:
                temp = {"home": None, "away": None}
                for oc in m.get("outcomes", []) or []:
                    typ = (oc.get("type") or "").lower()
                    desc = str(oc.get("description") or oc.get("name") or oc.get("competitor") or "").strip().lower()
                    price = _to_american(_safe_get(oc, "price", "american"))
                    if price is None:
                        continue
                    if typ in ("home", "h") or (typ == "" and home_norm and (home_norm in desc)):
                        temp["home"] = price
                    elif typ in ("away", "a") or (typ == "" and away_norm and (away_norm in desc)):
                        temp["away"] = price
                if temp["home"] is not None and temp["away"] is not None:
                    temp["prio"] = _dg_priority(dg_desc)
                    ml_cands.append(temp)
            # Point spread
            elif any(k in mtype for k in ["spread", "point spread", "handicap", "against the spread", "ats"]) and ("player" not in mtype):
                if _looks_team_total(mtype, dg_desc):
                    # safeguard: don't confuse team totals w/ spreads
                    pass
                # Home/Away outcomes with handicap
                for oc in m.get("outcomes", []) or []:
                    typ = (oc.get("type") or "").lower()
                    desc = str(oc.get("description") or oc.get("name") or oc.get("competitor") or "").strip().lower()
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
                        spr_price = _to_american(s) if s is not None else None
                    except Exception:
                        spr_price = None
                    key = round(abs(hval), 1)
                    rec = spread_cands.setdefault(key, {"home_h": None, "away_h": None, "home_price": None, "away_price": None, "prio": _dg_priority(dg_desc)})
                    if typ in ("home", "h") or (typ == "" and home_norm and (home_norm in desc)):
                        rec["home_h"] = hval
                        if spr_price is not None:
                            rec["home_price"] = spr_price
                    elif typ in ("away", "a") or (typ == "" and away_norm and (away_norm in desc)):
                        rec["away_h"] = hval
                        if spr_price is not None:
                            rec["away_price"] = spr_price
            # Game Total (exclude team totals and player/prop totals)
            elif any(k in mtype for k in ["total", "totals", "total points", "over/under", "o/u", "game total", "points total"]) and not any(k in mtype for k in ["team", "player", "race", "alt"]):
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
                    # Only set total when period indicates a game/match or in a group named game lines
                    if ("game lines" not in dg_desc) and not any(k in period_desc for k in ["game", "match", "full", "regular"]):
                        continue
                    # Exclude team totals by display group name
                    if "team" in dg_desc:
                        continue
                    # Sanity: ignore implausible NBA full-game totals (likely half/quarter/team totals slipped through)
                    if hval < 150 or hval > 330:
                        continue
                    rec = total_cands.setdefault(round(hval,1), {"over_price": None, "under_price": None, "prio": _dg_priority(dg_desc)})
                    # capture over/under prices
                    try:
                        s = price_obj.get("american")
                        if s is not None:
                            pr = _to_american(s)
                            if "over" in typ:
                                rec["over_price"] = pr
                            elif "under" in typ:
                                rec["under_price"] = pr
                    except Exception:
                        pass
            # Heuristic fallbacks when keywords are missing
            else:
                if not _is_full_game(period_desc, dg_desc):
                    continue
                outs = list(m.get("outcomes", []) or [])
                if not outs:
                    continue
                # Detect TOTAL: outcomes mention over/under and handicap in plausible range
                types_str = " ".join([(o.get("type") or o.get("description") or "").lower() for o in outs])
                if any(k in types_str for k in ["over", "under"]) and not _looks_team_total(mtype, dg_desc):
                    for oc in outs:
                        price_obj = oc.get("price") or {}
                        handicap = _safe_get(oc, "price", "handicap") or oc.get("handicap")
                        try:
                            hval = float(handicap) if handicap is not None else None
                        except Exception:
                            hval = None
                        if hval is None or hval < 150 or hval > 330:
                            continue
                        rec = total_cands.setdefault(round(hval,1), {"over_price": None, "under_price": None, "prio": _dg_priority(dg_desc)})
                        pr = _to_american(price_obj.get("american"))
                        t = (oc.get("type") or oc.get("description") or "").lower()
                        if pr is not None:
                            if "over" in t:
                                rec["over_price"] = pr
                            elif "under" in t:
                                rec["under_price"] = pr
                # Detect SPREAD: outcomes for both teams with handicap within range
                if out["home_spread"] is None or out["away_spread"] is None:
                    for oc in outs:
                        price_obj = oc.get("price") or {}
                        handicap = _safe_get(oc, "price", "handicap") or oc.get("handicap")
                        try:
                            hval = float(handicap) if handicap is not None else None
                        except Exception:
                            hval = None
                        if hval is None or abs(hval) > 50:
                            continue
                        typ = (oc.get("type") or "").lower()
                        desc = str(oc.get("description") or oc.get("name") or oc.get("competitor") or "").strip().lower()
                        pr = _to_american((price_obj or {}).get("american"))
                        key = round(abs(hval), 1)
                        rec = spread_cands.setdefault(key, {"home_h": None, "away_h": None, "home_price": None, "away_price": None})
                        if (typ in ("home", "h")) or (home_norm and home_norm in desc):
                            rec["home_h"] = hval
                            if pr is not None:
                                rec["home_price"] = pr
                        elif (typ in ("away", "a")) or (away_norm and away_norm in desc):
                            rec["away_h"] = hval
                            if pr is not None:
                                rec["away_price"] = pr
                # Detect ML: two team outcomes without handicap values
                if out["home_ml"] is None or out["away_ml"] is None:
                    # Ensure most outcomes have no handicap
                    no_hcap = [o for o in outs if _safe_get(o, "price", "handicap") is None and o.get("handicap") is None]
                    if len(no_hcap) >= 2:
                        temp = {"home": None, "away": None}
                        for oc in no_hcap:
                            typ = (oc.get("type") or "").lower()
                            desc = str(oc.get("description") or oc.get("name") or oc.get("competitor") or "").strip().lower()
                            pr = _to_american(_safe_get(oc, "price", "american"))
                            if pr is None:
                                continue
                            if (typ in ("home", "h")) or (home_norm and home_norm in desc):
                                temp["home"] = pr
                            elif (typ in ("away", "a")) or (away_norm and away_norm in desc):
                                temp["away"] = pr
                        if temp["home"] is not None and temp["away"] is not None:
                            ml_cands.append(temp)
    # Choose best ML candidate (first good one)
    if ml_cands:
        # Prefer display groups that look like main/game lines first
        ml_cands.sort(key=lambda d: (d.get("prio", 1), ))
        out["home_ml"] = ml_cands[0].get("home")
        out["away_ml"] = ml_cands[0].get("away")
    # Choose best spread candidate: smallest abs handicap then juice closeness to -110
    if spread_cands:
        def spread_score(k: float, rec: dict) -> tuple:
            hp = rec.get("home_price"); ap = rec.get("away_price")
            # distance from -110 on each side (use 200 if missing)
            d = (abs((abs(hp) if hp is not None else 200) - 110) + abs((abs(ap) if ap is not None else 200) - 110))
            return (rec.get("prio", 1), d, abs(k - (sum(spread_cands.keys())/max(len(spread_cands),1))))
        best_key = sorted(((k, v) for k, v in spread_cands.items() if v.get("home_h") is not None and v.get("away_h") is not None), key=lambda kv: spread_score(kv[0], kv[1]))
        if best_key:
            k, rec = best_key[0]
            out["home_spread"] = rec.get("home_h")
            out["away_spread"] = rec.get("away_h")
            out["home_spread_price"] = rec.get("home_price")
            out["away_spread_price"] = rec.get("away_price")
    # Choose best total candidate: require both prices; prefer prices near -110
    if total_cands:
        def total_score(line: float, rec: dict) -> tuple:
            op = rec.get("over_price"); up = rec.get("under_price")
            d = (abs((abs(op) if op is not None else 110) - 110) + abs((abs(up) if up is not None else 110) - 110))
            # prefer mid-range totals as tiebreaker (e.g., 210-240 typical)
            bias = abs(line - 225.0)
            return (rec.get("prio", 1), d, bias)
        # Accept lines even if one price missing; pick best by prio then juice closeness
        ln, rc = sorted(total_cands.items(), key=lambda it: total_score(it[0], it[1]))[0]
        out["total"] = ln
        out["total_over_price"] = rc.get("over_price")
        out["total_under_price"] = rc.get("under_price")
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


def fetch_bovada_odds_current(date: datetime | str, verbose: bool = False) -> pd.DataFrame:
    """Fetch current game odds from Bovada for events on the given calendar date (UTC date match).

    Returns a normalized DataFrame with columns:
      - date, commence_time, home_team, away_team, home_ml, away_ml, home_spread, away_spread, total, bookmaker
    """
    # Match by US/Eastern calendar day to align with slate dates; fallback to UTC if tzdata missing
    if ZoneInfo is not None:
        try:
            et = ZoneInfo("US/Eastern")
        except Exception:
            et = None
    else:
        et = None
    # Important: interpret the requested date as the ET calendar day provided by the caller,
    # not as UTC midnight converted to ET (which can underflow to the prior day).
    # Accept a string YYYY-MM-DD and use as-is (no tz involved)
    target_et = pd.to_datetime(str(date)).date()
    rows: list[dict] = []
    payloads = []
    for url in ENDPOINTS:
        try:
            r = requests.get(url, timeout=30, headers=HEADERS)
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
                        if dt is not None:
                            try:
                                ct = dt.tz_convert(et).date() if et is not None else dt.date()
                            except Exception:
                                ct = dt.date()
                        else:
                            ct = None
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
                    # Normalize to lowercase for internal name matching
                    home_l = home.lower()
                    away_l = away.lower()
                    mk = _extract_markets(ev, home_norm=home_l, away_norm=away_l)
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
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Sanity filters
    def _clamp_ml(x):
        try:
            xi = int(x)
            return xi if -20000 <= xi <= 20000 else None
        except Exception:
            return None
    df["home_ml"] = df["home_ml"].apply(_clamp_ml)
    df["away_ml"] = df["away_ml"].apply(_clamp_ml)
    def _ok_spread(x):
        try:
            xv = float(x)
            return xv if -50 <= xv <= 50 else None
        except Exception:
            return None
    for c in ["home_spread","away_spread"]:
        df[c] = df[c].apply(_ok_spread)
    def _ok_total(x):
        try:
            xv = float(x)
            return xv if 150 <= xv <= 330 else None
        except Exception:
            return None
    df["total"] = df["total"].apply(_ok_total)
    # De-duplicate: choose one row per game with the most complete data
    keep_cols = [
        "date","commence_time","home_team","visitor_team",
        "home_ml","away_ml","home_spread","away_spread",
        "home_spread_price","away_spread_price",
        "total","total_over_price","total_under_price","bookmaker"
    ]
    df = df[keep_cols]
    def pick_best(group: pd.DataFrame) -> pd.Series:
        # Score completeness: count non-nulls in core fields; prefer rows with total present
        core = ["home_ml","away_ml","home_spread","away_spread","total"]
        g = group.copy()
        g["_score"] = g[core].notna().sum(axis=1) + g["total"].notna().astype(int)
        # Prefer realistic totals closer to median of group to avoid outliers if multiple totals present
        if g["total"].notna().any():
            med = g["total"].median(skipna=True)
            g["_score"] += (g["total"].notna()).astype(int)
            # small penalty for totals far from median
            try:
                g["_score"] -= (abs(g["total"] - med) / 10.0).fillna(0)
            except Exception:
                pass
        # Highest score wins; tie-breaker earliest commence_time
        g = g.sort_values(["_score","commence_time"], ascending=[False, True])
        best = g.iloc[0].drop(labels=["_score"]) if "_score" in g.columns else g.iloc[0]
        # Fill missing odds from others in the group if possible
        for col in ["home_ml","away_ml","home_spread","away_spread","total",
                    "home_spread_price","away_spread_price","total_over_price","total_under_price"]:
            if pd.isna(best[col]):
                vals = g[col].dropna()
                if not vals.empty:
                    best[col] = vals.iloc[0]
        return best
    df = df.groupby(["date","home_team","visitor_team"], as_index=False, sort=False).apply(pick_best)
    # The groupby-apply can produce a multi-index in older pandas; ensure clean DataFrame
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    # Ensure bookmaker is set
    if "bookmaker" not in df.columns:
        df["bookmaker"] = "bovada"
    return df.reset_index(drop=True)


def probe_bovada(date: datetime | str, verbose: bool = False) -> dict:
    """Probe Bovada endpoints and report counts for the given date (US/Eastern match).

    Returns a dict with target_date and a list of results per URL including HTTP status and event counts.
    """
    # Compute target ET date similar to fetch function
    if ZoneInfo is not None:
        try:
            et = ZoneInfo("US/Eastern")
        except Exception:
            et = None
    else:
        et = None
    # Interpret requested date as the ET calendar day (no UTC shift)
    target_et = pd.to_datetime(str(date)).date()
    out = {"target_date": str(target_et), "results": []}
    for url in ENDPOINTS:
        ent = {"url": url, "status": None, "events_total": 0, "events_on_date": 0, "error": None}
        try:
            r = requests.get(url, timeout=30, headers=HEADERS)
            ent["status"] = r.status_code
            if r.ok:
                payload = r.json()
                total = 0
                on_date = 0
                first_ts = None
                for events in _walk_event_lists(payload):
                    lst = list(events or [])
                    total += len(lst)
                    for ev in lst:
                        try:
                            dt = _to_dt_utc(ev.get("startTime"))
                            if dt is None:
                                continue
                            if first_ts is None:
                                first_ts = dt.isoformat()
                            try:
                                ct = dt.tz_convert(et).date() if et is not None else dt.date()
                            except Exception:
                                ct = dt.date()
                            if ct == target_et:
                                on_date += 1
                        except Exception:
                            pass
                ent["events_total"] = total
                ent["events_on_date"] = on_date
                ent["first_event_ts_utc"] = first_ts
            else:
                ent["error"] = f"http {r.status_code}"
        except Exception as e:
            ent["error"] = str(e)
        out["results"].append(ent)
    return out
