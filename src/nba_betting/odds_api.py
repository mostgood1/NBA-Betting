from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

import pandas as pd
import requests
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type

from .config import paths
from .teams import normalize_team


ODDS_HOST = "https://api.the-odds-api.com"
NBA_SPORT_KEY = "basketball_nba"
# OddsAPI lists the NBA PRESEASON under its own sport key (like
# `icehockey_nhl_preseason`); `basketball_nba` is empty until the regular
# season. Game odds list events under both and price each under its own key.
NBA_PRESEASON_SPORT_KEY = "basketball_nba_preseason"
NBA_GAME_SPORT_KEYS = (NBA_SPORT_KEY, NBA_PRESEASON_SPORT_KEY)
# Empty means no bookmaker allowlist. With regions='us', this keeps the full US book set.
DEFAULT_PLAYER_PROP_BOOKMAKERS: tuple[str, ...] = tuple()

_PLAYER_PROP_BOOKMAKER_ALIASES = {
    "fanduel": "fanduel",
    "fd": "fanduel",
    "draftkings": "draftkings",
    "dk": "draftkings",
    "betmgm": "betmgm",
    "mgm": "betmgm",
    "bet365": "bet365",
    "bet365us": "bet365",
}


@dataclass
class OddsApiConfig:
    api_key: str
    regions: str = "us"  # focus on US books
    markets: str = "h2h,spreads,totals"
    odds_format: str = "american"
    bookmakers: str | None = None


def normalize_bookmaker_key(book: object) -> str:
    try:
        raw = str(book or "").strip().lower()
    except Exception:
        return ""
    if not raw:
        return ""
    compact = raw.replace(" ", "").replace("-", "").replace("_", "")
    return _PLAYER_PROP_BOOKMAKER_ALIASES.get(compact, raw)


def resolve_player_prop_bookmakers(raw: str | None = None) -> tuple[str, ...]:
    s = str(raw if raw is not None else (os.environ.get("PLAYER_PROP_BOOKMAKERS") or "")).strip()
    if not s:
        return tuple(DEFAULT_PLAYER_PROP_BOOKMAKERS)
    if s.lower() in {"all", "none", "off", "0", "false", "no"}:
        return tuple()
    out: list[str] = []
    seen: set[str] = set()
    for part in s.split(","):
        bk = normalize_bookmaker_key(part)
        if not bk or bk in seen:
            continue
        seen.add(bk)
        out.append(bk)
    return tuple(out)


def filter_player_prop_bookmakers_df(df: pd.DataFrame, raw: str | None = None) -> pd.DataFrame:
    if df is None or df.empty or "bookmaker" not in df.columns:
        return df
    out = df.copy()
    out["bookmaker"] = out["bookmaker"].astype(str).map(normalize_bookmaker_key)
    allowed = set(resolve_player_prop_bookmakers(raw))
    if not allowed:
        return out
    return out[out["bookmaker"].isin(allowed)].copy()


def player_prop_bookmakers_csv(raw: str | None = None) -> str | None:
    vals = resolve_player_prop_bookmakers(raw)
    return ",".join(vals) if vals else None


def _headers() -> dict:
    return {"Accept": "application/json", "User-Agent": "nba-betting/1.0"}


@retry(retry=retry_if_exception_type(Exception), wait=wait_exponential_jitter(initial=1, max=30), stop=stop_after_attempt(4), reraise=True)
def _get(url: str, params: dict) -> requests.Response:
    r = requests.get(url, params=params, headers=_headers(), timeout=45)
    r.raise_for_status()
    return r


def _iter_dates(start: datetime, end: datetime, step_days: int = 5) -> Iterable[datetime]:
    cur = start
    while cur <= end:
        yield cur
        cur = cur + timedelta(days=step_days)


def _flatten_bookmakers(row: dict, snapshot_ts: str) -> list[dict]:
    out: list[dict] = []
    event_id = row.get("id")
    commence_time = row.get("commence_time")
    home = normalize_team(row.get("home_team", ""))
    away = normalize_team(row.get("away_team", ""))
    for bk in row.get("bookmakers", []) or []:
        bk_key = bk.get("key"); bk_title = bk.get("title")
        for m in bk.get("markets", []) or []:
            mkey = m.get("key"); last_update = m.get("last_update") or bk.get("last_update")
            for oc in m.get("outcomes", []) or []:
                # Prefer player name from description for player_* markets; leave outcome_name as-is (team or Over/Under)
                player_name = oc.get("description")
                if (not player_name) and isinstance(mkey, str) and mkey.startswith("player_"):
                    # Fallback: some payloads are inconsistent; keep something for debugging
                    player_name = oc.get("description") or oc.get("participant") or oc.get("name")
                out.append({
                    "snapshot_ts": snapshot_ts,
                    "event_id": event_id,
                    "commence_time": commence_time,
                    "bookmaker": bk_key,
                    "bookmaker_title": bk_title,
                    "market": mkey,
                    "outcome_name": normalize_team(oc.get("name", "")),
                    "player_name": player_name,
                    "point": oc.get("point"),
                    "price": oc.get("price"),
                    "last_update": last_update,
                    "home_team": home,
                    "away_team": away,
                })
    return out


def fetch_game_odds_current(config: OddsApiConfig, date: datetime, markets: list[str] | None = None, verbose: bool = False) -> pd.DataFrame:
    """Fetch current game odds (h2h, spreads, totals) for events on a given calendar date.

    Uses the events list to filter by US/Eastern calendar day, then fetches per-event odds to ensure late games are included.
    """
    if markets is None:
        markets = [m.strip() for m in (config.markets.split(',') if config.markets else ["h2h","spreads","totals"])]

    # Fetch events (regular season AND preseason) and filter to the target ET
    # calendar day. Each event remembers which sport key listed it, because its
    # odds must be requested under that same key.
    events = []
    for sport_key in NBA_GAME_SPORT_KEYS:
        try:
            ev_resp = _get(f"{ODDS_HOST}/v4/sports/{sport_key}/events", {"apiKey": config.api_key})
            for ev in (ev_resp.json() or []):
                if isinstance(ev, dict):
                    ev.setdefault("sport_key", sport_key)
                    events.append(ev)
        except Exception as e:
            if verbose:
                print(f"[game-odds-current] events request failed for {sport_key}: {e}")
            continue
    if not events:
        return pd.DataFrame()

    target = pd.to_datetime(date).date()
    # Robust Eastern Time conversion: prefer America/New_York, fallback to US/Eastern, else approximate UTC-4/UTC-5 depending on month
    def _et_date(iso_str: str) -> Optional[object]:
        try:
            ct_raw = pd.to_datetime(iso_str, utc=True)
        except Exception:
            return None
        # Try with tz name strings first (pandas can resolve via dateutil without zoneinfo)
        for tzname in ("America/New_York", "US/Eastern"):
            try:
                return ct_raw.tz_convert(tzname).date()
            except Exception:
                continue
        # Last resort: rough offset based on DST (Mar-Nov ~ DST)
        try:
            month = int(ct_raw.month)
            # Assume DST between March (3) and November (11) inclusive
            offset_hours = 4 if 3 <= month <= 11 else 5
            return (ct_raw - pd.Timedelta(hours=offset_hours)).date()
        except Exception:
            return ct_raw.date()

    day_events = []
    for ev in events:
        try:
            ct_et = _et_date(ev.get("commence_time"))
            if ct_et == target:
                day_events.append(ev)
        except Exception:
            continue
    if not day_events:
        if verbose:
            print(f"[game-odds-current] no events on {target}")
        return pd.DataFrame()

    # For each event fetch odds and flatten
    rows: list[dict] = []
    snap = pd.Timestamp.utcnow().isoformat()
    params_common = {
        "apiKey": config.api_key,
        "regions": config.regions,
        "markets": ",".join(markets),
        "oddsFormat": config.odds_format,
    }
    for ev in day_events:
        eid = ev.get("id")
        sport_key = ev.get("sport_key") or NBA_SPORT_KEY
        try:
            r = _get(f"{ODDS_HOST}/v4/sports/{sport_key}/events/{eid}/odds", params_common)
            d = r.json()
            ev_obj = d if isinstance(d, dict) else None
            if not ev_obj:
                continue
            rows.extend(_flatten_bookmakers(ev_obj, snap))
        except Exception as e:
            if verbose:
                print(f"[game-odds-current] event {eid} failed: {e}")
            continue
    return pd.DataFrame(rows)


def list_events_current(config: OddsApiConfig, date: datetime | None = None, verbose: bool = False) -> list[dict]:
    """List upcoming/live NBA events from OddsAPI.

    If date is provided, filters events to the given US/Eastern calendar day.
    This endpoint does not count against usage quota.
    """
    events_url = f"{ODDS_HOST}/v4/sports/{NBA_SPORT_KEY}/events"
    try:
        ev_resp = _get(events_url, {"apiKey": config.api_key})
        events = ev_resp.json() or []
        if not isinstance(events, list):
            return []
    except Exception as e:
        if verbose:
            print(f"[events-current] events request failed: {e}")
        return []

    if date is None:
        return events

    target = pd.to_datetime(date).date()

    def _et_date(iso_str: str) -> Optional[object]:
        try:
            ct_raw = pd.to_datetime(iso_str, utc=True)
        except Exception:
            return None
        for tzname in ("America/New_York", "US/Eastern"):
            try:
                return ct_raw.tz_convert(tzname).date()
            except Exception:
                continue
        try:
            month = int(ct_raw.month)
            offset_hours = 4 if 3 <= month <= 11 else 5
            return (ct_raw - pd.Timedelta(hours=offset_hours)).date()
        except Exception:
            return ct_raw.date()

    day_events: list[dict] = []
    for ev in events:
        try:
            ct_et = _et_date(ev.get("commence_time"))
            if ct_et == target:
                day_events.append(ev)
        except Exception:
            continue
    return day_events


def discover_event_market_keys(config: OddsApiConfig, event_id: str, verbose: bool = False) -> set[str]:
    """Return market keys available for an event, across returned bookmakers."""
    if not event_id:
        return set()
    markets_url = f"{ODDS_HOST}/v4/sports/{NBA_SPORT_KEY}/events/{event_id}/markets"
    try:
        params = {"apiKey": config.api_key, "regions": config.regions}
        if config.bookmakers:
            params["bookmakers"] = config.bookmakers
        r = _get(markets_url, params)
        d = r.json()
        ev_obj = d if isinstance(d, dict) else None
        if not ev_obj:
            return set()
        keys: set[str] = set()
        for bk in ev_obj.get("bookmakers", []) or []:
            for m in bk.get("markets", []) or []:
                k = m.get("key")
                if k:
                    keys.add(str(k))
        return keys
    except Exception as e:
        if verbose:
            print(f"[event-markets] {event_id} failed: {e}")
        return set()


def fetch_event_odds_current(config: OddsApiConfig, event_id: str, markets: list[str], verbose: bool = False) -> pd.DataFrame:
    """Fetch current odds for a single event and flatten to a DataFrame."""
    if not event_id or not markets:
        return pd.DataFrame()
    odds_url = f"{ODDS_HOST}/v4/sports/{NBA_SPORT_KEY}/events/{event_id}/odds"
    params = {
        "apiKey": config.api_key,
        "regions": config.regions,
        "markets": ",".join([m for m in markets if m]),
        "oddsFormat": config.odds_format,
    }
    if config.bookmakers:
        params["bookmakers"] = config.bookmakers
    try:
        r = _get(odds_url, params)
        d = r.json()
        ev_obj = d if isinstance(d, dict) else None
        if not ev_obj:
            return pd.DataFrame()
        snap = pd.Timestamp.utcnow().isoformat()
        return pd.DataFrame(_flatten_bookmakers(ev_obj, snap))
    except requests.HTTPError as he:
        if verbose:
            code = he.response.status_code if he.response is not None else None
            print(f"[event-odds] {event_id} HTTP {code} for markets={markets}")
        return pd.DataFrame()
    except Exception as e:
        if verbose:
            print(f"[event-odds] {event_id} failed: {e}")
        return pd.DataFrame()


def fetch_player_props_current(config: OddsApiConfig, date: datetime, markets: list[str] | None = None, verbose: bool = False) -> pd.DataFrame:
    """Fetch current player props for events on a given calendar date using the event odds endpoint.

    Steps:
    - GET /v4/sports/{sport}/events to list upcoming/live events
    - Filter events whose commence_time date matches requested date (US/Eastern day)
    - For each event id, GET /v4/sports/{sport}/events/{eventId}/odds with player markets
    """
    if markets is None:
        markets = [
            # Core modeled markets
            "player_points",
            "player_rebounds",
            "player_assists",
            "player_points_rebounds_assists",
            "player_threes",
            # Additional markets requested
            "player_steals",
            "player_blocks",
            "player_turnovers",
            "player_points_rebounds",
            "player_points_assists",
            "player_rebounds_assists",
            "player_double_double",
            "player_triple_double",
        ]

    # List events then filter by Eastern calendar day
    events_url = f"{ODDS_HOST}/v4/sports/{NBA_SPORT_KEY}/events"
    try:
        ev_resp = _get(events_url, {"apiKey": config.api_key})
        events = ev_resp.json() or []
    except Exception as e:
        if verbose:
            print(f"[player-props-current] events request failed: {e}")
        return pd.DataFrame()

    target = pd.to_datetime(date).date()
    def _et_date(iso_str: str) -> Optional[object]:
        try:
            ct_raw = pd.to_datetime(iso_str, utc=True)
        except Exception:
            return None
        for tzname in ("America/New_York", "US/Eastern"):
            try:
                return ct_raw.tz_convert(tzname).date()
            except Exception:
                continue
        try:
            month = int(ct_raw.month)
            offset_hours = 4 if 3 <= month <= 11 else 5
            return (ct_raw - pd.Timedelta(hours=offset_hours)).date()
        except Exception:
            return ct_raw.date()

    day_events = []
    for ev in events:
        try:
            ct_et = _et_date(ev.get("commence_time"))
            if ct_et == target:
                day_events.append(ev)
        except Exception:
            continue
    if not day_events:
        if verbose:
            print(f"[player-props-current] no events on {target}")
        return pd.DataFrame()

    rows: list[dict] = []
    debug_rows: list[dict] = []
    snap = pd.Timestamp.utcnow().isoformat()
    odds_url_tpl = f"{ODDS_HOST}/v4/sports/{NBA_SPORT_KEY}/events/{{event_id}}/odds"
    markets_url_tpl = f"{ODDS_HOST}/v4/sports/{NBA_SPORT_KEY}/events/{{event_id}}/markets"
    bookmakers = str(config.bookmakers or "").strip()

    def _discover_markets(eid: str) -> list[str]:
        try:
            params = {"apiKey": config.api_key, "regions": config.regions}
            if bookmakers:
                params["bookmakers"] = bookmakers
            r = _get(markets_url_tpl.format(event_id=eid), params)
            d = r.json()
            ev_obj = d if isinstance(d, dict) else None
            if not ev_obj:
                return []
            keys = set()
            for bk in ev_obj.get("bookmakers", []) or []:
                for m in bk.get("markets", []) or []:
                    k = m.get("key")
                    if k:
                        keys.add(k)
            desired = set(markets)
            return [k for k in keys if k in desired]
        except Exception:
            return []

    for ev in day_events:
        eid = ev.get("id")
        if not eid:
            continue
        evt_markets = _discover_markets(eid)
        desired_markets = markets or []
        fetched_any = False

        def _fetch_once(mkts: list[str]) -> bool:
            if not mkts:
                return False
            params = {
                "apiKey": config.api_key,
                "regions": config.regions,
                "markets": ",".join(mkts),
                "oddsFormat": config.odds_format,
            }
            if bookmakers:
                params["bookmakers"] = bookmakers
            try:
                r = _get(odds_url_tpl.format(event_id=eid), params)
                d = r.json()
                ev_obj = d if isinstance(d, dict) else None
                if not ev_obj:
                    return False
                rows.extend(_flatten_bookmakers(ev_obj, snap))
                return True
            except requests.HTTPError as he:
                if verbose:
                    code = he.response.status_code if he.response is not None else None
                    print(f"[player-props-current] event {eid} HTTP {code} for markets={mkts}; will try fallback granularity")
                return False
            except Exception as e:
                if verbose:
                    print(f"[player-props-current] event {eid} failed: {e}")
                return False

        # Try discovery as a batch
        req_markets = evt_markets if evt_markets else []
        if req_markets and _fetch_once(req_markets):
            fetched_any = True
        else:
            # Conservative core set
            core = [
                "player_points",
                "player_rebounds",
                "player_assists",
                "player_threes",
                "player_points_rebounds_assists",
            ]
            if desired_markets:
                core = [m for m in core if m in desired_markets]
            if _fetch_once(core):
                fetched_any = True
            else:
                # One-by-one
                probe_list = evt_markets or (desired_markets if desired_markets else core)
                for mkt in probe_list:
                    if _fetch_once([mkt]):
                        fetched_any = True

        # Record per-event diagnostics
        try:
            debug_rows.append({
                "event_id": eid,
                "commence_time": ev.get("commence_time"),
                "home_team": ev.get("home_team"),
                "away_team": ev.get("away_team"),
                "discovered_markets": ";".join(sorted(set(evt_markets))) if evt_markets else "",
                "fetched_any": bool(fetched_any),
            })
        except Exception:
            pass
        if (not fetched_any) and verbose:
            print(f"[player-props-current] event {eid} returned no player markets")

    df = pd.DataFrame(rows)
    # Persist diagnostics and snapshot to processed for inspection
    try:
        paths.data_processed.mkdir(parents=True, exist_ok=True)
        day_str = pd.to_datetime(date).date().isoformat()
        pd.DataFrame(debug_rows).to_csv(paths.data_processed / f"oddsapi_event_markets_{day_str}.csv", index=False)
        df.to_csv(paths.data_processed / f"oddsapi_player_props_{day_str}.csv", index=False)
    except Exception:
        pass
    return df
def backfill_player_props(config: OddsApiConfig, date: datetime, markets: list[str] | None = None, verbose: bool = False) -> pd.DataFrame:
    """Fetch historical player props snapshot for a given date by querying per-event historical odds.

    Flow:
    1) Get historical events snapshot at the given timestamp.
    2) For each event id, fetch historical event odds for requested player markets.

    Notes:
    - Historical event odds for additional markets (player props, period, alternates) are available after 2023-05-03T05:30:00Z.
    - Usage cost: 10 x [#unique markets returned] x [#regions] per event.
    """
    paths.data_raw.mkdir(parents=True, exist_ok=True)
    out_parq = paths.data_raw / "odds_nba_player_props.parquet"
    out_csv = paths.data_raw / "odds_nba_player_props.csv"
    if markets is None:
        markets = [
            # Core modeled markets
            "player_points",
            "player_rebounds",
            "player_assists",
            "player_points_rebounds_assists",
            "player_threes",
            # Additional markets requested
            "player_steals",
            "player_blocks",
            "player_turnovers",
            "player_points_rebounds",
            "player_points_assists",
            "player_rebounds_assists",
            "player_double_double",
            "player_triple_double",
        ]

    # Load base (for append + de-dup)
    base = None
    if out_parq.exists():
        try:
            base = pd.read_parquet(out_parq)
        except Exception:
            base = None
    if base is None and out_csv.exists():
        try:
            base = pd.read_csv(out_csv)
        except Exception:
            base = None

    iso = date.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Step 1: historical events at timestamp
    events_url = f"{ODDS_HOST}/v4/historical/sports/{NBA_SPORT_KEY}/events"
    try:
        ev_resp = _get(events_url, {"apiKey": config.api_key, "date": iso})
        ev_data = ev_resp.json()
        snap_ts = ev_data.get("timestamp") if isinstance(ev_data, dict) else None
        events = ev_data.get("data") if isinstance(ev_data, dict) else ev_data
        if not events:
            if verbose:
                print(f"[props] no events at {iso}")
            return base if base is not None else pd.DataFrame()
    except Exception as e:
        if verbose:
            print(f"[props] events lookup failed: {e}")
        return base if base is not None else pd.DataFrame()

    rows: list[dict] = []

    def _flatten_event_odds(ev_obj: dict, snapshot_ts: str | None):
        eid = ev_obj.get("id"); commence_time = ev_obj.get("commence_time")
        home = normalize_team(ev_obj.get("home_team", "")); away = normalize_team(ev_obj.get("away_team", ""))
        for bk in ev_obj.get("bookmakers", []) or []:
            bk_key = bk.get("key"); bk_title = bk.get("title")
            for m in bk.get("markets", []) or []:
                mkey = m.get("key"); last_update = m.get("last_update") or bk.get("last_update")
                for oc in m.get("outcomes", []) or []:
                    # For props: oc.name Over/Under; oc.description = player name (when available)
                    rows.append({
                        "snapshot_ts": snapshot_ts or iso,
                        "event_id": eid,
                        "commence_time": commence_time,
                        "bookmaker": bk_key,
                        "bookmaker_title": bk_title,
                        "market": mkey,
                        "outcome_name": oc.get("name"),  # Over/Under
                        "player_name": oc.get("description") or oc.get("name"),
                        "point": oc.get("point"),
                        "price": oc.get("price"),
                        "last_update": last_update,
                        "home_team": home,
                        "away_team": away,
                    })

    # Step 2: per-event historical event odds
    odds_url_tpl = f"{ODDS_HOST}/v4/historical/sports/{NBA_SPORT_KEY}/events/{{event_id}}/odds"
    params_common = {
        "apiKey": config.api_key,
        "regions": config.regions,
        "markets": ",".join(markets),
        "oddsFormat": config.odds_format,
        "date": iso,
    }
    for ev in events:
        eid = ev.get("id")
        if not eid:
            continue
        try:
            r = _get(odds_url_tpl.format(event_id=eid), params_common)
            d = r.json()
            evt_ts = d.get("timestamp") if isinstance(d, dict) else None
            ev_obj = d.get("data") if isinstance(d, dict) else d
            if isinstance(ev_obj, dict):
                _flatten_event_odds(ev_obj, evt_ts)
            else:
                # Unexpected shape; skip
                if verbose:
                    print(f"[props] unexpected event odds payload for {eid}")
        except requests.HTTPError as he:
            # Skip events without these markets at the timestamp
            if verbose:
                code = he.response.status_code if he.response is not None else None
                print(f"[props] event {eid} HTTP {code}; skipping")
            continue
        except Exception as e:
            if verbose:
                print(f"[props] event {eid} failed: {e}")
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return base if base is not None else df

    # Append/merge de-dup
    if base is not None and not base.empty:
        keep = [
            "snapshot_ts","event_id","commence_time","bookmaker","bookmaker_title",
            "market","outcome_name","player_name","point","price","last_update","home_team","away_team"
        ]
        base = base[keep] if all(c in base.columns for c in keep) else base
        merged = pd.concat([base, df], ignore_index=True)
        merged.drop_duplicates(subset=["snapshot_ts","event_id","bookmaker","market","player_name","point","outcome_name"], keep="last", inplace=True)
        out_df = merged
    else:
        out_df = df

    out_df.to_csv(out_csv, index=False)
    try:
        out_df.to_parquet(out_parq, index=False)
    except Exception:
        pass
    return out_df


def backfill_historical_odds(config: OddsApiConfig, start_date: datetime, end_date: datetime, step_days: int = 5, verbose: bool = False) -> pd.DataFrame:
    """Pull OddsAPI historical odds snapshots across a date range and persist a long table.

    Note: Historical odds availability starts around 2020-06-06 per docs; earlier dates will return empty snapshots.
    We step through snapshots every `step_days`, capturing bookmaker odds for h2h, spreads, and totals.
    """
    paths.data_raw.mkdir(parents=True, exist_ok=True)
    out_parq = paths.data_raw / "odds_nba.parquet"
    out_csv = paths.data_raw / "odds_nba.csv"

    # Load existing to avoid duplicates
    base = None
    if out_parq.exists():
        try:
            base = pd.read_parquet(out_parq)
        except Exception:
            base = None
    if base is None and out_csv.exists():
        try:
            base = pd.read_csv(out_csv)
        except Exception:
            base = None

    rows: list[dict] = []
    for ts in _iter_dates(start_date, end_date, step_days=step_days):
        iso = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        if verbose:
            print(f"[odds] Snapshot {iso} ...")
        url = f"{ODDS_HOST}/v4/historical/sports/{NBA_SPORT_KEY}/odds"
        params = {
            "apiKey": config.api_key,
            "regions": config.regions,
            "markets": config.markets,
            "oddsFormat": config.odds_format,
            "date": iso,
        }
        try:
            r = _get(url, params)
            data = r.json()
            snap_ts = data.get("timestamp") if isinstance(data, dict) else None
            events = data.get("data") if isinstance(data, dict) else data
            if not events:
                continue
            for ev in events:
                rows.extend(_flatten_bookmakers(ev, snap_ts or iso))
        except requests.HTTPError as he:
            # 402/403 or quota issues: stop early
            if verbose:
                print(f"[odds] HTTP {he}")
            if he.response is not None and he.response.status_code in (401, 402, 403):
                break
        except Exception as e:
            if verbose:
                print(f"[odds] error: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        return base if base is not None else df

    # Append/merge
    if base is not None and not base.empty:
        # de-duplicate on snapshot_ts,event_id,bookmaker,market,outcome_name,point
        keep_cols = [
            "snapshot_ts","event_id","commence_time","bookmaker","bookmaker_title","market",
            "outcome_name","point","price","last_update","home_team","away_team"
        ]
        base = base[keep_cols] if all(c in base.columns for c in keep_cols) else base
        merged = pd.concat([base, df], ignore_index=True)
        merged.drop_duplicates(subset=["snapshot_ts","event_id","bookmaker","market","outcome_name","point"], keep="last", inplace=True)
        out_df = merged
    else:
        out_df = df

    out_df.to_csv(out_csv, index=False)
    try:
        out_df.to_parquet(out_parq, index=False)
    except Exception:
        pass
    return out_df


def consensus_lines_at_close(odds_df: pd.DataFrame) -> pd.DataFrame:
    """Compute approximate closing consensus per event by taking the latest snapshot per event/bookmaker and aggregating.

    For spreads/totals, average price at the modal point; for h2h, average price per side.
    Returns a wide per-event row with columns: home_ml, away_ml, home_spread, total, etc., using American prices and points.
    """
    if odds_df is None or odds_df.empty:
        return pd.DataFrame()
    df = odds_df.copy()
    # Compute last snapshot per bookmaker/event/market with correct keys per market type.
    df["snapshot_dt"] = pd.to_datetime(df["snapshot_ts"]) if "snapshot_ts" in df.columns else pd.NaT
    df.sort_values(["event_id","bookmaker","market","outcome_name","point","snapshot_dt"], inplace=True)
    h2h_last = df[df["market"].isin(["h2h","h2h_lay"])].groupby([
        "event_id","bookmaker","market","outcome_name"
    ], as_index=False, sort=False).tail(1)
    sp_last = df[df["market"] == "spreads"].groupby([
        "event_id","bookmaker","market","outcome_name","point"
    ], as_index=False, sort=False).tail(1)
    tot_last = df[df["market"] == "totals"].groupby([
        "event_id","bookmaker","market","outcome_name","point"
    ], as_index=False, sort=False).tail(1)

    # --- Price aggregation helpers ---
    # NOTE: Averaging American odds directly is invalid and can yield nonsense like -30.
    # We aggregate via implied probabilities, optionally normalizing the two sides per bookmaker.
    def _to_float(x) -> Optional[float]:
        try:
            v = pd.to_numeric(x, errors="coerce")
            fv = float(v)
            return fv if pd.notna(v) and pd.notna(fv) and pd.isna(pd.NA) is False else fv
        except Exception:
            try:
                fv = float(x)
                return fv if pd.notna(fv) else None
            except Exception:
                return None

    def _implied_prob_from_american_or_prob(price: Optional[float]) -> Optional[float]:
        """Convert American odds -> implied prob.

        Also supports already-probabilistic inputs in (0,1).
        """
        try:
            if price is None:
                return None
            p = float(price)
            if not pd.notna(p):
                return None
            # Already a probability
            if 0.0 < p < 1.0:
                return float(np.clip(p, 1e-6, 1.0 - 1e-6))

            # American odds
            if p > 0:
                ip = 100.0 / (p + 100.0)
            elif p < 0:
                ip = (-p) / ((-p) + 100.0)
            else:
                return None
            if not np.isfinite(ip) or ip <= 0.0 or ip >= 1.0:
                return None
            return float(np.clip(ip, 1e-6, 1.0 - 1e-6))
        except Exception:
            return None

    def _american_from_implied_prob(prob: Optional[float]) -> Optional[float]:
        """Convert implied prob -> American odds (float).

        Uses standard convention:
        - favorites (p>=0.5): negative
        - underdogs (p<0.5): positive
        """
        try:
            if prob is None:
                return None
            p = float(prob)
            if not np.isfinite(p) or p <= 0.0 or p >= 1.0:
                return None
            if p >= 0.5:
                ml = -100.0 * p / (1.0 - p)
            else:
                ml = 100.0 * (1.0 - p) / p
            if not np.isfinite(ml):
                return None
            return float(ml)
        except Exception:
            return None

    def _prob_mean_from_prices(series: pd.Series) -> Optional[float]:
        try:
            vals = pd.to_numeric(series, errors="coerce").dropna().astype(float).tolist()
            probs = [p for p in (_implied_prob_from_american_or_prob(v) for v in vals) if p is not None]
            if not probs:
                return None
            return float(np.mean(probs))
        except Exception:
            return None

    # h2h -> moneyline (aggregate via implied probs; normalize within each bookmaker)
    h2h = h2h_last.copy()
    try:
        h2h["price_num"] = pd.to_numeric(h2h["price"], errors="coerce")
    except Exception:
        h2h["price_num"] = pd.NA
    # Pivot per book to normalize the two sides (removes book vig effects on averaging)
    try:
        hp = h2h[h2h["outcome_name"] == h2h["home_team"]][["event_id", "bookmaker", "price_num"]].rename(columns={"price_num": "home_price"})
        ap = h2h[h2h["outcome_name"] == h2h["away_team"]][["event_id", "bookmaker", "price_num"]].rename(columns={"price_num": "away_price"})
        pv = hp.merge(ap, on=["event_id", "bookmaker"], how="inner")
        pv["p_home"] = pv["home_price"].map(_implied_prob_from_american_or_prob)
        pv["p_away"] = pv["away_price"].map(_implied_prob_from_american_or_prob)
        # normalize if both present
        s = pv[["p_home", "p_away"]].sum(axis=1)
        pv.loc[s > 0, "p_home"] = pv.loc[s > 0, "p_home"] / s[s > 0]
        pv.loc[s > 0, "p_away"] = pv.loc[s > 0, "p_away"] / s[s > 0]
        p_home_cons = pv.groupby("event_id")["p_home"].mean()
        p_away_cons = pv.groupby("event_id")["p_away"].mean()
        h2h_home = p_home_cons.map(_american_from_implied_prob)
        h2h_away = p_away_cons.map(_american_from_implied_prob)
    except Exception:
        # Fallback: per-side prob mean without normalization
        h2h_home = h2h[h2h["outcome_name"] == h2h["home_team"]].groupby("event_id")["price"].apply(lambda s: _american_from_implied_prob(_prob_mean_from_prices(s)))
        h2h_away = h2h[h2h["outcome_name"] == h2h["away_team"]].groupby("event_id")["price"].apply(lambda s: _american_from_implied_prob(_prob_mean_from_prices(s)))

    # spreads -> choose modal point per event and average price for both sides at that point
    sp = sp_last.copy()
    # Identify which outcome corresponds to home (price at negative point tends to be home favorite)
    # Odds API outcomes use team names; map to side by matching to home/away
    sp_home = sp[sp["outcome_name"] == sp["home_team"]]
    sp_away = sp[sp["outcome_name"] == sp["away_team"]]
    sp_mode_point = sp_home.groupby("event_id")["point"].agg(lambda x: x.mode().iloc[0] if not pd.isna(x).all() and len(pd.Series(x).mode())>0 else pd.NA)
    sp_at_mode = sp_home.merge(sp_mode_point.rename("mode_point"), left_on="event_id", right_index=True)
    # Avoid event_id being both index and column post-merge
    sp_at_mode = sp_at_mode.reset_index(drop=True)
    sp_at_mode = sp_at_mode[sp_at_mode["point"] == sp_at_mode["mode_point"]]

    # For away price, align to the same modal point rows for away side
    sp_at_mode_away = sp_away.merge(sp_mode_point.rename("mode_point"), left_on="event_id", right_index=True)
    sp_at_mode_away = sp_at_mode_away.reset_index(drop=True)
    sp_at_mode_away = sp_at_mode_away[sp_at_mode_away["point"] == sp_at_mode_away["mode_point"]]

    # Aggregate spread prices via implied probs (normalize per bookmaker when possible)
    try:
        sp_h = sp_at_mode[["event_id", "bookmaker", "price", "mode_point"]].copy()
        sp_a = sp_at_mode_away[["event_id", "bookmaker", "price", "mode_point"]].copy()
        sp_h["p_home"] = pd.to_numeric(sp_h["price"], errors="coerce").map(_implied_prob_from_american_or_prob)
        sp_a["p_away"] = pd.to_numeric(sp_a["price"], errors="coerce").map(_implied_prob_from_american_or_prob)
        spv = sp_h[["event_id", "bookmaker", "p_home"]].merge(sp_a[["event_id", "bookmaker", "p_away"]], on=["event_id", "bookmaker"], how="inner")
        s = spv[["p_home", "p_away"]].sum(axis=1)
        spv.loc[s > 0, "p_home"] = spv.loc[s > 0, "p_home"] / s[s > 0]
        spv.loc[s > 0, "p_away"] = spv.loc[s > 0, "p_away"] / s[s > 0]
        sp_price_home = spv.groupby("event_id")["p_home"].mean().map(_american_from_implied_prob)
        sp_price_away = spv.groupby("event_id")["p_away"].mean().map(_american_from_implied_prob)
    except Exception:
        sp_price_home = sp_at_mode.groupby("event_id")["price"].apply(lambda s: _american_from_implied_prob(_prob_mean_from_prices(s)))
        sp_price_away = sp_at_mode_away.groupby("event_id")["price"].apply(lambda s: _american_from_implied_prob(_prob_mean_from_prices(s)))

    # totals -> pick modal point; compute Over and Under average prices
    tot = tot_last.copy()
    tot_over = tot[tot["outcome_name"].str.lower() == "over"]
    tot_under = tot[tot["outcome_name"].str.lower() == "under"]
    tot_mode_point = tot_over.groupby("event_id")["point"].agg(lambda x: x.mode().iloc[0] if not pd.isna(x).all() and len(pd.Series(x).mode())>0 else pd.NA)
    tot_at_mode = tot_over.merge(tot_mode_point.rename("mode_point"), left_on="event_id", right_index=True)
    tot_at_mode = tot_at_mode.reset_index(drop=True)
    tot_at_mode = tot_at_mode[tot_at_mode["point"] == tot_at_mode["mode_point"]]

    tot_at_mode_under = tot_under.merge(tot_mode_point.rename("mode_point"), left_on="event_id", right_index=True)
    tot_at_mode_under = tot_at_mode_under.reset_index(drop=True)
    tot_at_mode_under = tot_at_mode_under[tot_at_mode_under["point"] == tot_at_mode_under["mode_point"]]

    # Aggregate totals prices via implied probs (normalize per bookmaker when possible)
    try:
        to = tot_at_mode[["event_id", "bookmaker", "price"]].copy()
        tu = tot_at_mode_under[["event_id", "bookmaker", "price"]].copy()
        to["p_over"] = pd.to_numeric(to["price"], errors="coerce").map(_implied_prob_from_american_or_prob)
        tu["p_under"] = pd.to_numeric(tu["price"], errors="coerce").map(_implied_prob_from_american_or_prob)
        tv = to[["event_id", "bookmaker", "p_over"]].merge(tu[["event_id", "bookmaker", "p_under"]], on=["event_id", "bookmaker"], how="inner")
        s = tv[["p_over", "p_under"]].sum(axis=1)
        tv.loc[s > 0, "p_over"] = tv.loc[s > 0, "p_over"] / s[s > 0]
        tv.loc[s > 0, "p_under"] = tv.loc[s > 0, "p_under"] / s[s > 0]
        tot_price_over = tv.groupby("event_id")["p_over"].mean().map(_american_from_implied_prob)
        tot_price_under = tv.groupby("event_id")["p_under"].mean().map(_american_from_implied_prob)
    except Exception:
        tot_price_over = tot_at_mode.groupby("event_id")["price"].apply(lambda s: _american_from_implied_prob(_prob_mean_from_prices(s)))
        tot_price_under = tot_at_mode_under.groupby("event_id")["price"].apply(lambda s: _american_from_implied_prob(_prob_mean_from_prices(s)))

    # Liquidity metrics: count distinct bookmakers contributing per event
    try:
        h2h_books = (h2h.groupby("event_id")["bookmaker"].nunique() if ("bookmaker" in h2h.columns) else pd.Series(dtype=int))
    except Exception:
        h2h_books = pd.Series(dtype=int)
    try:
        sp_books = (sp_at_mode.groupby("event_id")["bookmaker"].nunique() if ("bookmaker" in sp_at_mode.columns) else pd.Series(dtype=int))
    except Exception:
        sp_books = pd.Series(dtype=int)
    try:
        tot_books = (tot_at_mode.groupby("event_id")["bookmaker"].nunique() if ("bookmaker" in tot_at_mode.columns) else pd.Series(dtype=int))
    except Exception:
        tot_books = pd.Series(dtype=int)
    # Combine: take max across markets as overall books_count
    books_df = pd.DataFrame({
        "books_h2h": h2h_books,
        "books_spreads": sp_books,
        "books_totals": tot_books,
    })
    books_df["books_count"] = books_df.fillna(0)[["books_h2h","books_spreads","books_totals"]].max(axis=1)

    wide = pd.DataFrame({
        "home_ml": h2h_home,
        "away_ml": h2h_away,
        "home_spread_price": sp_price_home,
        "away_spread_price": sp_price_away,
        "spread_point": sp_mode_point,
        "total_over_price": tot_price_over,
        "total_under_price": tot_price_under,
        "total_point": tot_mode_point,
    })
    # Attach team names and commence
    # Build meta using union of subsets
    meta_source = pd.concat([h2h, sp, tot], ignore_index=True) if not (h2h.empty and sp.empty and tot.empty) else df
    meta = meta_source.groupby("event_id").agg({
        "home_team":"last","away_team":"last","commence_time":"last"
    })
    wide = meta.join(wide, how="left")
    # Attach liquidity counts
    try:
        wide = wide.join(books_df, how="left")
    except Exception:
        pass
    wide.reset_index(inplace=True)
    return wide
