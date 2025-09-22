from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type

from .config import paths
from .teams import normalize_team


ODDS_HOST = "https://api.the-odds-api.com"
NBA_SPORT_KEY = "basketball_nba"


@dataclass
class OddsApiConfig:
    api_key: str
    regions: str = "us"  # focus on US books
    markets: str = "h2h,spreads,totals"
    odds_format: str = "american"


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
                out.append({
                    "snapshot_ts": snapshot_ts,
                    "event_id": event_id,
                    "commence_time": commence_time,
                    "bookmaker": bk_key,
                    "bookmaker_title": bk_title,
                    "market": mkey,
                    "outcome_name": normalize_team(oc.get("name", "")),
                    "point": oc.get("point"),
                    "price": oc.get("price"),
                    "last_update": last_update,
                    "home_team": home,
                    "away_team": away,
                })
    return out


def fetch_game_odds_current(config: OddsApiConfig, date: datetime, markets: list[str] | None = None, verbose: bool = False) -> pd.DataFrame:
    """Fetch current game odds (h2h, spreads, totals) for events on a given calendar date.

    Calls /v4/sports/{sport}/odds and filters events by commence_time date.
    """
    if markets is None:
        markets = [m.strip() for m in (config.markets.split(',') if config.markets else ["h2h","spreads","totals"])]
    url = f"{ODDS_HOST}/v4/sports/{NBA_SPORT_KEY}/odds"
    params = {
        "apiKey": config.api_key,
        "regions": config.regions,
        "markets": ",".join(markets),
        "oddsFormat": config.odds_format,
    }
    try:
        r = _get(url, params)
        data = r.json() or []
    except Exception as e:
        if verbose:
            print(f"[game-odds-current] request failed: {e}")
        return pd.DataFrame()
    target = pd.to_datetime(date).date()
    snap = pd.Timestamp.utcnow().isoformat()
    rows: list[dict] = []
    for ev in data:
        try:
            ct = pd.to_datetime(ev.get("commence_time")).date()
            if ct != target:
                continue
            rows.extend(_flatten_bookmakers(ev, snap))
        except Exception:
            continue
    return pd.DataFrame(rows)


def fetch_player_props_current(config: OddsApiConfig, date: datetime, markets: list[str] | None = None, verbose: bool = False) -> pd.DataFrame:
    """Fetch current player props for events on a given calendar date using the event odds endpoint.

    Steps:
    - GET /v4/sports/{sport}/events to list upcoming/live events
    - Filter events whose commence_time date matches requested date
    - For each event id, GET /v4/sports/{sport}/events/{eventId}/odds with player markets
    """
    if markets is None:
        markets = [
            "player_points",
            "player_rebounds",
            "player_assists",
            "player_pr_points_rebounds_assists",
            "player_three_pointers",
        ]
    events_url = f"{ODDS_HOST}/v4/sports/{NBA_SPORT_KEY}/events"
    try:
        ev_resp = _get(events_url, {"apiKey": config.api_key})
        events = ev_resp.json()
    except Exception as e:
        if verbose:
            print(f"[props-current] events request failed: {e}")
        return pd.DataFrame()
    # Normalize date filter
    target = pd.to_datetime(date).date()
    events = events or []
    day_events = []
    for ev in events:
        try:
            ct = pd.to_datetime(ev.get("commence_time")).date()
            if ct == target:
                day_events.append(ev)
        except Exception:
            continue
    if not day_events:
        if verbose:
            print(f"[props-current] no events found on {target}")
        return pd.DataFrame()

    rows: list[dict] = []
    odds_url_tpl = f"{ODDS_HOST}/v4/sports/{NBA_SPORT_KEY}/events/{{event_id}}/odds"
    params_common = {
        "apiKey": config.api_key,
        "regions": config.regions,
        "markets": ",".join(markets),
        "oddsFormat": config.odds_format,
    }
    for ev in day_events:
        eid = ev.get("id"); commence_time = ev.get("commence_time")
        home = normalize_team(ev.get("home_team", "")); away = normalize_team(ev.get("away_team", ""))
        try:
            r = _get(odds_url_tpl.format(event_id=eid), params_common)
            d = r.json()
            # Response is a single event object for current event odds
            ev_obj = d if isinstance(d, dict) else None
            if not ev_obj:
                continue
            for bk in ev_obj.get("bookmakers", []) or []:
                bk_key = bk.get("key"); bk_title = bk.get("title")
                for m in bk.get("markets", []) or []:
                    mkey = m.get("key"); last_update = m.get("last_update") or bk.get("last_update")
                    for oc in m.get("outcomes", []) or []:
                        rows.append({
                            "snapshot_ts": pd.Timestamp.utcnow().isoformat(),
                            "event_id": eid,
                            "commence_time": commence_time,
                            "bookmaker": bk_key,
                            "bookmaker_title": bk_title,
                            "market": mkey,
                            "outcome_name": oc.get("name"),
                            "player_name": oc.get("description") or oc.get("name"),
                            "point": oc.get("point"),
                            "price": oc.get("price"),
                            "last_update": last_update,
                            "home_team": home,
                            "away_team": away,
                        })
        except requests.HTTPError as he:
            if verbose:
                code = he.response.status_code if he.response is not None else None
                print(f"[props-current] event {eid} HTTP {code}; skipping")
            continue
        except Exception as e:
            if verbose:
                print(f"[props-current] event {eid} failed: {e}")
            continue
    return pd.DataFrame(rows)
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
            "player_points",
            "player_rebounds",
            "player_assists",
            "player_pr_points_rebounds_assists",
            "player_three_pointers",
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

    # Pivot helpers
    def american_mean(series):
        try:
            return float(pd.to_numeric(series, errors="coerce").dropna().mean())
        except Exception:
            return pd.NA

    # h2h -> moneyline
    h2h = h2h_last.copy()
    # Note: outcome_name here is the team name; group by event and average across books
    h2h_home = h2h[h2h["outcome_name"] == h2h["home_team"]].groupby("event_id")["price"].apply(american_mean)
    h2h_away = h2h[h2h["outcome_name"] == h2h["away_team"]].groupby("event_id")["price"].apply(american_mean)

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
    sp_price_home = sp_at_mode.groupby("event_id")["price"].apply(american_mean)
    # For away price, align to the same modal point rows for away side
    sp_at_mode_away = sp_away.merge(sp_mode_point.rename("mode_point"), left_on="event_id", right_index=True)
    sp_at_mode_away = sp_at_mode_away.reset_index(drop=True)
    sp_at_mode_away = sp_at_mode_away[sp_at_mode_away["point"] == sp_at_mode_away["mode_point"]]
    sp_price_away = sp_at_mode_away.groupby("event_id")["price"].apply(american_mean)

    # totals -> pick modal point; compute Over and Under average prices
    tot = tot_last.copy()
    tot_over = tot[tot["outcome_name"].str.lower() == "over"]
    tot_under = tot[tot["outcome_name"].str.lower() == "under"]
    tot_mode_point = tot_over.groupby("event_id")["point"].agg(lambda x: x.mode().iloc[0] if not pd.isna(x).all() and len(pd.Series(x).mode())>0 else pd.NA)
    tot_at_mode = tot_over.merge(tot_mode_point.rename("mode_point"), left_on="event_id", right_index=True)
    tot_at_mode = tot_at_mode.reset_index(drop=True)
    tot_at_mode = tot_at_mode[tot_at_mode["point"] == tot_at_mode["mode_point"]]
    tot_price_over = tot_at_mode.groupby("event_id")["price"].apply(american_mean)
    tot_at_mode_under = tot_under.merge(tot_mode_point.rename("mode_point"), left_on="event_id", right_index=True)
    tot_at_mode_under = tot_at_mode_under.reset_index(drop=True)
    tot_at_mode_under = tot_at_mode_under[tot_at_mode_under["point"] == tot_at_mode_under["mode_point"]]
    tot_price_under = tot_at_mode_under.groupby("event_id")["price"].apply(american_mean)

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
    wide.reset_index(inplace=True)
    return wide
