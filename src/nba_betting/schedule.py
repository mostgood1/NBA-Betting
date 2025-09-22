from __future__ import annotations

from datetime import datetime
from typing import Any
import pandas as pd


def team_last_game_dates(games: pd.DataFrame) -> dict[str, datetime]:
    df = games.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date")
    last_dates: dict[str, datetime] = {}
    for _, row in df.iterrows():
        d = row["date"]
        last_dates[row["home_team"]] = d
        last_dates[row["visitor_team"]] = d
    return last_dates


def compute_rest_for_matchups(matchups: pd.DataFrame, history_games: pd.DataFrame) -> pd.DataFrame:
    """Adds rest_days and b2b flags for home/visitor using last game dates from history.

    Expects matchups columns: date, home_team, visitor_team
    """
    out = matchups.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    last_dates = team_last_game_dates(history_games)
    home_rest = []
    away_rest = []
    home_b2b = []
    away_b2b = []
    for _, r in out.iterrows():
        d = r["date"]
        h = r["home_team"]
        v = r["visitor_team"]
        ld_h = last_dates.get(h)
        ld_v = last_dates.get(v)
        rd_h = (d - ld_h).days if ld_h is not None else None
        rd_v = (d - ld_v).days if ld_v is not None else None
        home_rest.append(rd_h)
        away_rest.append(rd_v)
        home_b2b.append(1 if rd_h == 1 else 0 if rd_h is not None else None)
        away_b2b.append(1 if rd_v == 1 else 0 if rd_v is not None else None)
    out["home_rest_days"] = home_rest
    out["visitor_rest_days"] = away_rest
    out["home_b2b"] = home_b2b
    out["visitor_b2b"] = away_b2b
    return out


def _request_schedule_payload() -> dict[str, Any]:
    """Fetch schedule JSON payload from cdn.nba.com with a simple fallback.

    Returns the parsed JSON dict. Raises on failure.
    """
    import requests
    urls = [
        "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json",
        "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json",
    ]
    last_err: Exception | None = None
    for url in urls:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # pragma: no cover - network
            last_err = e
            continue
    raise RuntimeError(f"Failed to fetch NBA schedule payload: {last_err}")


def fetch_schedule_2025_26() -> pd.DataFrame:
    """Fetch and normalize the 2025-26 NBA schedule from the public CDN feed.

    Returns a DataFrame with a stable schema suitable for frontend consumption:
    - game_id (str)
    - season_year (str like '2025-26')
    - game_label (e.g., 'Regular Season', 'Preseason', 'Playoffs')
    - game_subtype (e.g., 'PlayIn', 'InSeasonTournament' when applicable)
    - game_status (int) and game_status_text (str)
    - date_utc (YYYY-MM-DD) and time_utc (HH:MM) and datetime_utc (ISO8601)
    - date_est, time_est, datetime_est (local Eastern)
    - home_team_id, home_tricode, home_city, home_name
    - away_team_id, away_tricode, away_city, away_name
    - arena_name, arena_city, arena_state
    - broadcasters_national (pipe-delimited)
    """
    payload = _request_schedule_payload()
    ls = payload.get("leagueSchedule", {})
    season_year = ls.get("seasonYear") or "2025-26"
    game_dates = ls.get("gameDates", []) or []

    rows: list[dict[str, Any]] = []
    for gd in game_dates:
        games = gd.get("games", []) or []
        for g in games:
            # Basic fields with safe access
            game_id = str(g.get("gameId"))
            game_label = g.get("gameLabel")
            game_subtype = g.get("gameSubtype")
            game_status = g.get("gameStatus")
            game_status_text = g.get("gameStatusText")

            # Times
            dt_utc = g.get("gameDateTimeUTC") or g.get("gameTimeUTC")
            date_utc = g.get("gameDateUTC")
            time_utc = g.get("gameTimeUTC")
            dt_est = g.get("gameDateTimeEst") or g.get("gameTimeEst")
            date_est = g.get("gameDateEst")
            time_est = g.get("gameTimeEst")

            # Teams
            h = g.get("homeTeam", {}) or {}
            a = g.get("awayTeam", {}) or {}
            home_team_id = h.get("teamId")
            home_tricode = h.get("teamTricode")
            home_city = h.get("teamCity")
            home_name = h.get("teamName")
            away_team_id = a.get("teamId")
            away_tricode = a.get("teamTricode")
            away_city = a.get("teamCity")
            away_name = a.get("teamName")

            # Arena
            arena_name = g.get("arenaName")
            arena_city = g.get("arenaCity")
            arena_state = g.get("arenaState")

            # Broadcasters: national
            nat = []
            try:
                for b in (g.get("broadcasters", {}) or {}).get("national", []) or []:
                    name = b.get("broadcasterName")
                    if name:
                        nat.append(str(name))
            except Exception:
                pass
            broadcasters_national = " | ".join(nat) if nat else None

            rows.append({
                "game_id": game_id,
                "season_year": season_year,
                "game_label": game_label,
                "game_subtype": game_subtype,
                "game_status": game_status,
                "game_status_text": game_status_text,
                "date_utc": date_utc,
                "time_utc": time_utc,
                "datetime_utc": dt_utc,
                "date_est": date_est,
                "time_est": time_est,
                "datetime_est": dt_est,
                "home_team_id": home_team_id,
                "home_tricode": home_tricode,
                "home_city": home_city,
                "home_name": home_name,
                "away_team_id": away_team_id,
                "away_tricode": away_tricode,
                "away_city": away_city,
                "away_name": away_name,
                "arena_name": arena_name,
                "arena_city": arena_city,
                "arena_state": arena_state,
                "broadcasters_national": broadcasters_national,
            })

    df = pd.DataFrame(rows)
    # Normalize date columns to useful types for downstream if needed
    for col in ("datetime_utc", "datetime_est"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ("date_utc", "date_est"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df
