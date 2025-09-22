from __future__ import annotations

from datetime import timedelta
import pandas as pd
from collections import deque, defaultdict
import numpy as np

from .elo import Elo, EloConfig


def build_features(games: pd.DataFrame) -> pd.DataFrame:
    df = games.copy()
    # ensure date is datetime
    df["date"] = pd.to_datetime(df["date"]) 
    df = df.sort_values(["date"])  # chronological

    # rest days
    for side, team_col in (("home", "home_team"), ("visitor", "visitor_team")):
        last_dates = {}
        rests = []
        for _, row in df.iterrows():
            team = row[team_col]
            d = row["date"].normalize()
            prev = last_dates.get(team)
            rest = (d - prev).days if prev is not None else None
            rests.append(rest)
            last_dates[team] = d
        df[f"{side}_rest_days"] = rests
        df[f"{side}_b2b"] = (df[f"{side}_rest_days"] == 1).astype("Int64")

    # Elo ratings pre-game
    elo = Elo(EloConfig())
    home_elo = []
    away_elo = []

    for _, row in df.iterrows():
        home_elo.append(elo.get(row["home_team"]))
        away_elo.append(elo.get(row["visitor_team"]))
        # update after game if final scores are present
        if pd.notna(row.get("home_pts")) and pd.notna(row.get("visitor_pts")):
            try:
                elo.update_game(row["home_team"], row["visitor_team"], int(row["home_pts"]), int(row["visitor_pts"]))
            except Exception:
                pass

    df["home_elo"] = home_elo
    df["visitor_elo"] = away_elo
    df["elo_diff"] = df["home_elo"] - df["visitor_elo"]

    # basic matchup features
    df["is_home"] = 1

    # Rolling form features (last 5 games) and schedule intensity
    team_state = {}
    # For counting recent games in time windows, store recent dates per team
    recent_dates = defaultdict(lambda: deque(maxlen=20))
    # Deques to compute PF/PA rolling means
    pf_hist = defaultdict(lambda: deque(maxlen=5))
    pa_hist = defaultdict(lambda: deque(maxlen=5))

    def mean_or_nan(dq):
        return float(np.mean(dq)) if len(dq) > 0 else np.nan

    h_off_5 = []
    h_def_5 = []
    v_off_5 = []
    v_def_5 = []
    h_g_last3 = []
    v_g_last3 = []
    h_g_last5 = []
    v_g_last5 = []
    h_3in4 = []
    v_3in4 = []
    h_4in6 = []
    v_4in6 = []

    for _, row in df.iterrows():
        d = pd.to_datetime(row["date"]).normalize()
        h = row["home_team"]
        v = row["visitor_team"]

        # Pre-game rolling means
        h_off_5.append(mean_or_nan(pf_hist[h]))
        h_def_5.append(mean_or_nan(pa_hist[h]))
        v_off_5.append(mean_or_nan(pf_hist[v]))
        v_def_5.append(mean_or_nan(pa_hist[v]))

        # Schedule intensity counts (prior days only)
        def count_recent(team, days):
            return sum(1 for x in recent_dates[team] if 0 < (d - x).days <= days)

        hg3 = count_recent(h, 3)
        vg3 = count_recent(v, 3)
        hg5 = count_recent(h, 5)
        vg5 = count_recent(v, 5)
        h_g_last3.append(hg3)
        v_g_last3.append(vg3)
        h_g_last5.append(hg5)
        v_g_last5.append(vg5)
        # Fatigue flags: playing today would make it 3-in-4 (>=2 in last 3) or 4-in-6 (>=3 in last 5)
        h_3in4.append(1 if hg3 >= 2 else 0)
        v_3in4.append(1 if vg3 >= 2 else 0)
        h_4in6.append(1 if hg5 >= 3 else 0)
        v_4in6.append(1 if vg5 >= 3 else 0)

        # Post-game update with final scores if present
        if pd.notna(row.get("home_pts")) and pd.notna(row.get("visitor_pts")):
            try:
                pf_hist[h].append(int(row["home_pts"]))
                pa_hist[h].append(int(row["visitor_pts"]))
                pf_hist[v].append(int(row["visitor_pts"]))
                pa_hist[v].append(int(row["home_pts"]))
                recent_dates[h].append(d)
                recent_dates[v].append(d)
            except Exception:
                pass

    df["home_form_off_5"] = h_off_5
    df["home_form_def_5"] = h_def_5
    df["visitor_form_off_5"] = v_off_5
    df["visitor_form_def_5"] = v_def_5
    df["home_games_last3"] = h_g_last3
    df["visitor_games_last3"] = v_g_last3
    df["home_games_last5"] = h_g_last5
    df["visitor_games_last5"] = v_g_last5
    df["home_3in4"] = h_3in4
    df["visitor_3in4"] = v_3in4
    df["home_4in6"] = h_4in6
    df["visitor_4in6"] = v_4in6

    # targets
    df["target_home_win"] = (df["home_pts"] > df["visitor_pts"]).astype("Int64")
    df["target_margin"] = df["home_pts"] - df["visitor_pts"]
    df["target_total"] = df["home_pts"] + df["visitor_pts"]

    # derivative targets: halves
    for half in ("h1", "h2"):
        if f"home_{half}" in df.columns and f"visitor_{half}" in df.columns:
            df[f"target_{half}_margin"] = df[f"home_{half}"] - df[f"visitor_{half}"]
            df[f"target_{half}_total"] = df[f"home_{half}"] + df[f"visitor_{half}"]
            df[f"target_{half}_home_win"] = (df[f"home_{half}"] > df[f"visitor_{half}"]).astype("Int64")

    # derivative targets: quarters
    for q in ("q1", "q2", "q3", "q4"):
        if f"home_{q}" in df.columns and f"visitor_{q}" in df.columns:
            df[f"target_{q}_margin"] = df[f"home_{q}"] - df[f"visitor_{q}"]
            df[f"target_{q}_total"] = df[f"home_{q}"] + df[f"visitor_{q}"]
            df[f"target_{q}_home_win"] = (df[f"home_{q}"] > df[f"visitor_{q}"]).astype("Int64")

    return df
