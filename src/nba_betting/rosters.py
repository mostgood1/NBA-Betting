from __future__ import annotations

import pandas as pd
from typing import List

from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams as static_teams

from .config import paths


def fetch_rosters(season: str = "2025-26") -> pd.DataFrame:
    """Fetch all team rosters for a given season and save to processed folder.

    Parameters
    - season: NBA season string (e.g., '2025-26')

    Returns a DataFrame with concatenated rosters across all teams.
    """
    team_list = static_teams.get_teams()
    rows = []
    for t in team_list:
        tid = t.get('id'); tri = t.get('abbreviation'); name = t.get('full_name')
        if not tid:
            continue
        try:
            res = commonteamroster.CommonTeamRoster(team_id=tid, season=season)
            nd = res.get_normalized_dict()
            df = pd.DataFrame(nd.get('CommonTeamRoster', []))
            if df.empty:
                continue
            df['TEAM_ID'] = tid
            df['TEAM_ABBREVIATION'] = tri
            df['TEAM_NAME'] = name
            df['SEASON'] = season
            rows.append(df)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out_dir = paths.data_processed
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"rosters_{season.replace('/', '-')}.csv"
    out_parq = out_dir / f"rosters_{season.replace('/', '-')}.parquet"
    out.to_csv(out_csv, index=False)
    try:
        out.to_parquet(out_parq, index=False)
    except Exception:
        pass
    return out
