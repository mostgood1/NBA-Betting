from __future__ import annotations

import pandas as pd
from typing import Iterable, List

from nba_api.stats.endpoints import leaguegamelog

from .config import paths


def fetch_player_logs(seasons: Iterable[str]) -> pd.DataFrame:
    """Fetch league-wide player game logs for given seasons and save to processed.

    seasons: iterable of season strings like ['2023-24','2024-25','2025-26']
    Returns concatenated DataFrame.
    """
    frames: List[pd.DataFrame] = []
    for season in seasons:
        try:
            res = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star='Regular Season',
                player_or_team_abbreviation='P',
                counter=0,
            )
            nd = res.get_normalized_dict()
            df = pd.DataFrame(nd.get('LeagueGameLog', []))
            if df.empty:
                continue
            df['SEASON'] = season
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out_dir = paths.data_processed
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / 'player_logs.parquet', index=False)
    out.to_csv(out_dir / 'player_logs.csv', index=False)
    return out
