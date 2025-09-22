import pandas as pd
from pathlib import Path
from nba_api.stats.endpoints import boxscoresummaryv2

RAW_CSV = Path(r"C:\Users\mostg\OneDrive\Coding\NBA-Betting\data\raw\games_nba_api.csv")

def extract_linescore_df(bs):
    nd = bs.get_normalized_dict()
    if isinstance(nd, dict) and "LineScore" in nd and nd["LineScore"]:
        return pd.DataFrame(nd["LineScore"]).copy()
    frames = bs.get_data_frames()
    for f in frames:
        cols = set(map(str.upper, f.columns))
        if {"PTS_QTR1","PTS_QTR2","PTS_QTR3","PTS_QTR4"}.issubset(cols):
            return f.copy()
    return None


def main():
    assert RAW_CSV.exists(), f"Raw CSV not found: {RAW_CSV}"
    # Prefer most recent games (last 50 rows)
    df = pd.read_csv(RAW_CSV)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    sample = df.tail(200)
    gids = sample['game_id'].astype(str).unique().tolist()[:10]
    print(f"Testing {len(gids)} game_ids: {gids}")
    ok = 0
    for gid in gids:
        try:
            bs = boxscoresummaryv2.BoxScoreSummaryV2(game_id=gid)
            ls = extract_linescore_df(bs)
            if ls is not None and not ls.empty:
                ok += 1
                print(f"{gid}: LineScore OK, cols={list(ls.columns)[:8]} rows={len(ls)}")
            else:
                print(f"{gid}: LineScore NOT FOUND")
        except Exception as e:
            print(f"{gid}: ERROR {e}")
    print(f"Done. Success {ok}/{len(gids)}")

if __name__ == "__main__":
    main()
