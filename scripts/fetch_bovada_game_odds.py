import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd


def main():
    # Ensure local package imports work when running directly
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from nba_betting.odds_bovada import fetch_bovada_odds_current
    from nba_betting.config import paths

    if len(sys.argv) < 2:
        print("Usage: python scripts/fetch_bovada_game_odds.py YYYY-MM-DD")
        sys.exit(2)

    date_str = sys.argv[1]
    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        print("Invalid date. Use YYYY-MM-DD")
        sys.exit(2)

    try:
        df = fetch_bovada_odds_current(target_date)
        if df is None or df.empty:
            print(f"No Bovada odds found for {date_str}")
            sys.exit(0)
        out = paths.data_processed / f"game_odds_{date_str}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        # Normalize date column to date type if present
        if "date" in df.columns:
            try:
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
            except Exception:
                pass
        df.to_csv(out, index=False)
        print(f"Wrote {len(df)} rows to {out}")
    except Exception as e:
        print(f"Error fetching Bovada odds: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
