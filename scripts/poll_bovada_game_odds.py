import sys
import time
from pathlib import Path
from datetime import datetime


def main():
    # Usage
    if len(sys.argv) < 2:
        print("Usage: python scripts/poll_bovada_game_odds.py YYYY-MM-DD [interval_seconds]")
        sys.exit(2)
    date_str = sys.argv[1]
    try:
        _ = datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        print("Invalid date. Use YYYY-MM-DD")
        sys.exit(2)
    try:
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 600  # 10 minutes default
        if interval < 60:
            interval = 60
    except Exception:
        interval = 600

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from nba_betting.odds_bovada import fetch_bovada_odds_current
    from nba_betting.config import paths

    out_csv = paths.data_processed / f"game_odds_{date_str}.csv"

    print(f"Polling Bovada preseason odds for {date_str} every {interval}s ...")
    while True:
        try:
            import pandas as pd
            df = fetch_bovada_odds_current(date_str)
            if df is not None and not df.empty:
                out_csv.parent.mkdir(parents=True, exist_ok=True)
                # normalize date column to date type if present
                if "date" in df.columns:
                    try:
                        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
                    except Exception:
                        pass
                df.to_csv(out_csv, index=False)
                print(f"Found odds. Wrote {len(df)} rows to {out_csv}")
                # Merge into predictions using CLI (keeps edges logic consistent)
                py = str((repo_root / ".venv" / "Scripts" / "python.exe"))
                if not Path(py).exists():
                    py = "python"
                import subprocess
                cmd = [py, "-m", "nba_betting.cli", "predict-date", "--date", date_str, "--merge-odds", str(out_csv), "--out", str(paths.data_processed / f"predictions_{date_str}.csv")]
                print("Merging odds into predictions:", " ".join(cmd))
                try:
                    subprocess.run(cmd, cwd=str(repo_root), check=False)
                except Exception:
                    pass
                return
            else:
                print(f"No odds yet for {date_str}. Next try in {interval}s ...")
        except Exception as e:
            print(f"Poll error: {e}. Next try in {interval}s ...")
        time.sleep(interval)


if __name__ == "__main__":
    main()
