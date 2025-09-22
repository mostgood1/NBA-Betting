# NBA Betting Predictor

End-to-end pipeline to fetch NBA historical games (last 10 seasons) via NBA Stats API, engineer features (Elo + rest), train predictive models for:
- Full game: winner, spread (ATS), total points
- Derivatives: quarters and halves (winner, spread, totals)

Includes a CLI to run: fetch, build-features, train, evaluate, and predict. Also supports historical odds (OddsAPI), consensus closing lines, period backtests, and player prop actuals via nbastatR.

## Quick start

1) Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Fetch data (NBA Stats API, optionally with period scoring):

```powershell
python -m nba_betting.cli fetch --years 10 --with-periods
```

3) Build features and train models:

```powershell
python -m nba_betting.cli build-features
python -m nba_betting.cli train
```

4) Evaluate and backtest:

```powershell
# Evaluate on a single holdout season (defaults to latest season if not provided)
python -m nba_betting.cli evaluate --holdout-season 2024

# Backtest across seasons (choose a range or last N seasons)
python -m nba_betting.cli backtest --start 2018 --end 2024
python -m nba_betting.cli backtest --last-n 5
```

Per-season metrics are saved to `data/processed/backtest_metrics.csv`.

5) Predict upcoming games (uses trained models):

```powershell
python -m nba_betting.cli predict --input .\samples\matchups.csv
```

Outputs:
- Raw data in `data/raw/`
- Features in `data/processed/`
- Backtest results in `data/processed/`
- Models in `models/`

Notes:
- NBA API usage may be rate-limited; the fetch includes small delays between calls.
- You can rerun build-features and train without re-fetching.

## Odds & Market Data

### Configure OddsAPI

Copy `.env.example` to `.env` and set your key (get one from https://the-odds-api.com/):

```bash
ODDS_API_KEY=your_key_here
```

### Backfill historical odds and build closing lines

```powershell
python -m nba_betting.cli backfill-odds --api-key $env:ODDS_API_KEY --start 2016-10-01T00:00:00Z --end 2025-06-30T23:59:59Z --step-days 5 --markets h2h,spreads,totals
python -m nba_betting.cli make-closing-lines
python -m nba_betting.cli attach-closing-lines
```

This produces `data/processed/closing_lines.parquet` and merges it into your features as `features_with_market.parquet`.

### Player props odds (experimental)

```powershell
# Snapshot for a specific date; tries historical, then falls back to current
python -m nba_betting.cli backfill-player-props --date 2025-10-24 --mode auto
```

Notes: Some snapshots may return 422 if props aren’t available at that timestamp; try `--mode current` on game day.

## Player Prop Actuals (nbastatR)

We use nbastatR (R) to fetch player game logs and compute actuals for props: PTS, REB, AST, 3PM, PRA.

### Install R and nbastatR

1. Install R for Windows (https://cran.r-project.org/) so `Rscript.exe` is available.
2. In an R session:

```r
install.packages("remotes")
remotes::install_github("abresler/nbastatR")
```

If `Rscript.exe` isn’t on PATH, set an environment variable for this session:

```powershell
$env:RSCRIPT_PATH = "C:\\Program Files\\R\\R-4.4.1\\bin\\Rscript.exe"
```

### Fetch actuals by date or range

```powershell
python -m nba_betting.cli fetch-prop-actuals --date 2025-01-15
python -m nba_betting.cli fetch-prop-actuals --start 2024-10-01 --end 2025-06-30
```

Output is upserted to `data/processed/props_actuals.csv` and `.parquet` with columns:
- date, game_id, team, player_id, player_name, pts, reb, ast, threes, pra

### Build, train, predict, evaluate props

```powershell
# Build features and train Ridge models for props
python -m nba_betting.cli build-props-features
python -m nba_betting.cli train-props

# Predict for a given slate date (filters to scoreboard teams by default)
python -m nba_betting.cli predict-props --date 2025-01-15

# Evaluate model vs actuals over a range (falls back to player_logs if nbastatR not available)
python -m nba_betting.cli evaluate-props --start 2025-01-15 --end 2025-01-15
```

### Compute props edges (EV)

Merge your predictions with OddsAPI player props to compute model probabilities, edge, and EV:

```powershell
# If you have saved props odds for that date in data/raw, use them; otherwise set ODDS_API_KEY and it will fetch
$env:ODDS_API_KEY = "<your_key>"
python -m nba_betting.cli props-edges --date 2025-01-15 --use-saved --mode auto
```

Outputs `data/processed/props_edges_YYYY-MM-DD.csv` with columns:
- date, player_name, stat, side, line, price, implied_prob, model_prob, edge, ev, bookmaker

### No historical props? Current-day only workflow

If you can’t access historical player props lines, you can still use the pipeline daily:

1) Ensure player logs are refreshed periodically and models are trained.
2) On game days, about 1–3 hours before tip, run:

```powershell
$env:ODDS_API_KEY = "<your_key>"
python -m nba_betting.cli predict-props --date 2025-10-24
python -m nba_betting.cli props-edges --date 2025-10-24 --no-use-saved --mode current --min-edge 0.03 --min-ev 0 --top 100
```

3) Optionally filter to a specific set of books:

```powershell
python -m nba_betting.cli props-edges --date 2025-10-24 --mode current --bookmakers draftkings,fanduel,pinnacle
```

This produces an edges CSV you can sort by EV or edge for actionable picks without any historical props archive.

## Troubleshooting

- Rscript not found: install R, add `Rscript.exe` to PATH, or set `RSCRIPT_PATH` to the full path.
- OddsAPI 422 for player props: the API may not have props at your snapshot timestamp; re-run with `--mode current` near tip-off or try a different date.
