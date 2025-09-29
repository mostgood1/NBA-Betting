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

### Daily game odds with automatic fallback

The `predict-date` command now automatically attaches odds for the slate date:

- It first tries current OddsAPI lines if `ODDS_API_KEY` is set (via env or `.env`).
- If unavailable or empty, it falls back to scraping Bovada for that date.
- A standardized CSV is written to `data/processed/game_odds_YYYY-MM-DD.csv` and merged into `predictions_YYYY-MM-DD.csv` with implied probabilities and model edges (winner, spread, total).

Example (PowerShell):

```powershell
# Optional for OddsAPI (will fall back to Bovada if not set)
$env:ODDS_API_KEY = "<your_key>"

# Generate predictions and odds for a given date
python -m nba_betting.cli predict-date --date 2025-04-13

# Outputs:
# - predictions_2025-04-13.csv (repo root)
# - data/processed/game_odds_2025-04-13.csv
```

The frontend `web/app.js` will look for `data/processed/game_odds_YYYY-MM-DD.csv` (among other candidates) and render ML/spread/total plus EV lines on each game card.

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

## Frontend & Render Deployment (mirrors NHL-Betting)

This repo deploys like the NHL site: a minimal Flask app serves the cards at `/` using the static assets in `web/`. Gunicorn runs the Flask app on Render.

What’s included for deploy:
- `app.py` – Flask server that serves `web/` assets and a `/health` endpoint.
- `Procfile` – Gunicorn process type.
- `start.sh` – Start script used on Render (threads, timeouts akin to NFL-Betting).
- `render.yaml` – Blueprint for a Python Web Service (env: python).

Run locally (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py  # http://localhost:5050
```

Deploy on Render (Blueprint):
1) Push to GitHub (done if you’re viewing this there).
2) In Render: New → Blueprint → paste repo URL. It will detect `render.yaml` and create a Python Web Service.
3) Build Command: `pip install -r requirements.txt` (from render.yaml)
4) Start Command: `bash start.sh` (from render.yaml)

Environment variables (optional but consistent with NFL app):
- `FLASK_ENV=production`
- `PYTHONUNBUFFERED=1`

Routing:
- `/` serves the cards page directly
- Static assets are under `/web/*` (e.g., `/web/assets/logos/BOS.svg`)
- Legacy `/web/` and `/web/index.html` redirect to `/` for a single canonical entrypoint

Notes:
- The UI is static; the app serves JSON/CSV artifacts directly from the repo (`/data/...`) for the client to render.
- Prefer hitting cron endpoints to regenerate predictions/odds artifacts; the page will reflect updates without redeploy.

## Admin endpoints (optional)

For parity with NFL-Betting, the Flask server exposes lightweight admin endpoints:

- Start a daily update job:
	- GET/POST `/api/admin/daily-update?push=1` (push optional)
- Check job status and logs:
	- GET `/api/admin/daily-update/status?tail=200`

Auth:
- If `ADMIN_KEY` is set in environment, include it via query `?key=...` or header `X-Admin-Key: ...`.
- If no key is set, only local/lan clients are allowed (127.0.0.1 or private ranges like 192.168.x.x) for convenience.

What the job does:
- It’s a placeholder that can run CLI tasks (e.g., predictions or exports) and append logs under `logs/`.
- Git push is attempted if `push=1` and Git is configured with write access.

Examples (PowerShell):

```powershell
# Start job
$resp = Invoke-RestMethod http://localhost:5050/api/admin/daily-update
$resp

# Poll status
Invoke-RestMethod http://localhost:5050/api/admin/daily-update/status?tail=100 | Format-List
```

## Cron endpoints (NHL parity)

For ops parity with the NHL app, token-gated cron endpoints are provided. Set an environment variable `CRON_TOKEN` in Render (or locally) and use one of:

- Header: `Authorization: Bearer <CRON_TOKEN>`
- Header: `X-Cron-Token: <CRON_TOKEN>`
- Query: `?token=<CRON_TOKEN>`

Endpoints:
- `POST /api/cron/predict-date?date=YYYY-MM-DD` – runs the CLI `predict-date` and writes `predictions_YYYY-MM-DD.csv` and `data/processed/game_odds_YYYY-MM-DD.csv`.
- `POST /api/cron/refresh-bovada?date=YYYY-MM-DD` – fetches Bovada odds and writes `data/processed/game_odds_YYYY-MM-DD.csv`.
- `POST /api/cron/capture-closing?date=YYYY-MM-DD` – exports `data/processed/closing_lines_YYYY-MM-DD.csv` from consensus data (build via `make-closing-lines`).
- `POST /api/cron/daily-update` – triggers the same in-process daily update as `/api/admin/daily-update` (push disabled by default for cron).
- `GET  /api/cron/config` – safe introspection of available cron/admin booleans.

Render examples (PowerShell syntax equivalent using Invoke-RestMethod):

```powershell
$base = "https://your-render-url.onrender.com"
$token = "$(RenderEnv:CRON_TOKEN)"  # or paste token for local testing
$date = (Get-Date).ToString('yyyy-MM-dd')

# Predict slate and auto-attach odds (OddsAPI if configured, else Bovada fallback)
Invoke-RestMethod -Method Post -Uri "$base/api/cron/predict-date?date=$date" -Headers @{ Authorization = "Bearer $token" }

# Just refresh Bovada odds
Invoke-RestMethod -Method Post -Uri "$base/api/cron/refresh-bovada?date=$date" -Headers @{ Authorization = "Bearer $token" }

# Capture consensus closing lines (requires closing_lines.parquet or raw odds)
Invoke-RestMethod -Method Post -Uri "$base/api/cron/capture-closing?date=$date" -Headers @{ Authorization = "Bearer $token" }

# Kick off daily update sequence (lightweight)
Invoke-RestMethod -Method Post -Uri "$base/api/cron/daily-update" -Headers @{ Authorization = "Bearer $token" }
```

Scheduling on Render:
- Use Render Cron Jobs to hit these endpoints on your cadence. Include the Bearer header.
- If you attach a Render Disk, ensure `data/` points to that disk path (default here is repo-relative). Update paths in `src/nba_betting/config.py` if needed.
