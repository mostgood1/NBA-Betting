# NBA Frontend (Local Preview)

This is a lightweight static UI that renders the 2025–26 NBA schedule by date and shows model recommendations if a predictions CSV is present.

## Data sources
- Schedule JSON: `../data/processed/schedule_2025_26.json` (generated via `python -m nba_betting.cli fetch-schedule`)
- Predictions CSV (optional): `../predictions_YYYY-MM-DD.csv` (generated via `python -m nba_betting.cli predict-date --date YYYY-MM-DD [--merge-odds odds.csv]`)

## Run locally
Use any static server from the repo root so relative paths resolve:

- Python (already used in this repo):
  - Start from repo root: `python -m http.server 8080`
  - Open http://localhost:8080/web/index.html

The date picker defaults to today if available in the schedule; otherwise the first schedule date. If a `predictions_YYYY-MM-DD.csv` exists for the selected date, recommendation badges are shown.

## Team assets (logos)
Provide actual NBA team logos locally (not included in repo). Put files here:

```
web/assets/logos/
  BOS.svg
  LAL.svg
  ... (TRICODE.svg)
```

- SVG preferred; optionally provide a PNG with the same name for fallback.
- If neither SVG nor PNG exists for a team, the UI automatically falls back to a colored badge with the team’s tricode.

Note: Ensure you have rights to use and distribute these assets. If you’re unsure, keep them local and out of version control.

## Predictions CSV format (minimal)
Columns expected (case-sensitive) for basic badges:
- `date` (YYYY-MM-DD)
- `home_team` (Full name, e.g., "Los Angeles Lakers")
- `visitor_team` (Full name)
- `home_win_prob` (0–1)
- `pred_margin` (home margin, points)
- `pred_total` (total points)

Optional edge columns (if merging odds):
- `edge_spread` (positive favors HOME ATS, negative favors AWAY ATS)
- `edge_total` (positive favors OVER, negative favors UNDER)

## Accessibility
- Inputs have accessible labels; team logos include alt text; team names are rendered visibly next to logos.

## Customization
- Styles: `web/styles.css`
- Team colors/names: `web/assets/teams_nba.json`
- Card rendering logic: `web/app.js`
