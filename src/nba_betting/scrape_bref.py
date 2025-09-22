from __future__ import annotations

import time
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

import pandas as pd
import requests
import requests_cache
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import paths

BASE = "https://www.basketball-reference.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
}

_SESSION: Optional[requests.Session] = None


class ScrapeError(Exception):
    pass


@dataclass
class GameRef:
    date: datetime
    visitor: str
    home: str
    box_url: Optional[str]


def seasons_to_fetch(last_n: int = 10) -> list[int]:
    year = datetime.now().year
    # NBA seasons end in the year name, e.g., 2024 season ends in 2024
    return list(range(year - last_n, year))


@retry(wait=wait_exponential(multiplier=1, min=1, max=15), stop=stop_after_attempt(5), reraise=True,
       retry=retry_if_exception_type((requests.RequestException, ScrapeError)))
def get(url: str, referer: Optional[str] = None) -> requests.Response:
    global _SESSION
    if _SESSION is None:
        # Use cached session if requests_cache is installed
        try:
            _SESSION = requests_cache.CachedSession()
        except Exception:
            _SESSION = requests.Session()
        _SESSION.headers.update(HEADERS)
    headers = {}
    if referer:
        headers["Referer"] = referer
    resp = _SESSION.get(url, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise ScrapeError(f"Status {resp.status_code} for {url}")
    return resp


def parse_schedule_page(html: str) -> list[GameRef]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id="schedule")
    out: list[GameRef] = []
    if not table:
        return out
    for tr in table.tbody.find_all("tr"):
        if tr.get("class") == ["thead"]:
            continue
        tds = tr.find_all("td")
        if not tds:
            continue
        # date is in th
        th = tr.find("th")
        date_str = (th.text or "").strip()
        if not date_str:
            continue
        try:
            date = datetime.strptime(date_str, "%a, %b %d, %Y")
        except ValueError:
            # sometimes empty rows
            continue
        visitor = tr.find("td", {"data-stat": "visitor_team_name"}).text.strip()
        home = tr.find("td", {"data-stat": "home_team_name"}).text.strip()
        box_td = tr.find("td", {"data-stat": "box_score_text"})
        box_url = None
        if box_td and box_td.find("a"):
            box_url = BASE + box_td.find("a")["href"]
        out.append(GameRef(date=date, visitor=visitor, home=home, box_url=box_url))
    return out


def fetch_season_schedule(season: int) -> list[GameRef]:
    url = f"{BASE}/leagues/NBA_{season}_games.html"
    games: list[GameRef] = []
    resp = get(url)
    time.sleep(0.5 + random.random() * 0.7)
    games.extend(parse_schedule_page(resp.text))
    # also monthly pages linked under content
    soup = BeautifulSoup(resp.text, "lxml")
    for a in soup.select("div#content a"):
        href = a.get("href", "")
        if href.startswith(f"/leagues/NBA_{season}_games-") and href.endswith(".html"):
            sub = get(BASE + href, referer=url)
            games.extend(parse_schedule_page(sub.text))
            time.sleep(0.8 + random.random() * 0.9)
    return games


def parse_line_scores_from_box(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    line_score = soup.find("table", id="line_score")
    if not line_score:
        return {}
    rows = line_score.tbody.find_all("tr")
    if len(rows) != 2:
        return {}
    # Map BRef keys to normalized names
    def norm_key(k: str) -> str:
        k = (k or "").lower()
        mapping = {
            "1": "q1", "2": "q2", "3": "q3", "4": "q4",
            "t": "pts", "pts": "pts",
        }
        if k.startswith("ot"):
            return k  # "ot1", "ot2", ...
        return mapping.get(k, k)

    def row_to_dict(tr):
        d = {}
        # Use both header and data-stat to be robust
        tds = tr.find_all("td")
        for td in tds:
            key = td.get("data-stat") or ""
            key = norm_key(key)
            d[key] = td.text.strip()
        return d
    vis = row_to_dict(rows[0])
    home = row_to_dict(rows[1])
    # keys include: visitor, home pts, and q1..q4, ot1.. etc
    return {"visitor": vis, "home": home}


def scrape_games(last_n: int = 10, rate_delay: float = 1.0, use_cache: bool = True, resume: bool = True) -> pd.DataFrame:
    seasons = seasons_to_fetch(last_n)
    records = []
    # HTTP caching
    if use_cache:
        requests_cache.install_cache(str(paths.root / ".http_cache"), backend="sqlite", expire_after=60 * 60 * 24 * 7)

    # Resume support: load existing rows and skip fetching those box scores
    seen = set()
    out_csv = paths.data_raw / "games_bref.csv"
    if resume and out_csv.exists():
        try:
            existing = pd.read_csv(out_csv)
            for _, r in existing.iterrows():
                seen.add((r.get("season"), str(r.get("date")), r.get("visitor_team"), r.get("home_team")))
        except Exception:
            pass
    for season in seasons:
        games = fetch_season_schedule(season)
        for g in games:
            if not g.box_url:
                continue
            key = (season, str(g.date.date()), g.visitor, g.home)
            if key in seen:
                continue
            try:
                resp = get(g.box_url, referer=f"{BASE}/leagues/NBA_{season}_games.html")
            except Exception:
                continue
            time.sleep(max(0.5, rate_delay) + random.random() * 0.6)
            ls = parse_line_scores_from_box(resp.text)
            if not ls:
                continue
            rec = {
                "season": season,
                "date": g.date.date(),
                "visitor_team": g.visitor,
                "home_team": g.home,
            }
            # merge line scores
            for side, prefix in ((ls.get("visitor"), "visitor"), (ls.get("home"), "home")):
                if not side:
                    continue
                # totals, q1..q4, maybe ot1..otN
                pts = side.get("pts")
                if pts is not None:
                    rec[f"{prefix}_pts"] = int(pts)
                for q in ["q1", "q2", "q3", "q4"]:
                    v = side.get(q)
                    if v:
                        rec[f"{prefix}_{q}"] = int(v)
                # collect OT columns defensively
                for k, v in side.items():
                    if k.startswith("ot") and v.isdigit():
                        rec[f"{prefix}_{k}"] = int(v)
            records.append(rec)
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    # Deduplicate games (some pages overlap); keep first occurrence
    df = df.drop_duplicates(subset=["season", "date", "visitor_team", "home_team"]).reset_index(drop=True)
    # derive halves
    for side in ("visitor", "home"):
        for half, cols in (("h1", ["q1", "q2"]), ("h2", ["q3", "q4"])):
            qcols = [f"{side}_{c}" for c in cols]
            df[f"{side}_{half}"] = df[qcols].sum(axis=1, min_count=1)
    # outcomes
    df["total_points"] = df[["visitor_pts", "home_pts"]].sum(axis=1, min_count=2)
    df["home_win"] = (df["home_pts"] > df["visitor_pts"]).astype("Int64")
    df["margin"] = df["home_pts"] - df["visitor_pts"]
    # save
    paths.data_raw.mkdir(parents=True, exist_ok=True)
    out_parq = paths.data_raw / "games_bref.parquet"
    df.to_csv(out_csv, index=False)
    try:
        df.to_parquet(out_parq, index=False)
    except Exception:
        pass
    return df


if __name__ == "__main__":
    df = scrape_games(last_n=10)
    print(f"Scraped {len(df)} games")
