from nba_betting.scrape_bref import seasons_to_fetch


def test_seasons_default_len():
    seasons = seasons_to_fetch()
    assert len(seasons) == 10
