from __future__ import annotations

import pandas as pd

import tools.tune_live_prop_shape_playoffs as tune_module


def test_load_recon_lookup_falls_back_to_props_actuals(tmp_path, monkeypatch):
    monkeypatch.setattr(tune_module, "PROCESSED", tmp_path)

    pd.DataFrame(
        [
            {
                "date": "2026-04-25",
                "game_id": "0042500103",
                "player_id": 1631094,
                "player_name": "Paolo Banchero",
                "team_abbr": "ORL",
                "pts": 25.0,
                "reb": 12.0,
                "ast": 9.0,
                "threes": 2.0,
                "stl": 3.0,
                "blk": 2.0,
                "tov": 3.0,
                "pra": 46.0,
            }
        ]
    ).to_csv(tmp_path / "props_actuals_2026-04-25.csv", index=False)

    lookup = tune_module._load_recon_lookup("2026-04-25")

    row = lookup[("ORL", "PAOLO BANCHERO")]
    assert row["pts"] == 25.0
    assert row["pra"] == 46.0
    assert row["team_tri"] == "ORL"
    assert row["name_key"] == "PAOLO BANCHERO"


def test_load_recon_lookup_prefers_recon_props_over_props_actuals(tmp_path, monkeypatch):
    monkeypatch.setattr(tune_module, "PROCESSED", tmp_path)

    pd.DataFrame(
        [
            {
                "date": "2026-04-25",
                "game_id": "0042500103",
                "player_id": 1631094,
                "player_name": "Paolo Banchero",
                "team_abbr": "ORL",
                "pts": 25.0,
                "reb": 12.0,
                "ast": 9.0,
                "threes": 2.0,
                "stl": 3.0,
                "blk": 2.0,
                "tov": 3.0,
                "pra": 46.0,
            }
        ]
    ).to_csv(tmp_path / "props_actuals_2026-04-25.csv", index=False)

    pd.DataFrame(
        [
            {
                "date": "2026-04-25",
                "game_id": "0042500103",
                "player_id": 1631094,
                "player_name": "Paolo Banchero",
                "team_abbr": "ORL",
                "pts": 27.0,
                "reb": 12.0,
                "ast": 9.0,
                "threes": 2.0,
                "stl": 3.0,
                "blk": 2.0,
                "tov": 3.0,
                "pra": 48.0,
            }
        ]
    ).to_csv(tmp_path / "recon_props_2026-04-25.csv", index=False)

    lookup = tune_module._load_recon_lookup("2026-04-25")

    row = lookup[("ORL", "PAOLO BANCHERO")]
    assert row["pts"] == 27.0
    assert row["pra"] == 48.0