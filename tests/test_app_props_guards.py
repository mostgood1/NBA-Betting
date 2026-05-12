from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pandas as pd
import requests

import app as app_module


def test_player_prop_bookmakers_default_to_all_us_books_for_pregame(monkeypatch):
    monkeypatch.delenv("PLAYER_PROP_BOOKMAKERS", raising=False)

    assert app_module._player_prop_bookmakers_tuple(env_name="PLAYER_PROP_BOOKMAKERS") == ()
    assert app_module._player_prop_bookmakers_csv(env_name="PLAYER_PROP_BOOKMAKERS") is None


def test_player_prop_bookmakers_default_to_four_books_for_live(monkeypatch):
    monkeypatch.delenv("PLAYER_PROP_BOOKMAKERS", raising=False)
    monkeypatch.delenv("LIVE_PLAYER_PROP_BOOKMAKERS", raising=False)

    assert app_module._player_prop_bookmakers_tuple(env_name="LIVE_PLAYER_PROP_BOOKMAKERS") == (
        "fanduel",
        "draftkings",
        "betmgm",
        "bet365",
    )
    assert app_module._player_prop_bookmakers_csv(env_name="LIVE_PLAYER_PROP_BOOKMAKERS") == "fanduel,draftkings,betmgm,bet365"


def test_prune_invalid_props_recommendations_artifact_removes_model_only_cards(tmp_path):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)

    rec_path = processed / "props_recommendations_2026-03-13.csv"
    pd.DataFrame(
        [
            {
                "player": "Cade Cunningham",
                "team": "DET",
                "plays": "[]",
                "ladders": "[]",
                "model": "{'pts': 25.8}",
            }
        ]
    ).to_csv(rec_path, index=False)

    messages: list[str] = []
    removed = app_module._prune_invalid_props_recommendations_artifact(
        processed / "props_edges_2026-03-13.csv",
        rec_path,
        log_cb=messages.append,
    )

    assert removed is True
    assert not rec_path.exists()
    assert any("Removed same-day props recommendations" in msg for msg in messages)


def test_prune_invalid_props_recommendations_artifact_preserves_line_bearing_cards(tmp_path):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)

    rec_path = processed / "props_recommendations_2026-03-13.csv"
    pd.DataFrame(
        [
            {
                "player": "Cade Cunningham",
                "team": "DET",
                "plays": "[{'market': 'pts', 'side': 'OVER', 'line': 25.5, 'price': -110}]",
                "ladders": "[]",
                "model": "{'pts': 25.8}",
            }
        ]
    ).to_csv(rec_path, index=False)

    messages: list[str] = []
    removed = app_module._prune_invalid_props_recommendations_artifact(
        processed / "props_edges_2026-03-13.csv",
        rec_path,
        log_cb=messages.append,
    )

    assert removed is False
    assert rec_path.exists()
    assert any("Preserving existing same-day line-bearing props recommendations" in msg for msg in messages)


def test_api_cards_skips_missing_prop_players_warning_when_smartsim_errors(tmp_path, monkeypatch):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    smart_sim_path = processed / "smart_sim_2026-03-13_MEM_DET.json"
    smart_sim_path.write_text(
        json.dumps(
            {
                "error": "missing_players",
                "home": "MEM",
                "away": "DET",
                "home_players": 0,
                "away_players": 0,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(app_module, "DATA_PROCESSED_DIR", processed)
    monkeypatch.setattr(app_module, "_load_game_odds_map", lambda _date: {("MEM", "DET"): {"home_team": "Memphis Grizzlies", "visitor_team": "Detroit Pistons"}})
    monkeypatch.setattr(app_module, "_load_props_predictions_map", lambda _date: {})
    monkeypatch.setattr(app_module, "_compute_player_minutes_priors", lambda _date, days_back=21: {})
    monkeypatch.setattr(app_module, "_live_find_processed_csv", lambda _stem, _date: None)
    monkeypatch.setattr(app_module, "_live_load_props_edges_index", lambda _date: {})
    monkeypatch.setattr(
        app_module,
        "_load_props_recommendations_by_team",
        lambda _date: {
            "MEM": [
                {
                    "player": "Cam Spencer",
                    "team": "MEM",
                    "plays": [{"market": "pts", "side": "over", "line": 10.5, "price": -110, "book": "fanduel"}],
                    "top_play": None,
                    "top_play_reasons": [],
                }
            ],
            "DET": [
                {
                    "player": "Cade Cunningham",
                    "team": "DET",
                    "plays": [{"market": "ast", "side": "over", "line": 8.5, "price": -110, "book": "fanduel"}],
                    "top_play": None,
                    "top_play_reasons": [],
                }
            ],
        },
    )
    monkeypatch.setattr(app_module, "_load_injury_context_map", lambda _date: {})
    monkeypatch.setattr(app_module, "_roster_players_for_date", lambda _date: {})
    monkeypatch.setattr(app_module, "_matchup_writeup", lambda _game: "")

    with app_module.app.test_request_context("/api/cards?date=2026-03-13&include_players=1"):
        response = app_module.api_cards()

    payload = response.get_json()
    warnings = payload["games"][0].get("warnings") or []

    assert f"SmartSim error for MEM-DET: missing_players" in warnings
    assert all("Players with prop lines missing from SmartSim boxscore" not in warning for warning in warnings)


def test_api_cards_falls_back_to_predictions_when_smart_sim_missing(tmp_path, monkeypatch):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "date": "2026-03-28",
                "home_team": "PHOENIX SUNS",
                "visitor_team": "UTAH JAZZ",
                "home_win_prob": 0.6191085240899175,
                "spread_margin": 5.150077403744712,
                "totals": 229.85171222031929,
                "quarters_q1_win": 0.5727539225959937,
                "quarters_q1_margin": 1.8587640828792336,
                "quarters_q1_total": 56.03260037762033,
                "quarters_q2_win": 0.5810418441135232,
                "quarters_q2_margin": 2.298050030953906,
                "quarters_q2_total": 59.169649448678136,
                "quarters_q3_win": 0.4742736864944752,
                "quarters_q3_margin": 0.1874770016348637,
                "quarters_q3_total": 56.0023145337821,
                "quarters_q4_win": 0.4556026591423514,
                "quarters_q4_margin": 0.3610672101987547,
                "quarters_q4_total": 57.03009650947159,
                "home_win_prob_cal": 0.6090342803663917,
                "commence_time": "2026-03-28T22:00:00Z",
            }
        ]
    ).to_csv(processed / "predictions_2026-03-28.csv", index=False)

    monkeypatch.setattr(app_module, "DATA_PROCESSED_DIR", processed)
    monkeypatch.setattr(app_module, "_load_smart_sim_files_for_authoritative_slate", lambda _date: [])
    monkeypatch.setattr(app_module, "_find_next_available_smart_sim_date", lambda *_args, **_kwargs: (None, []))
    monkeypatch.setattr(
        app_module,
        "_load_game_odds_map",
        lambda _date: {
            ("PHX", "UTA"): {
                "home_team": "Phoenix Suns",
                "visitor_team": "Utah Jazz",
                "home_spread": -5.5,
                "total": 230.5,
                "commence_time": "2026-03-28T22:00:00Z",
            }
        },
    )
    monkeypatch.setattr(app_module, "_load_props_predictions_map", lambda _date: {})
    monkeypatch.setattr(app_module, "_compute_player_minutes_priors", lambda _date, days_back=21: {})
    monkeypatch.setattr(app_module, "_live_load_props_edges_index", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_props_recommendations_by_team", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_injury_context_map", lambda _date: {})
    monkeypatch.setattr(app_module, "_roster_players_for_date", lambda _date: {})
    monkeypatch.setattr(
        app_module,
        "_load_best_bets_game_context",
        lambda _date: {
            "by_pair": {
                ("PHX", "UTA"): {
                    "home_team": "Phoenix Suns",
                    "away_team": "Utah Jazz",
                    "home_tri": "PHX",
                    "away_tri": "UTA",
                    "pred_total_raw": 229.85171222031929,
                    "pred_total_adjusted": 232.4,
                    "pred_total": 232.4,
                    "pred_margin_raw": 5.150077403744712,
                    "pred_margin_adjusted": 6.2,
                    "pred_margin": 6.2,
                    "home_win_prob_raw": 0.6090342803663917,
                    "home_win_prob_adjusted": 0.651,
                    "home_win_prob": 0.651,
                    "home_pred_points": 119.3,
                    "away_pred_points": 113.1,
                    "market_total": 230.5,
                    "commence_time": "2026-03-28T22:00:00Z",
                }
            },
            "slate_total_median": 230.5,
        },
    )
    monkeypatch.setattr(app_module, "_load_best_bets_props_prediction_lookup", lambda _date: {})
    monkeypatch.setattr(app_module, "_best_bets_load_injury_snapshot", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_cards_game_recommendations_index", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_cards_prop_snapshot_index", lambda _date: {})
    monkeypatch.setattr(app_module, "_sim_vs_line_prop_recommendations", lambda *args, **kwargs: {"home": [], "away": []})
    monkeypatch.setattr(app_module, "_build_cards_game_market_recommendations", lambda **kwargs: [])
    monkeypatch.setattr(app_module, "_matchup_writeup", lambda _game: "")

    with app_module.app.test_request_context("/api/cards?date=2026-03-28"):
        response = app_module.api_cards()

    payload = response.get_json()

    assert payload["date"] == "2026-03-28"
    assert len(payload["games"]) == 1

    game = payload["games"][0]
    assert game["home_tri"] == "PHX"
    assert game["away_tri"] == "UTA"
    assert game["sim"]["mode"] == "prediction_fallback"
    assert game["sim"]["score"]["p_home_win"] == 0.6090342803663917
    assert game["sim"]["score"]["total_mean"] == 229.85171222031929
    assert game["sim"]["context"]["pregame_prior"]["pred_total_adjusted"] == 232.4
    assert game["sim"]["context"]["pregame_prior"]["pred_margin_adjusted"] == 6.2
    assert game["sim"]["context"]["pregame_prior"]["home_win_prob_adjusted"] == 0.651
    assert "Using predictions fallback because SmartSim artifact is missing for this matchup." in (game.get("warnings") or [])


def test_load_smart_sim_files_for_date_prefers_repo_copy_over_active_data_root(tmp_path, monkeypatch):
    repo_processed = tmp_path / "repo" / "data" / "processed"
    active_processed = tmp_path / "active" / "data" / "processed"
    repo_processed.mkdir(parents=True)
    active_processed.mkdir(parents=True)

    repo_file = repo_processed / "smart_sim_2026-04-01_HOU_MIL.json"
    active_file = active_processed / "smart_sim_2026-04-01_HOU_MIL.json"
    repo_file.write_text(json.dumps({"home": "HOU", "away": "MIL", "source": "repo"}), encoding="utf-8")
    active_file.write_text(json.dumps({"home": "HOU", "away": "MIL", "source": "active"}), encoding="utf-8")

    monkeypatch.setattr(app_module, "REPO_DATA_PROCESSED_DIR", repo_processed)
    monkeypatch.setattr(app_module, "DATA_PROCESSED_DIR", active_processed)

    files = app_module._load_smart_sim_files_for_date("2026-04-01")

    assert files == [repo_file]


def test_load_cards_sim_detail_snapshot_prefers_repo_copy_over_active_data_root(tmp_path, monkeypatch):
    repo_processed = tmp_path / "repo" / "data" / "processed"
    active_processed = tmp_path / "active" / "data" / "processed"
    repo_processed.mkdir(parents=True)
    active_processed.mkdir(parents=True)

    fname = "cards_sim_detail_2026-04-01.json"
    (repo_processed / fname).write_text(
        json.dumps({"games": [{"home_tri": "HOU", "away_tri": "MIL", "sim": {"players_summary": {"home": 15}}}]}),
        encoding="utf-8",
    )
    (active_processed / fname).write_text(
        json.dumps({"games": [{"home_tri": "HOU", "away_tri": "MIL", "sim": {"players_summary": {"home": 1}}}]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(app_module, "REPO_DATA_PROCESSED_DIR", repo_processed)
    monkeypatch.setattr(app_module, "DATA_PROCESSED_DIR", active_processed)
    monkeypatch.setattr(app_module, "_maybe_fetch_remote_processed", lambda _name: None)

    payload = app_module._load_cards_sim_detail_snapshot("2026-04-01")

    assert payload == {"games": [{"home_tri": "HOU", "away_tri": "MIL", "sim": {"players_summary": {"home": 15}}}]}


def test_live_lens_override_prefers_repo_copy_over_active_data_root(tmp_path, monkeypatch):
    repo_processed = tmp_path / "repo" / "data" / "processed"
    active_live_lens = tmp_path / "active" / "data" / "live_lens"
    repo_processed.mkdir(parents=True)
    active_live_lens.mkdir(parents=True)

    (repo_processed / "live_lens_tuning_override.json").write_text(
        json.dumps({"markets": {"player_prop": {"watch": 3.0}}}),
        encoding="utf-8",
    )
    (active_live_lens / "live_lens_tuning_override.json").write_text(
        json.dumps({"markets": {"player_prop": {"watch": 9.0}}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(app_module, "REPO_DATA_PROCESSED_DIR", repo_processed)
    monkeypatch.setattr(app_module, "_live_lens_artifacts_dir", lambda: active_live_lens)
    monkeypatch.setattr(app_module, "_live_lens_override_cache", (0.0, "", 0.0, None))

    payload = app_module._live_load_lens_override()

    assert payload == {"markets": {"player_prop": {"watch": 3.0}}}


def test_api_live_lens_tuning_prefers_repo_override_and_latest_repo_optimized_adjustments(tmp_path, monkeypatch):
    repo_processed = tmp_path / "repo" / "data" / "processed"
    active_live_lens = tmp_path / "active" / "data" / "live_lens"
    repo_processed.mkdir(parents=True)
    active_live_lens.mkdir(parents=True)

    (repo_processed / "live_lens_tuning_override.json").write_text(
        json.dumps(
            {
                "markets": {"player_prop": {"watch": 3.0}},
                "trained": {"source": "repo-override"},
            }
        ),
        encoding="utf-8",
    )
    (active_live_lens / "live_lens_tuning_override.json").write_text(
        json.dumps({"markets": {"player_prop": {"watch": 9.0}}}),
        encoding="utf-8",
    )
    (repo_processed / "live_lens_adjustments_optimized_2026-04-01_2026-04-07.json").write_text(
        json.dumps(
            {
                "window": {"start": "2026-04-01", "end": "2026-04-07"},
                "generated_at": "2026-04-08T05:00:00Z",
                "best": {
                    "params": {"pace_weight": 0.4, "min_elapsed_min": 9.0},
                    "bets": 31,
                    "profit": 12.5,
                    "roi_per_bet": 0.4,
                },
            }
        ),
        encoding="utf-8",
    )
    (active_live_lens / "live_lens_adjustments_optimized_2026-04-01_2026-04-07.json").write_text(
        json.dumps(
            {
                "window": {"start": "2026-04-01", "end": "2026-04-07"},
                "best": {"params": {"pace_weight": 0.1, "min_elapsed_min": 3.0}},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(app_module, "REPO_DATA_PROCESSED_DIR", repo_processed)
    monkeypatch.setattr(app_module, "_live_lens_artifacts_dir", lambda: active_live_lens)
    monkeypatch.setattr(app_module, "_live_tuning_cache", {})

    with app_module.app.test_request_context("/api/live_lens_tuning?ttl=300"):
        response = app_module.api_live_lens_tuning()

    payload = response.get_json()

    assert response.status_code == 200
    assert payload["markets"]["player_prop"]["watch"] == 3.0
    assert payload["adjustments"]["game_total"]["pace_weight"] == 0.4
    assert payload["adjustments"]["game_total"]["min_elapsed_min"] == 9.0
    assert payload["trained"]["source"] == "repo-override"
    assert payload["trained"]["game_total_optimization"]["path"] == "live_lens_adjustments_optimized_2026-04-01_2026-04-07.json"


def test_api_download_live_lens_adjustments_optimized_prefers_latest_repo_copy(tmp_path, monkeypatch):
    repo_processed = tmp_path / "repo" / "data" / "processed"
    active_live_lens = tmp_path / "active" / "data" / "live_lens"
    repo_processed.mkdir(parents=True)
    active_live_lens.mkdir(parents=True)

    latest_name = "live_lens_adjustments_optimized_2026-04-01_2026-04-07.json"
    older_name = "live_lens_adjustments_optimized_2026-03-25_2026-03-31.json"
    (repo_processed / older_name).write_text(json.dumps({"best": {"params": {"pace_weight": 0.2}}}), encoding="utf-8")
    (repo_processed / latest_name).write_text(json.dumps({"best": {"params": {"pace_weight": 0.4}}}), encoding="utf-8")
    (active_live_lens / latest_name).write_text(json.dumps({"best": {"params": {"pace_weight": 0.1}}}), encoding="utf-8")

    monkeypatch.setattr(app_module, "REPO_DATA_PROCESSED_DIR", repo_processed)
    monkeypatch.setattr(app_module, "_live_lens_artifacts_dir", lambda: active_live_lens)

    with app_module.app.test_client() as client:
        response = client.get("/api/download_live_lens_adjustments_optimized")

    assert response.status_code == 200
    assert json.loads(response.data) == {"best": {"params": {"pace_weight": 0.4}}}


def test_api_prop_ladders_requests_players_from_api_cards(monkeypatch):
    observed: dict[str, str | None] = {"include_players": None}

    def _fake_api_cards():
        observed["include_players"] = app_module.request.args.get("include_players")
        return app_module.jsonify({"games": []})

    monkeypatch.setattr(app_module, "api_cards", _fake_api_cards)

    with app_module.app.test_request_context("/api/prop-ladders?date=2026-04-01"):
        response = app_module.api_prop_ladders()

    payload = response.get_json()

    assert response.status_code == 200
    assert observed["include_players"] == "1"
    assert payload["date"] == "2026-04-01"
    assert payload["rows"] == []


def test_api_cards_prefers_cards_props_snapshot_over_runtime_recompute(tmp_path, monkeypatch):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)

    (processed / "cards_props_snapshot_2026-03-28.json").write_text(
        json.dumps(
            {
                "date": "2026-03-28",
                "games": [
                    {
                        "home_tri": "PHX",
                        "away_tri": "UTA",
                        "prop_recommendations": {
                            "home": [
                                {
                                    "player": "Devin Booker",
                                    "card_bucket": "official",
                                    "card_rank": 1,
                                    "best": {
                                        "market": "pts",
                                        "side": "OVER",
                                        "line": 27.5,
                                        "price": -110,
                                    },
                                    "picks": [],
                                }
                            ],
                            "away": [],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(app_module, "DATA_PROCESSED_DIR", processed)
    monkeypatch.setattr(app_module, "_load_smart_sim_files_for_authoritative_slate", lambda _date: [])
    monkeypatch.setattr(app_module, "_find_next_available_smart_sim_date", lambda *_args, **_kwargs: (None, []))
    monkeypatch.setattr(
        app_module,
        "_load_game_odds_map",
        lambda _date: {
            ("PHX", "UTA"): {
                "home_team": "Phoenix Suns",
                "visitor_team": "Utah Jazz",
                "home_spread": -5.5,
                "total": 230.5,
                "commence_time": "2026-03-28T22:00:00Z",
            }
        },
    )
    monkeypatch.setattr(
        app_module,
        "_load_predictions_rows_map",
        lambda _date: {
            ("PHX", "UTA"): {
                "date": "2026-03-28",
                "home_team": "PHOENIX SUNS",
                "visitor_team": "UTAH JAZZ",
                "home_win_prob_cal": 0.61,
                "spread_margin": 5.1,
                "totals": 229.8,
                "commence_time": "2026-03-28T22:00:00Z",
            }
        },
    )
    monkeypatch.setattr(app_module, "_load_props_predictions_map", lambda _date: {})
    monkeypatch.setattr(app_module, "_compute_player_minutes_priors", lambda _date, days_back=21: {})
    monkeypatch.setattr(app_module, "_live_load_props_edges_index", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_props_recommendations_by_team", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_injury_context_map", lambda _date: {})
    monkeypatch.setattr(app_module, "_roster_players_for_date", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_best_bets_game_context", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_best_bets_props_prediction_lookup", lambda _date: {})
    monkeypatch.setattr(app_module, "_best_bets_load_injury_snapshot", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_cards_game_recommendations_index", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_cards_prop_snapshot_index", lambda _date: {})
    monkeypatch.setattr(app_module, "_build_cards_game_market_recommendations", lambda **kwargs: [])
    monkeypatch.setattr(app_module, "_matchup_writeup", lambda _game: "")

    def _unexpected_runtime(*args, **kwargs):
        raise AssertionError("runtime prop recompute should not run when a cards snapshot exists")

    monkeypatch.setattr(app_module, "_sim_vs_line_prop_recommendations", _unexpected_runtime)

    with app_module.app.test_request_context("/api/cards?date=2026-03-28"):
        response = app_module.api_cards()

    payload = response.get_json()
    game = payload["games"][0]

    assert game["home_tri"] == "PHX"
    assert game["away_tri"] == "UTA"
    assert len(game["prop_recommendations"]["home"]) == 1
    assert game["prop_recommendations"]["home"][0]["player"] == "Devin Booker"
    assert game["prop_recommendations"]["home"][0]["card_bucket"] == "official"


def test_build_cards_game_market_recommendations_skips_pregame_overrides_for_started_games():
    rows = [
        {
            "home": "Phoenix Suns",
            "away": "Utah Jazz",
            "market": "TOTAL",
            "side": "under",
            "line": 229.5,
            "price": -110,
            "ev": 0.05,
            "edge": -1.2,
            "date": "2026-03-28",
        }
    ]

    payload = app_module._build_cards_game_market_recommendations(
        date_str="2026-03-28",
        home_name="Phoenix Suns",
        away_name="Utah Jazz",
        home_tri="PHX",
        away_tri="UTA",
        bet={"total": 229.5, "under_ev": 0.05, "over_ev": -0.02},
        game_context={
            "by_pair": {
                ("PHX", "UTA"): {"pred_total_adjusted": 241.0, "pred_total": 241.0}
            }
        },
        injury_snapshot={},
        slate_total_median=None,
        raw_rows=rows,
        allow_pregame_updates=False,
    )

    assert len(payload) == 1
    assert payload[0]["market"] == "TOTAL"
    assert str(payload[0]["selection"] or "").lower() == "under"


def test_api_cards_started_game_uses_source_props_when_snapshot_missing(monkeypatch):
    now_local = app_module.datetime(2026, 3, 28, 19, 30, 0)

    monkeypatch.setattr(app_module, "_best_bets_local_now_naive", lambda: now_local)
    monkeypatch.setattr(
        app_module,
        "_live_build_scoreboard_games",
        lambda _date: (
            "espn",
            [
                {
                    "home": "PHX",
                    "away": "UTA",
                    "in_progress": True,
                    "final": False,
                    "status": "Q1 08:12",
                }
            ],
        ),
    )
    monkeypatch.setattr(app_module, "_load_smart_sim_files_for_authoritative_slate", lambda _date: [])
    monkeypatch.setattr(app_module, "_find_next_available_smart_sim_date", lambda *_args, **_kwargs: (None, []))
    monkeypatch.setattr(
        app_module,
        "_load_game_odds_map",
        lambda _date: {
            ("PHX", "UTA"): {
                "home_team": "Phoenix Suns",
                "visitor_team": "Utah Jazz",
                "home_spread": -5.5,
                "total": 229.5,
                "commence_time": "2026-03-28T22:00:00Z",
            }
        },
    )
    monkeypatch.setattr(
        app_module,
        "_load_predictions_rows_map",
        lambda _date: {
            ("PHX", "UTA"): {
                "date": "2026-03-28",
                "home_team": "PHOENIX SUNS",
                "visitor_team": "UTAH JAZZ",
                "commence_time": "2026-03-28T22:00:00Z",
            }
        },
    )
    monkeypatch.setattr(app_module, "_load_props_predictions_map", lambda _date: {})
    monkeypatch.setattr(app_module, "_compute_player_minutes_priors", lambda _date, days_back=21: {})
    monkeypatch.setattr(app_module, "_live_load_props_edges_index", lambda _date: {})
    monkeypatch.setattr(
        app_module,
        "_load_props_recommendations_by_team",
        lambda _date: {
            "PHX": [
                {
                    "player": "Devin Booker",
                    "top_play": {
                        "market": "pts",
                        "side": "OVER",
                        "line": 27.5,
                        "price": -110,
                        "ev": 0.11,
                        "ev_pct": 11.0,
                    },
                    "plays": [
                        {
                            "market": "pts",
                            "side": "OVER",
                            "line": 27.5,
                            "price": -110,
                            "ev": 0.11,
                            "ev_pct": 11.0,
                        }
                    ],
                    "top_play_reasons": ["Hold the original pregame play"],
                    "top_play_explain": "Frozen after tip.",
                    "top_play_consensus": 0.75,
                    "top_play_line_adv": 1.0,
                }
            ]
        },
    )
    monkeypatch.setattr(app_module, "_load_injury_context_map", lambda _date: {})
    monkeypatch.setattr(app_module, "_roster_players_for_date", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_best_bets_game_context", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_best_bets_props_prediction_lookup", lambda _date: {})
    monkeypatch.setattr(app_module, "_best_bets_load_injury_snapshot", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_cards_game_recommendations_index", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_cards_prop_snapshot_index", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_cards_prop_recommendations_index", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_cards_sim_detail_index", lambda _date: {})
    monkeypatch.setattr(app_module, "_build_cards_game_market_recommendations", lambda **kwargs: [])
    monkeypatch.setattr(app_module, "_matchup_writeup", lambda _game: "")

    def _unexpected_runtime(*args, **kwargs):
        raise AssertionError("runtime prop recompute should not run for started games without a snapshot")

    monkeypatch.setattr(app_module, "_sim_vs_line_prop_recommendations", _unexpected_runtime)

    with app_module.app.test_request_context("/api/cards?date=2026-03-28"):
        response = app_module.api_cards()

    payload = response.get_json()
    game = payload["games"][0]

    assert game["plays_locked"] is True
    assert len(game["prop_recommendations"]["home"]) == 1
    assert game["prop_recommendations"]["home"][0]["player"] == "Devin Booker"
    assert game["prop_recommendations"]["home"][0]["best"]["line"] == 27.5


def test_api_cards_enriches_each_prop_recommendation_once(tmp_path, monkeypatch):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)

    smart_sim_path = processed / "smart_sim_2026-04-02_PHX_UTA.json"
    smart_sim_path.write_text(
        json.dumps(
            {
                "home": "PHX",
                "away": "UTA",
                "score": {},
                "players": {"home": [], "away": []},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(app_module, "_load_smart_sim_files_for_authoritative_slate", lambda _date: [smart_sim_path])
    monkeypatch.setattr(app_module, "_smart_sim_matchup_from_path", lambda _date, _path, prefix=None: ("PHX", "UTA"))
    monkeypatch.setattr(
        app_module,
        "_load_game_odds_map",
        lambda _date: {("PHX", "UTA"): {"home_team": "Phoenix Suns", "visitor_team": "Utah Jazz"}},
    )
    monkeypatch.setattr(app_module, "_load_predictions_rows_map", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_props_predictions_map", lambda _date: {})
    monkeypatch.setattr(app_module, "_compute_player_minutes_priors", lambda _date, days_back=21: {})
    monkeypatch.setattr(app_module, "_live_load_props_edges_index", lambda _date: {})
    monkeypatch.setattr(
        app_module,
        "_load_props_recommendations_by_team",
        lambda _date: {"PHX": [{"player": "Devin Booker", "team": "PHX"}]},
    )
    monkeypatch.setattr(app_module, "_load_injury_context_map", lambda _date: {})
    monkeypatch.setattr(app_module, "_roster_players_for_date", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_best_bets_game_context", lambda _date: {})
    monkeypatch.setattr(
        app_module,
        "_load_best_bets_props_prediction_lookup",
        lambda _date: {(app_module._norm_player_name("Devin Booker"), "PHX"): {"pts_mean": 28.0}},
    )
    monkeypatch.setattr(app_module, "_best_bets_load_injury_snapshot", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_cards_game_recommendations_index", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_cards_prop_snapshot_index", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_cards_prop_recommendations_index", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_cards_sim_detail_index", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_finals_lookup", lambda _date: {})
    monkeypatch.setattr(
        app_module,
        "_load_recon_props_lookup",
        lambda _date: (
            {},
            {
                ("PHX", app_module._norm_player_name("Devin Booker")): {"pts": 30.0},
            },
        ),
    )
    monkeypatch.setattr(app_module, "_cards_started_matchups_index", lambda _date, _now=None: {})
    monkeypatch.setattr(
        app_module,
        "_build_cards_prop_recommendations_from_source",
        lambda *args, **kwargs: {
            "home": [
                {
                    "team": "PHX",
                    "player": "Devin Booker",
                    "best": {"market": "pts", "side": "OVER", "line": 27.5, "price": -110},
                }
            ],
            "away": [],
        },
    )

    counts = {"flatten": 0, "decorate": 0, "settle": 0}

    def _fake_flatten(row, **kwargs):
        counts["flatten"] += 1
        return row

    def _fake_decorate(**kwargs):
        counts["decorate"] += 1
        return {
            "reasons": ["Reason one"],
            "top_play_reasons": ["Reason one"],
            "basketball_reasons": ["Reason one"],
        }

    def _fake_settle_prop_pick(*, actual, line, side):
        counts["settle"] += 1
        return "win"

    def _fake_apply_buckets(prop_recommendations, *, snapshot_picks):
        for rows in prop_recommendations.values():
            for row in rows:
                if isinstance(row, dict):
                    row["card_bucket"] = "official"
                    row["card_rank"] = 1

    monkeypatch.setattr(app_module, "_flatten_prop_recommendation_row", _fake_flatten)
    monkeypatch.setattr(app_module, "_decorate_prop_recommendation_payload", _fake_decorate)
    monkeypatch.setattr(app_module, "_best_bets_settle_prop_pick", _fake_settle_prop_pick)
    monkeypatch.setattr(app_module, "_apply_cards_prop_recommendation_buckets", _fake_apply_buckets)
    monkeypatch.setattr(app_module, "_build_cards_game_market_recommendations", lambda **kwargs: [])
    monkeypatch.setattr(app_module, "_matchup_writeup", lambda _game: "")

    with app_module.app.test_request_context("/api/cards?date=2026-04-02&props_source=source"):
        response = app_module.api_cards()

    payload = response.get_json()
    prop_row = payload["games"][0]["prop_recommendations"]["home"][0]

    assert counts == {"flatten": 1, "decorate": 1, "settle": 1}
    assert prop_row["top_play_reasons"] == ["Reason one"]
    assert prop_row["actual"] == 30.0
    assert prop_row["result"] == "win"


def test_api_cards_normalizes_snapshot_names_and_backfills_roster_coverage(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    processed = data_dir / "processed"
    raw = data_dir / "raw"
    overrides = data_dir / "overrides"
    processed.mkdir(parents=True)
    raw.mkdir(parents=True)
    overrides.mkdir(parents=True)

    det_was = processed / "smart_sim_2026-03-19_WAS_DET.json"
    det_was.write_text(
        json.dumps(
            {
                "home": "WAS",
                "away": "DET",
                "score": {},
                "players": {
                    "home": [{"player_name": "Bub Carrington", "player_id": 1, "pts_mean": 10.0}],
                    "away": [{"player_name": "Ronald Holland II", "player_id": 2, "pts_mean": 11.0}],
                },
            }
        ),
        encoding="utf-8",
    )
    sac_phi = processed / "smart_sim_2026-03-19_SAC_PHI.json"
    sac_phi.write_text(
        json.dumps(
            {
                "home": "SAC",
                "away": "PHI",
                "score": {},
                "players": {
                    "home": [{"player_name": "Keegan Murray", "player_id": 3, "pts_mean": 16.0}],
                    "away": [{"player_name": "Cameron Payne", "player_id": 4, "pts_mean": 8.0}],
                },
            }
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {"TEAM_ABBREVIATION": "DET", "PLAYER": "Ronald Holland II"},
            {"TEAM_ABBREVIATION": "WAS", "PLAYER": "Bub Carrington"},
            {"TEAM_ABBREVIATION": "SAC", "PLAYER": "Keegan Murray"},
        ]
    ).to_csv(processed / "league_status_2026-03-19.csv", index=False)
    pd.DataFrame(
        [
            {"TEAM_ABBREVIATION": "DET", "PLAYER": "Ronald Holland II", "PLAYER_ID": 2},
            {"TEAM_ABBREVIATION": "PHI", "PLAYER": "Cameron Payne", "PLAYER_ID": 4},
            {"TEAM_ABBREVIATION": "SAC", "PLAYER": "Keegan Murray", "PLAYER_ID": 3},
            {"TEAM_ABBREVIATION": "WAS", "PLAYER": "Bub Carrington", "PLAYER_ID": 1},
        ]
    ).to_csv(processed / "rosters_2025-26.csv", index=False)

    pd.DataFrame(
        [
            {
                "snapshot_ts": "2026-03-19T16:00:00Z",
                "event_id": "evt-det-was",
                "commence_time": "2026-03-19T23:00:00Z",
                "bookmaker": "fanduel",
                "bookmaker_title": "FanDuel",
                "market": "player_points",
                "outcome_name": "Over",
                "player_name": "Ron Holland",
                "point": 10.5,
                "price": -110,
                "last_update": "2026-03-19T16:01:00Z",
                "home_team": "Washington Wizards",
                "away_team": "Detroit Pistons",
            },
            {
                "snapshot_ts": "2026-03-19T16:00:00Z",
                "event_id": "evt-sac-phi",
                "commence_time": "2026-03-20T02:00:00Z",
                "bookmaker": "fanduel",
                "bookmaker_title": "FanDuel",
                "market": "player_points",
                "outcome_name": "Over",
                "player_name": "Cameron Payne",
                "point": 7.5,
                "price": -108,
                "last_update": "2026-03-19T16:01:00Z",
                "home_team": "Sacramento Kings",
                "away_team": "Philadelphia 76ers",
            },
        ]
    ).to_csv(raw / "odds_nba_player_props_2026-03-19.csv", index=False)

    monkeypatch.setattr(app_module, "DATA_PROCESSED_DIR", processed)
    monkeypatch.setattr(app_module, "DATA_RAW_DIR", raw)
    monkeypatch.setattr(app_module, "DATA_OVERRIDES_DIR", overrides)
    monkeypatch.setattr(app_module, "_load_smart_sim_files_for_authoritative_slate", lambda _date: [det_was, sac_phi])
    monkeypatch.setattr(
        app_module,
        "_load_game_odds_map",
        lambda _date: {
            ("WAS", "DET"): {"home_team": "Washington Wizards", "visitor_team": "Detroit Pistons"},
            ("SAC", "PHI"): {"home_team": "Sacramento Kings", "visitor_team": "Philadelphia 76ers"},
        },
    )
    monkeypatch.setattr(app_module, "_load_props_predictions_map", lambda _date: {})
    monkeypatch.setattr(app_module, "_compute_player_minutes_priors", lambda _date, days_back=21: {})
    monkeypatch.setattr(app_module, "_live_find_processed_csv", lambda _stem, _date: None)
    monkeypatch.setattr(app_module, "_live_load_props_edges_index", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_props_recommendations_by_team", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_injury_context_map", lambda _date: {})
    monkeypatch.setattr(app_module, "_matchup_writeup", lambda _game: "")

    with app_module.app.test_request_context("/api/cards?date=2026-03-19&include_players=1"):
        response = app_module.api_cards()

    payload = response.get_json()
    warnings = [warning for game in payload["games"] for warning in (game.get("warnings") or [])]

    assert all("Players with prop lines missing from SmartSim boxscore" not in warning for warning in warnings)


def test_cards_select_prop_buckets_does_not_cap_total_qualifying_props(monkeypatch):
    rows = [
        {"team": "PHX", "player": "Devin Booker", "row_id": 1},
        {"team": "PHX", "player": "Kevin Durant", "row_id": 2},
        {"team": "PHX", "player": "Bradley Beal", "row_id": 3},
    ]

    monkeypatch.setattr(app_module, "_cards_locked_policy_annotate", lambda row: dict(row))
    monkeypatch.setattr(app_module, "_cards_locked_policy_sort_key", lambda row: (float(row["row_id"]), 0.0, 0.0, 0.0, 0.0, 0.0))
    monkeypatch.setattr(app_module, "_cards_prop_sleeve_policy", lambda _row: None)
    monkeypatch.setattr(app_module, "_cards_prop_official_via_sleeve_policy", lambda _row: True)
    monkeypatch.setattr(app_module, "_cards_prop_playable_via_sleeve_policy", lambda _row: True)

    official, playable = app_module._cards_select_prop_buckets(rows)

    assert [row["player"] for row in official] == ["Bradley Beal", "Kevin Durant", "Devin Booker"]
    assert playable == []
    assert [row["card_rank"] for row in official] == [1, 2, 3]


def test_api_cards_normalizes_claxton_alias_without_missing_prop_warning(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    processed = data_dir / "processed"
    raw = data_dir / "raw"
    overrides = data_dir / "overrides"
    processed.mkdir(parents=True)
    raw.mkdir(parents=True)
    overrides.mkdir(parents=True)

    atl_bkn = processed / "smart_sim_2026-03-19_ATL_BKN.json"
    atl_bkn.write_text(
        json.dumps(
            {
                "home": "ATL",
                "away": "BKN",
                "score": {},
                "players": {
                    "home": [{"player_name": "Trae Young", "player_id": 1, "pts_mean": 28.0}],
                    "away": [{"player_name": "Nic Claxton", "player_id": 2, "reb_mean": 7.5}],
                },
            }
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {"TEAM_ABBREVIATION": "ATL", "PLAYER": "Trae Young", "PLAYER_ID": 1},
            {"TEAM_ABBREVIATION": "BKN", "PLAYER": "Nicolas Claxton", "PLAYER_ID": 2},
        ]
    ).to_csv(processed / "rosters_2025-26.csv", index=False)

    pd.DataFrame(
        [
            {
                "snapshot_ts": "2026-03-19T16:00:00Z",
                "event_id": "evt-atl-bkn",
                "commence_time": "2026-03-19T23:30:00Z",
                "bookmaker": "fanduel",
                "bookmaker_title": "FanDuel",
                "market": "player_rebounds",
                "outcome_name": "Over",
                "player_name": "Nicolas Claxton",
                "point": 6.5,
                "price": -115,
                "last_update": "2026-03-19T16:01:00Z",
                "home_team": "Atlanta Hawks",
                "away_team": "Brooklyn Nets",
            }
        ]
    ).to_csv(raw / "odds_nba_player_props_2026-03-19.csv", index=False)

    monkeypatch.setattr(app_module, "DATA_PROCESSED_DIR", processed)
    monkeypatch.setattr(app_module, "DATA_RAW_DIR", raw)
    monkeypatch.setattr(app_module, "DATA_OVERRIDES_DIR", overrides)
    monkeypatch.setattr(app_module, "_load_smart_sim_files_for_authoritative_slate", lambda _date: [atl_bkn])
    monkeypatch.setattr(
        app_module,
        "_load_game_odds_map",
        lambda _date: {("ATL", "BKN"): {"home_team": "Atlanta Hawks", "visitor_team": "Brooklyn Nets"}},
    )
    monkeypatch.setattr(app_module, "_load_props_predictions_map", lambda _date: {})
    monkeypatch.setattr(app_module, "_compute_player_minutes_priors", lambda _date, days_back=21: {})
    monkeypatch.setattr(app_module, "_live_find_processed_csv", lambda _stem, _date: None)
    monkeypatch.setattr(app_module, "_live_load_props_edges_index", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_props_recommendations_by_team", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_injury_context_map", lambda _date: {})
    monkeypatch.setattr(app_module, "_matchup_writeup", lambda _game: "")

    with app_module.app.test_request_context("/api/cards?date=2026-03-19&include_players=1"):
        response = app_module.api_cards()

    payload = response.get_json()
    warnings = [warning for game in payload["games"] for warning in (game.get("warnings") or [])]
    away_players = payload["games"][0]["sim"]["players"]["away"]
    claxton_row = next(row for row in away_players if row.get("player_name") == "Nic Claxton")

    assert all("Players with prop lines missing from SmartSim boxscore" not in warning for warning in warnings)
    assert claxton_row["prop_lines"]["reb"] == 6.5


def test_api_cards_filters_stale_smartsim_matchups_not_in_authoritative_slate(tmp_path, monkeypatch):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)

    (processed / "smart_sim_2026-03-18_MEM_DEN.json").write_text(
        json.dumps({"home": "MEM", "away": "DEN", "score": {}, "players": {"home": [], "away": []}}),
        encoding="utf-8",
    )
    (processed / "smart_sim_2026-03-18_MEM_NYK.json").write_text(
        json.dumps({"home": "MEM", "away": "NYK", "score": {}, "players": {"home": [], "away": []}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(app_module, "DATA_PROCESSED_DIR", processed)
    monkeypatch.setattr(
        app_module,
        "_load_game_odds_map",
        lambda _date: {("MEM", "DEN"): {"home_team": "Memphis Grizzlies", "visitor_team": "Denver Nuggets"}},
    )
    monkeypatch.setattr(app_module, "_load_props_predictions_map", lambda _date: {})
    monkeypatch.setattr(app_module, "_compute_player_minutes_priors", lambda _date, days_back=21: {})
    monkeypatch.setattr(app_module, "_live_find_processed_csv", lambda _stem, _date: None)
    monkeypatch.setattr(app_module, "_live_load_props_edges_index", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_props_recommendations_by_team", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_injury_context_map", lambda _date: {})
    monkeypatch.setattr(app_module, "_roster_players_for_date", lambda _date: {})
    monkeypatch.setattr(app_module, "_matchup_writeup", lambda _game: "")

    with app_module.app.test_request_context("/api/cards?date=2026-03-18"):
        response = app_module.api_cards()

    payload = response.get_json()
    games = payload["games"]

    assert len(games) == 1
    assert games[0]["home_tri"] == "MEM"
    assert games[0]["away_tri"] == "DEN"


def test_api_cards_surfaces_snapshot_prop_line_options_and_marks_recommendations(tmp_path, monkeypatch):
    processed = tmp_path / "data" / "processed"
    raw = tmp_path / "data" / "raw"
    processed.mkdir(parents=True)
    raw.mkdir(parents=True)

    smart_sim_path = processed / "smart_sim_2026-03-13_MEM_DET.json"
    smart_sim_path.write_text(
        json.dumps(
            {
                "home": "MEM",
                "away": "DET",
                "n_sims": 100,
                "score": {},
                "periods": [],
                "players": {
                    "home": [
                        {
                            "player_name": "Ja Morant",
                            "player_id": 1,
                            "pts_mean": 26.2,
                            "reb_mean": 5.1,
                            "ast_mean": 8.3,
                            "threes_mean": 1.9,
                            "stl_mean": 1.2,
                            "blk_mean": 0.3,
                            "tov_mean": 3.7,
                            "pra_mean": 39.6,
                        }
                    ],
                    "away": [
                        {
                            "player_name": "Cade Cunningham",
                            "player_id": 2,
                            "pts_mean": 25.4,
                            "reb_mean": 6.0,
                            "ast_mean": 8.8,
                            "threes_mean": 2.1,
                            "stl_mean": 1.1,
                            "blk_mean": 0.6,
                            "tov_mean": 4.0,
                            "pra_mean": 40.2,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "snapshot_ts": "2026-03-13T17:00:00Z",
                "event_id": "evt-1",
                "commence_time": "2026-03-13T23:00:00Z",
                "bookmaker": "fanduel",
                "bookmaker_title": "FanDuel",
                "market": "player_points",
                "outcome_name": "Over",
                "player_name": "Ja Morant",
                "point": 24.5,
                "price": -115,
                "last_update": "2026-03-13T17:01:00Z",
                "home_team": "Memphis Grizzlies",
                "away_team": "Detroit Pistons",
            },
            {
                "snapshot_ts": "2026-03-13T17:00:00Z",
                "event_id": "evt-1",
                "commence_time": "2026-03-13T23:00:00Z",
                "bookmaker": "draftkings",
                "bookmaker_title": "DraftKings",
                "market": "player_points",
                "outcome_name": "Over",
                "player_name": "Ja Morant",
                "point": 24.5,
                "price": -110,
                "last_update": "2026-03-13T17:01:30Z",
                "home_team": "Memphis Grizzlies",
                "away_team": "Detroit Pistons",
            },
            {
                "snapshot_ts": "2026-03-13T17:00:00Z",
                "event_id": "evt-1",
                "commence_time": "2026-03-13T23:00:00Z",
                "bookmaker": "fanduel",
                "bookmaker_title": "FanDuel",
                "market": "player_points",
                "outcome_name": "Under",
                "player_name": "Ja Morant",
                "point": 24.5,
                "price": -105,
                "last_update": "2026-03-13T17:01:00Z",
                "home_team": "Memphis Grizzlies",
                "away_team": "Detroit Pistons",
            },
            {
                "snapshot_ts": "2026-03-13T17:00:00Z",
                "event_id": "evt-1",
                "commence_time": "2026-03-13T23:00:00Z",
                "bookmaker": "betmgm",
                "bookmaker_title": "BetMGM",
                "market": "player_points",
                "outcome_name": "Over",
                "player_name": "Ja Morant",
                "point": 25.5,
                "price": 105,
                "last_update": "2026-03-13T17:02:00Z",
                "home_team": "Memphis Grizzlies",
                "away_team": "Detroit Pistons",
            },
            {
                "snapshot_ts": "2026-03-13T17:00:00Z",
                "event_id": "evt-1",
                "commence_time": "2026-03-13T23:00:00Z",
                "bookmaker": "fanduel",
                "bookmaker_title": "FanDuel",
                "market": "player_threes",
                "outcome_name": "Over",
                "player_name": "Cam Spencer",
                "point": 1.5,
                "price": -120,
                "last_update": "2026-03-13T17:03:00Z",
                "home_team": "Memphis Grizzlies",
                "away_team": "Detroit Pistons",
            },
            {
                "snapshot_ts": "2026-03-13T17:00:00Z",
                "event_id": "evt-1",
                "commence_time": "2026-03-13T23:00:00Z",
                "bookmaker": "fanduel",
                "bookmaker_title": "FanDuel",
                "market": "player_threes",
                "outcome_name": "Under",
                "player_name": "Cam Spencer",
                "point": 1.5,
                "price": -110,
                "last_update": "2026-03-13T17:03:00Z",
                "home_team": "Memphis Grizzlies",
                "away_team": "Detroit Pistons",
            },
        ]
    ).to_csv(raw / "odds_nba_player_props_2026-03-13.csv", index=False)

    monkeypatch.setattr(app_module, "DATA_PROCESSED_DIR", processed)
    monkeypatch.setattr(app_module, "DATA_RAW_DIR", raw)
    monkeypatch.setattr(app_module, "_load_smart_sim_files_for_date", lambda _date, prefix=None: [smart_sim_path])
    monkeypatch.setattr(app_module, "_load_game_odds_map", lambda _date: {("MEM", "DET"): {"home_team": "Memphis Grizzlies", "visitor_team": "Detroit Pistons"}})
    monkeypatch.setattr(app_module, "_load_props_predictions_map", lambda _date: {})
    monkeypatch.setattr(app_module, "_compute_player_minutes_priors", lambda _date, days_back=21: {})
    monkeypatch.setattr(app_module, "_live_find_processed_csv", lambda _stem, _date: None)
    monkeypatch.setattr(app_module, "_live_load_props_edges_index", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_props_recommendations_by_team", lambda _date: {})
    monkeypatch.setattr(app_module, "_load_injury_context_map", lambda _date: {})
    monkeypatch.setattr(
        app_module,
        "_roster_players_for_date",
        lambda _date: {
            "MEM": {
                app_module._norm_player_name_for_keys("Ja Morant"),
                app_module._norm_player_name_for_keys("Cam Spencer"),
            },
            "DET": {app_module._norm_player_name_for_keys("Cade Cunningham")},
        },
    )
    monkeypatch.setattr(app_module, "_matchup_writeup", lambda _game: "")
    monkeypatch.setattr(
        app_module,
        "_sim_vs_line_prop_recommendations",
        lambda players_out, props_recs_by_team, **kwargs: {
            "home": [
                {
                    "team": "MEM",
                    "player": "Ja Morant",
                    "best": {
                        "market": "pts",
                        "side": "OVER",
                        "line": 24.5,
                        "book": "fanduel",
                        "price": -115,
                        "ev_pct": 4.2,
                        "guidance": {
                            "action": "play",
                            "action_code": "PLAY",
                            "play_to_line": 25.5,
                            "summary": "Model clears the opener.",
                        },
                    },
                    "picks": [
                        {
                            "market": "pts",
                            "side": "OVER",
                            "line": 24.5,
                            "book": "fanduel",
                            "price": -115,
                            "ev_pct": 4.2,
                            "guidance": {
                                "action": "play",
                                "action_code": "PLAY",
                                "play_to_line": 25.5,
                                "summary": "Model clears the opener.",
                            },
                        }
                    ],
                }
            ],
            "away": [],
        },
    )

    with app_module.app.test_request_context("/api/cards?date=2026-03-13&include_players=1"):
        response = app_module.api_cards()

    payload = response.get_json()
    game = payload["games"][0]
    assert isinstance(game["game_market_recommendations"], list)
    assert game["sim"]["players_loaded"] is True

    prop_rows = game["prop_recommendations"]["home"]
    assert len(prop_rows) == 1
    assert prop_rows[0]["card_bucket"] == "official"
    assert prop_rows[0]["card_rank"] == 1

    ja_row = next(row for row in game["sim"]["players"]["home"] if row["player_name"] == "Ja Morant")
    pts_options = ja_row["prop_line_options"]["pts"]

    assert any(option["side"] == "OVER" and option["line"] == 24.5 and option["book_count"] == 2 for option in pts_options)
    recommended_option = next(option for option in pts_options if option["side"] == "OVER" and option["line"] == 24.5)
    assert recommended_option["recommended"] is True
    assert recommended_option["recommended_primary"] is True
    assert recommended_option["recommendation_action"] == "play"
    assert recommended_option["recommendation_play_to_line"] == 25.5

    missing_home = game["sim"]["missing_prop_players"]["home"]
    cam_row = next(row for row in missing_home if row["player_name"] == "Cam Spencer")

    assert "threes" in cam_row["prop_line_options"]
    assert any(option["side"] == "OVER" and option["line"] == 1.5 for option in cam_row["prop_line_options"]["threes"])


def test_live_oddsapi_player_props_maps_alternate_threes_lines(monkeypatch):
    class _FakeResponse:
        def __init__(self, payload, ok=True):
            self._payload = payload
            self.ok = ok

        def json(self):
            return self._payload

    def _fake_get(url, params=None, headers=None, timeout=None):
        if url.endswith("/v4/sports/basketball_nba/events"):
            return _FakeResponse(
                [
                    {
                        "id": "evt-1",
                        "home_team": "Houston Rockets",
                        "away_team": "Los Angeles Lakers",
                        "commence_time": "2026-03-16T23:00:00Z",
                    }
                ]
            )
        if url.endswith("/v4/sports/basketball_nba/events/evt-1/markets"):
            return _FakeResponse(
                {
                    "bookmakers": [
                        {
                            "key": "draftkings",
                            "markets": [
                                {"key": "player_threes_alternate"},
                            ],
                        }
                    ]
                }
            )
        if url.endswith("/v4/sports/basketball_nba/events/evt-1/odds"):
            return _FakeResponse(
                {
                    "bookmakers": [
                        {
                            "key": "draftkings",
                            "last_update": "2026-03-16T23:05:00Z",
                            "markets": [
                                {
                                    "key": "player_threes_alternate",
                                    "last_update": "2026-03-16T23:05:10Z",
                                    "outcomes": [
                                        {
                                            "name": "Over",
                                            "description": "Austin Reaves",
                                            "point": 1.5,
                                            "price": -120,
                                        },
                                        {
                                            "name": "Under",
                                            "description": "Austin Reaves",
                                            "point": 1.5,
                                            "price": 100,
                                        },
                                    ],
                                }
                            ],
                        }
                    ]
                }
            )
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    monkeypatch.setenv("LIVE_PLAYER_PROPS_ODDSAPI", "1")
    monkeypatch.setenv("LIVE_PLAYER_PROPS_ALL_MARKETS", "0")
    monkeypatch.setattr(app_module.requests, "get", _fake_get)

    app_module._live_oddsapi_player_props_cache.clear()
    app_module._live_oddsapi_player_props_event_cache.clear()
    app_module._live_oddsapi_player_props_markets_cache.clear()

    meta = app_module._live_oddsapi_player_props_for_game("2026-03-16", "HOU", "LAL")

    assert meta["markets_requested"] == ["player_threes_alternate"]
    assert meta["lines"]["AUSTIN REAVES|threes"] == 1.5
    assert meta["prices"]["AUSTIN REAVES|threes"] == {"over": -120.0, "under": 100.0}


def test_api_live_player_lens_blends_adjusted_pregame_prior_into_player_projection(monkeypatch):
    nk = app_module._norm_player_name("Devin Booker")

    monkeypatch.setattr(
        app_module,
        "_load_best_bets_game_context",
        lambda _date: {
            "by_pair": {
                ("PHX", "UTA"): {
                    "home_team": "Phoenix Suns",
                    "away_team": "Utah Jazz",
                    "home_tri": "PHX",
                    "away_tri": "UTA",
                    "pred_total_raw": 210.0,
                    "pred_total_adjusted": 236.0,
                    "pred_total": 236.0,
                    "pred_margin_raw": 0.0,
                    "pred_margin_adjusted": 8.0,
                    "pred_margin": 8.0,
                    "home_pred_points": 120.0,
                    "away_pred_points": 116.0,
                    "home_win_prob_adjusted": 0.69,
                }
            }
        },
    )
    monkeypatch.setattr(
        app_module,
        "_live_build_scoreboard_games",
        lambda _date: (
            "espn",
            [
                {
                    "espn_event_id": "evt-1",
                    "game_id": "001",
                    "home": "PHX",
                    "away": "UTA",
                    "period": 2,
                    "clock": "06:00",
                    "in_progress": True,
                    "final": False,
                    "home_pts": 55,
                    "away_pts": 48,
                }
            ],
        ),
    )
    monkeypatch.setattr(app_module, "_live_fetch_espn_summary", lambda _eid: {"ok": True})
    monkeypatch.setattr(
        app_module,
        "_live_extract_player_boxscore_from_espn_summary",
        lambda _summary: [
            {
                "team_tri": "PHX",
                "player": "Devin Booker",
                "player_id": 1626164,
                "starter": True,
                "mp": 12,
                "pf": 1,
                "pts": 10,
                "reb": 2,
                "ast": 3,
                "threes_made": 2,
                "stl": 1,
                "blk": 0,
                "tov": 1,
            }
        ],
    )
    monkeypatch.setattr(app_module, "_live_load_props_edges_index", lambda _date: {("PHX", nk, "pts"): 20.5})
    monkeypatch.setattr(app_module, "_live_load_props_recommendations_line_index", lambda _date: {})
    monkeypatch.setattr(app_module, "_live_load_props_predictions_index", lambda _date: {("PHX", nk): {"pred_pts": 20.0, "roll10_min": 36.0}})
    monkeypatch.setattr(app_module, "_live_roster_pid_by_team_nk", lambda: {("PHX", nk): 1626164})
    monkeypatch.setattr(app_module, "_live_oddsapi_player_props_for_game", lambda _date, _home, _away: {})
    monkeypatch.setattr(app_module, "_live_load_smart_sim_by_game_id", lambda _date, _gid: {})
    monkeypatch.setattr(app_module, "_live_fetch_pbp_actions", lambda _gid: [])
    monkeypatch.setattr(app_module, "_live_append_snapshot", lambda *_args, **_kwargs: None)

    app_module._live_player_lens_multi_cache.clear()

    with app_module.app.test_request_context("/api/live_player_lens?date=2026-03-30&event_ids=evt-1"):
        response = app_module.api_live_player_lens()

    payload = response.get_json()
    assert payload["ok"] is True
    game = payload["games"][0]
    assert game["pregame_prior"]["pred_total_adjusted"] == 236.0

    pts_row = next(row for row in game["rows"] if row["player"] == "Devin Booker" and row["stat"] == "pts")
    assert pts_row["sim_mu"] == 20.0
    assert pts_row["sim_mu_adjusted"] > pts_row["sim_mu"]
    assert pts_row["pregame_team_total_ratio"] > 1.0
    assert pts_row["pregame_stat_multiplier"] > 1.0
    assert pts_row["pace_proj"] > 26.0
    assert pts_row["sim_vs_line_adjusted"] > pts_row["sim_vs_line"]
    assert pts_row["pregame_margin_blended"] > 0.0


def test_api_live_player_lens_separates_debug_and_compact_cache_entries(monkeypatch):
    nk = app_module._norm_player_name("Devin Booker")
    calls = {"summary": 0}

    monkeypatch.setattr(app_module, "_load_best_bets_game_context", lambda _date: {"by_pair": {}})
    monkeypatch.setattr(
        app_module,
        "_live_build_scoreboard_games",
        lambda _date: (
            "espn",
            [
                {
                    "espn_event_id": "evt-1",
                    "game_id": "001",
                    "home": "PHX",
                    "away": "UTA",
                    "period": 2,
                    "clock": "06:00",
                    "in_progress": True,
                    "final": False,
                    "home_pts": 55,
                    "away_pts": 48,
                }
            ],
        ),
    )

    def _fake_summary(_eid):
        calls["summary"] += 1
        return {"ok": True}

    monkeypatch.setattr(app_module, "_live_fetch_espn_summary", _fake_summary)
    monkeypatch.setattr(
        app_module,
        "_live_extract_player_boxscore_from_espn_summary",
        lambda _summary: [
            {
                "team_tri": "PHX",
                "player": "Devin Booker",
                "player_id": 1626164,
                "starter": True,
                "mp": 12,
                "pf": 1,
                "pts": 10,
                "reb": 2,
                "ast": 3,
                "threes_made": 2,
                "stl": 1,
                "blk": 0,
                "tov": 1,
            }
        ],
    )
    monkeypatch.setattr(app_module, "_live_load_props_edges_index", lambda _date: {("PHX", nk, "pts"): 20.5})
    monkeypatch.setattr(app_module, "_live_load_props_recommendations_line_index", lambda _date: {})
    monkeypatch.setattr(app_module, "_live_load_props_predictions_index", lambda _date: {("PHX", nk): {"pred_pts": 20.0, "roll10_min": 36.0}})
    monkeypatch.setattr(app_module, "_live_roster_pid_by_team_nk", lambda: {("PHX", nk): 1626164})
    monkeypatch.setattr(app_module, "_live_oddsapi_player_props_for_game", lambda _date, _home, _away: {})
    monkeypatch.setattr(app_module, "_live_load_smart_sim_by_game_id", lambda _date, _gid: {})
    monkeypatch.setattr(app_module, "_live_fetch_pbp_actions", lambda _gid: [])
    monkeypatch.setattr(app_module, "_live_append_snapshot", lambda *_args, **_kwargs: None)

    app_module._live_player_lens_multi_cache.clear()

    with app_module.app.test_request_context("/api/live_player_lens?date=2026-03-30&event_ids=evt-1"):
        compact_response = app_module.api_live_player_lens()

    with app_module.app.test_request_context("/api/live_player_lens?date=2026-03-30&event_ids=evt-1&include_debug_rows=1"):
        debug_response = app_module.api_live_player_lens()

    compact_payload = compact_response.get_json()
    debug_payload = debug_response.get_json()
    compact_row = next(row for row in compact_payload["games"][0]["rows"] if row["player"] == "Devin Booker" and row["stat"] == "pts")
    debug_row = next(row for row in debug_payload["games"][0]["rows"] if row["player"] == "Devin Booker" and row["stat"] == "pts")

    assert calls["summary"] == 2
    assert "mp" not in compact_row
    assert debug_row["mp"] == 12


def test_cards_started_matchups_index_uses_live_state_cache(monkeypatch):
    now_local = pd.Timestamp("2026-03-30T19:00:00").to_pydatetime()
    app_module._live_state_cache.clear()
    app_module._live_state_cache["2026-03-30:12"] = (
        time.time(),
        {
            "date": "2026-03-30",
            "ttl": 12,
            "games": [
                {
                    "game_id": "001",
                    "home": "PHX",
                    "away": "UTA",
                    "home_pts": 55,
                    "away_pts": 48,
                    "status": "Q2 06:00",
                    "period": 2,
                    "clock": "06:00",
                    "in_progress": True,
                    "final": False,
                    "periods": [],
                }
            ],
        },
    )

    def _unexpected_scoreboard(_date):
        raise AssertionError("scoreboard fetch should not run when live-state cache is warm")

    monkeypatch.setattr(app_module, "_live_build_scoreboard_games", _unexpected_scoreboard)

    started = app_module._cards_started_matchups_index("2026-03-30", now_local=now_local)

    assert started[("PHX", "UTA")]["in_progress"] is True
    assert started[("PHX", "UTA")]["period"] == 2


def test_maybe_fetch_remote_processed_caches_misses(monkeypatch, tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)

    calls = {"count": 0}

    class _Resp:
        status_code = 404
        content = b""

    def _fake_get(url, timeout=None, headers=None):
        calls["count"] += 1
        return _Resp()

    monkeypatch.setenv("ALLOW_REMOTE_ARTIFACTS", "1")
    monkeypatch.setenv("REMOTE_PROCESSED_MISS_TTL_SEC", "900")
    monkeypatch.setattr(app_module, "DATA_PROCESSED_DIR", Path(processed))
    monkeypatch.setattr(requests, "get", _fake_get)
    app_module._remote_processed_miss_cache.clear()

    fname = "cards_props_snapshot_2026-05-09.json"

    assert app_module._maybe_fetch_remote_processed(fname) is None
    assert app_module._maybe_fetch_remote_processed(fname) is None
    assert calls["count"] == 1


def test_api_live_player_lens_date_only_returns_empty_payload_when_no_live_events(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "_live_build_scoreboard_games",
        lambda _date: (
            "espn",
            [
                {"espn_event_id": "evt-final", "in_progress": False, "final": True},
                {"espn_event_id": "evt-scheduled", "in_progress": False, "final": False},
            ],
        ),
    )

    app_module._live_player_lens_multi_cache.clear()

    with app_module.app.test_request_context("/api/live_player_lens?date=2026-03-30"):
        response = app_module.api_live_player_lens()

    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["date"] == "2026-03-30"
    assert payload["games"] == []


def test_api_live_player_lens_uses_pbp_event_timing_for_live_prop_projection(monkeypatch):
    nk = app_module._norm_player_name("Devin Booker")

    monkeypatch.setattr(
        app_module,
        "_load_best_bets_game_context",
        lambda _date: {"by_pair": {}},
    )
    monkeypatch.setattr(
        app_module,
        "_live_build_scoreboard_games",
        lambda _date: (
            "espn",
            [
                {
                    "espn_event_id": "evt-1",
                    "game_id": "001",
                    "home": "PHX",
                    "away": "UTA",
                    "period": 2,
                    "clock": "06:00",
                    "in_progress": True,
                    "final": False,
                    "home_pts": 55,
                    "away_pts": 48,
                }
            ],
        ),
    )
    monkeypatch.setattr(app_module, "_live_fetch_espn_summary", lambda _eid: {"ok": True})
    monkeypatch.setattr(
        app_module,
        "_live_extract_player_boxscore_from_espn_summary",
        lambda _summary: [
            {
                "team_tri": "PHX",
                "player": "Devin Booker",
                "player_id": 1626164,
                "starter": True,
                "mp": 12,
                "pf": 1,
                "pts": 10,
                "reb": 2,
                "ast": 3,
                "threes_made": 2,
                "stl": 1,
                "blk": 0,
                "tov": 1,
            }
        ],
    )
    monkeypatch.setattr(app_module, "_live_load_props_edges_index", lambda _date: {("PHX", nk, "pts"): 20.5})
    monkeypatch.setattr(app_module, "_live_load_props_recommendations_line_index", lambda _date: {})
    monkeypatch.setattr(app_module, "_live_load_props_predictions_index", lambda _date: {("PHX", nk): {"pred_pts": 20.0, "roll10_min": 36.0}})
    monkeypatch.setattr(app_module, "_live_roster_pid_by_team_nk", lambda: {("PHX", nk): 1626164})
    monkeypatch.setattr(app_module, "_live_oddsapi_player_props_for_game", lambda _date, _home, _away: {})
    monkeypatch.setattr(app_module, "_live_load_smart_sim_by_game_id", lambda _date, _gid: {})
    monkeypatch.setattr(app_module, "_live_append_snapshot", lambda *_args, **_kwargs: None)

    def _fake_actions(_gid):
        return [
            {
                "period": 1,
                "clock": "11:30",
                "teamTricode": "PHX",
                "playerName": "Devin Booker",
                "actionType": "Jump Shot",
                "shotResult": "Made",
                "isFieldGoal": 1,
                "shotValue": 2,
            },
            {
                "period": 1,
                "clock": "08:00",
                "teamTricode": "PHX",
                "playerName": "Devin Booker",
                "actionType": "Jump Shot",
                "shotResult": "Made",
                "isFieldGoal": 1,
                "shotValue": 2,
            },
            {
                "period": 1,
                "clock": "06:00",
                "teamTricode": "PHX",
                "playerName": "Devin Booker",
                "actionType": "Jump Shot",
                "shotResult": "Made",
                "isFieldGoal": 1,
                "shotValue": 2,
            },
            {
                "period": 2,
                "clock": "08:20",
                "teamTricode": "PHX",
                "playerName": "Devin Booker",
                "actionType": "Jump Shot",
                "shotResult": "Made",
                "isFieldGoal": 1,
                "shotValue": 2,
            },
            {
                "period": 2,
                "clock": "07:45",
                "teamTricode": "PHX",
                "playerName": "Devin Booker",
                "actionType": "Jump Shot",
                "shotResult": "Made",
                "isFieldGoal": 1,
                "shotValue": 2,
            },
            {
                "period": 2,
                "clock": "07:00",
                "teamTricode": "PHX",
                "playerName": "Devin Booker",
                "actionType": "Jump Shot",
                "shotResult": "Made",
                "isFieldGoal": 1,
                "shotValue": 2,
            },
        ]

    monkeypatch.setattr(app_module, "_live_fetch_pbp_actions", _fake_actions)

    app_module._live_player_lens_multi_cache.clear()

    with app_module.app.test_request_context("/api/live_player_lens?date=2026-03-30&event_ids=evt-1&recent_window_sec=180"):
        response = app_module.api_live_player_lens()

    payload = response.get_json()
    pts_row = next(row for row in payload["games"][0]["rows"] if row["player"] == "Devin Booker" and row["stat"] == "pts")

    assert pts_row["event_timing_mult"] is not None
    assert pts_row["event_timing_mult"] > 1.0
    assert pts_row["event_rate_recent"] > pts_row["event_rate_game"]
    assert pts_row["points_rate_recent"] > pts_row["points_rate_game"]
    assert pts_row["seconds_since_event"] == 60
    assert pts_row["seconds_since_score"] == 60
    assert pts_row["pace_proj"] > 30.0


def test_api_live_player_lens_damps_late_competitive_script_lift(monkeypatch):
    nk = app_module._norm_player_name("Devin Booker")

    monkeypatch.setattr(
        app_module,
        "_load_best_bets_game_context",
        lambda _date: {
            "by_pair": {
                ("PHX", "UTA"): {
                    "home_team": "Phoenix Suns",
                    "away_team": "Utah Jazz",
                    "home_tri": "PHX",
                    "away_tri": "UTA",
                    "pred_total_raw": 210.0,
                    "pred_total_adjusted": 236.0,
                    "pred_total": 236.0,
                    "pred_margin_raw": 0.0,
                    "pred_margin_adjusted": 8.0,
                    "pred_margin": 8.0,
                    "home_pred_points": 120.0,
                    "away_pred_points": 116.0,
                    "home_win_prob_adjusted": 0.69,
                }
            }
        },
    )

    scoreboard = {
        "home_pts": 95,
        "away_pts": 70,
    }

    def _fake_scoreboard(_date):
        return (
            "espn",
            [
                {
                    "espn_event_id": "evt-1",
                    "game_id": "001",
                    "home": "PHX",
                    "away": "UTA",
                    "period": 4,
                    "clock": "04:00",
                    "in_progress": True,
                    "final": False,
                    "home_pts": scoreboard["home_pts"],
                    "away_pts": scoreboard["away_pts"],
                }
            ],
        )

    monkeypatch.setattr(app_module, "_live_build_scoreboard_games", _fake_scoreboard)
    monkeypatch.setattr(app_module, "_live_fetch_espn_summary", lambda _eid: {"ok": True})
    monkeypatch.setattr(
        app_module,
        "_live_extract_player_boxscore_from_espn_summary",
        lambda _summary: [
            {
                "team_tri": "PHX",
                "player": "Devin Booker",
                "player_id": 1626164,
                "starter": True,
                "mp": 28,
                "pf": 1,
                "pts": 22,
                "reb": 4,
                "ast": 6,
                "threes_made": 3,
                "stl": 1,
                "blk": 0,
                "tov": 1,
            }
        ],
    )
    monkeypatch.setattr(app_module, "_live_load_props_edges_index", lambda _date: {("PHX", nk, "pts"): 26.5})
    monkeypatch.setattr(app_module, "_live_load_props_recommendations_line_index", lambda _date: {})
    monkeypatch.setattr(app_module, "_live_load_props_predictions_index", lambda _date: {("PHX", nk): {"pred_pts": 20.0, "roll10_min": 36.0}})
    monkeypatch.setattr(app_module, "_live_roster_pid_by_team_nk", lambda: {("PHX", nk): 1626164})
    monkeypatch.setattr(app_module, "_live_oddsapi_player_props_for_game", lambda _date, _home, _away: {})
    monkeypatch.setattr(app_module, "_live_load_smart_sim_by_game_id", lambda _date, _gid: {})
    monkeypatch.setattr(app_module, "_live_fetch_pbp_actions", lambda _gid: [])
    monkeypatch.setattr(app_module, "_live_append_snapshot", lambda *_args, **_kwargs: None)

    def _run_payload() -> dict:
        app_module._live_player_lens_multi_cache.clear()
        with app_module.app.test_request_context("/api/live_player_lens?date=2026-03-30&event_ids=evt-1"):
            response = app_module.api_live_player_lens()
        return response.get_json()

    blowout_payload = _run_payload()
    blowout_row = next(row for row in blowout_payload["games"][0]["rows"] if row["player"] == "Devin Booker" and row["stat"] == "pts")

    scoreboard["home_pts"] = 95
    scoreboard["away_pts"] = 92
    close_payload = _run_payload()
    close_row = next(row for row in close_payload["games"][0]["rows"] if row["player"] == "Devin Booker" and row["stat"] == "pts")

    assert blowout_row["pregame_margin_blended"] > 15.0
    assert abs(close_row["pregame_margin_blended"]) < 8.0
    assert close_row["sim_mu_adjusted"] < blowout_row["sim_mu_adjusted"]
    assert close_row["pregame_stat_multiplier"] < blowout_row["pregame_stat_multiplier"]
    assert close_row["sim_vs_line_adjusted"] < blowout_row["sim_vs_line_adjusted"]


def test_api_live_player_props_projection_audit_scores_adjusted_rows(tmp_path, monkeypatch):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)

    (processed / "live_lens_projections_2026-03-29.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "date": "2026-03-29",
                        "market": "player_prop",
                        "game_id": "0000000001",
                        "home": "PHX",
                        "away": "UTA",
                        "player": "Devin Booker",
                        "name_key": "Devin Booker",
                        "team_tri": "PHX",
                        "stat": "pts",
                        "proj": 31.0,
                        "sim_mu": 27.0,
                        "sim_mu_adjusted": 29.5,
                        "elapsed": 30.0,
                        "strength": 4.2,
                        "received_at": "2026-03-29T22:10:00Z",
                        "context": {
                            "pregame_team_total_ratio": 1.09,
                            "pregame_game_total_ratio": 1.05,
                            "pregame_margin_blended": 7.0,
                            "pregame_stat_multiplier": 1.09,
                            "sim_vs_line": 1.5,
                            "sim_vs_line_adjusted": 4.0,
                        },
                    }
                )
            ]
        ) + "\n",
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "game_id": "0000000001",
                "player_name": "Devin Booker",
                "pts": 30,
                "reb": 4,
                "ast": 6,
                "threes": 3,
                "stl": 1,
                "blk": 0,
                "tov": 2,
            }
        ]
    ).to_csv(processed / "recon_props_2026-03-29.csv", index=False)

    monkeypatch.setattr(app_module, "DATA_PROCESSED_DIR", processed)
    monkeypatch.setenv("NBA_LIVE_LENS_DIR", str(processed))

    with app_module.app.test_client() as client:
        resp = client.get("/api/live_player_props_projection_audit?date=2026-03-29")

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["ok"] is True
    assert payload["overall"]["n"] == 1
    assert payload["overall"]["mae_proj"] == 1.0
    assert payload["overall"]["mae_adjusted"] == 0.5
    assert payload["overall"]["mae_raw"] == 3.0
    assert payload["overall"]["adjusted_beats_raw_rate"] == 1.0
    assert payload["overall"]["proj_beats_adjusted_rate"] == 0.0
    assert payload["by_stat"][0]["stat"] == "pts"


def test_api_live_player_props_projection_audit_backfills_missing_recon_props(tmp_path, monkeypatch):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)

    (processed / "live_lens_projections_2026-03-30.jsonl").write_text(
        json.dumps(
            {
                "date": "2026-03-30",
                "market": "player_prop",
                "game_id": "0000000002",
                "home": "PHX",
                "away": "UTA",
                "player": "Devin Booker",
                "name_key": "DEVIN BOOKER",
                "team_tri": "PHX",
                "stat": "pts",
                "proj": 28.0,
                "sim_mu": 25.0,
                "sim_mu_adjusted": 26.5,
                "elapsed": 18.0,
                "strength": 3.1,
                "received_at": "2026-03-30T21:10:00Z",
                "context": {},
            }
        ) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(app_module, "DATA_PROCESSED_DIR", processed)
    monkeypatch.setenv("NBA_LIVE_LENS_DIR", str(processed))
    app_module._load_recon_props_frame.cache_clear()

    def _fake_backfill(_date):
        return (
            pd.DataFrame(
                [
                    {
                        "date": "2026-03-30",
                        "game_id": "0000000002",
                        "player_id": 1626164,
                        "player_name": "Devin Booker",
                        "team_abbr": "PHX",
                        "pts": 27,
                        "reb": 4,
                        "ast": 6,
                        "threes": 3,
                        "stl": 1,
                        "blk": 0,
                        "tov": 2,
                        "pra": 37,
                    }
                ]
            ),
            "recon_props_2026-03-30.csv",
        )

    monkeypatch.setattr(app_module, "_maybe_backfill_recon_props_frame", _fake_backfill)

    with app_module.app.test_client() as client:
        resp = client.get("/api/live_player_props_projection_audit?date=2026-03-30")

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["ok"] is True
    assert payload["overall"]["n"] == 1
    assert payload["overall"]["mae_proj"] == 1.0
    assert payload["overall"]["mae_adjusted"] == 0.5


def test_api_live_player_props_projection_audit_replays_close_game_signals(tmp_path, monkeypatch):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)

    (processed / "live_lens_signals_2026-03-31.jsonl").write_text(
        json.dumps(
            {
                "date": "2026-03-31",
                "market": "player_prop",
                "game_id": "0000000003",
                "event_id": "401999999",
                "home": "PHX",
                "away": "UTA",
                "period": 4,
                "sec_left_period": 300,
                "player": "Devin Booker",
                "name_key": "Devin Booker",
                "team_tri": "PHX",
                "stat": "pts",
                "side": "OVER",
                "mp": 20.0,
                "pf": 2.0,
                "starter": True,
                "line": 35.0,
                "pace_proj": 40.0,
                "sim_mu": 30.0,
                "sim_mu_adjusted": 36.0,
                "strength": 5.0,
                "elapsed": 43.0,
                "received_at": "2026-03-31T22:10:00Z",
                "context": {
                    "pregame_team_total_ratio": 1.08,
                    "pregame_game_total_ratio": 1.04,
                    "pregame_margin_blended": 6.0,
                    "pregame_stat_multiplier": 1.2,
                    "pace_mult": 1.1,
                    "role_mult": 1.1,
                    "event_timing_mult": 1.05,
                    "hot_cold_mult": 1.0,
                    "foul_mult": 1.0,
                    "injury_flag": False,
                    "rot_on_court": True,
                },
            }
        ) + "\n",
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "game_id": "0000000003",
                "player_name": "Devin Booker",
                "pts": 34,
                "reb": 4,
                "ast": 6,
                "threes": 3,
                "stl": 1,
                "blk": 0,
                "tov": 2,
            }
        ]
    ).to_csv(processed / "recon_props_2026-03-31.csv", index=False)

    monkeypatch.setattr(app_module, "DATA_PROCESSED_DIR", processed)
    monkeypatch.setenv("NBA_LIVE_LENS_DIR", str(processed))
    app_module._load_recon_props_frame.cache_clear()
    app_module._ll_event_summary_actions.cache_clear()

    def _fake_summary(_event_id):
        return {
            "header": {
                "competitions": [
                    {
                        "competitors": [
                            {"team": {"id": "2", "abbreviation": "PHX"}},
                            {"team": {"id": "3", "abbreviation": "UTA"}},
                        ]
                    }
                ]
            },
            "plays": [
                {
                    "period": {"number": 4},
                    "clock": {"displayValue": "05:00"},
                    "homeScore": 102,
                    "awayScore": 100,
                    "team": {"id": "2"},
                    "shootingPlay": True,
                    "pointsAttempted": 2,
                    "scoreValue": 2,
                    "text": "Devin Booker made jumper",
                    "type": {"text": "Jump Shot"},
                }
            ],
        }

    monkeypatch.setattr(app_module, "_live_fetch_espn_summary", _fake_summary)

    with app_module.app.test_client() as client:
        resp = client.get("/api/live_player_props_projection_audit?date=2026-03-31&replay=signals&include_rows=1")

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["ok"] is True
    assert payload["overall"]["n"] == 1
    assert payload["debug"]["days"][0]["audit_source"] == "signals"

    row = payload["history"]["rows"][0]
    assert row["proj_original"] == 40.0
    assert row["sim_mu_adjusted"] < 36.0
    assert row["sim_mu_adjusted_original"] == 36.0
    assert row["proj"] < 40.0
    assert row["pregame_stat_multiplier_original"] == 1.2
    assert row["pregame_stat_multiplier"] < 1.2


def test_api_cron_live_lens_tick_requests_debug_player_rows(tmp_path, monkeypatch):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)

    monkeypatch.setattr(app_module, "DATA_PROCESSED_DIR", processed)
    monkeypatch.setenv("NBA_LIVE_LENS_DIR", str(processed))
    monkeypatch.setattr(app_module, "_cron_auth_ok", lambda _req: True)
    monkeypatch.setattr(app_module, "_admin_auth_ok", lambda _req: False)
    monkeypatch.setattr(
        app_module,
        "_live_build_scoreboard_games",
        lambda _ds: (
            "espn",
            [
                {
                    "espn_event_id": "401999999",
                    "game_id": "0000000003",
                    "home": "PHX",
                    "away": "UTA",
                    "in_progress": True,
                    "final": False,
                    "period": 4,
                    "clock": "05:00",
                    "home_pts": 102,
                    "away_pts": 100,
                }
            ],
        ),
    )

    app_module._live_lens_tick_last_logged.clear()
    app_module._live_lens_tick_last_proj_logged.clear()
    app_module._live_lens_tick_first_bet_logged.clear()

    requested_paths: list[str] = []

    class _FakeResponse:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        def get_json(self, silent=False):
            return self._payload

    class _FakeInnerClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, path):
            requested_paths.append(path)
            if path.startswith("/api/live_lens_tuning"):
                return _FakeResponse(200, {"markets": {"player_prop": {"watch": 2.0, "bet": 4.0}}})
            if path.startswith("/api/live_player_lens"):
                return _FakeResponse(
                    200,
                    {
                        "games": [
                            {
                                "event_id": "401999999",
                                "game_id": "0000000003",
                                "home": "PHX",
                                "away": "UTA",
                                "status": {"in_progress": True, "final": False, "period": 4, "clock": "05:00", "elapsed_min": 43.0},
                                "rows": [
                                    {
                                        "player": "Devin Booker",
                                        "team_tri": "PHX",
                                        "name_key": "DEVIN BOOKER",
                                        "stat": "pts",
                                        "line": 35.0,
                                        "pace_vs_line": 5.0,
                                        "pace_proj": 40.0,
                                        "sim_mu": 30.0,
                                        "sim_mu_adjusted": 36.0,
                                        "strength": 5.0,
                                        "sim_vs_line": -5.0,
                                        "sim_vs_line_adjusted": 1.0,
                                    }
                                ],
                            }
                        ]
                    },
                )
            raise AssertionError(f"unexpected internal path: {path}")

    monkeypatch.setattr(app_module.app, "test_client", lambda: _FakeInnerClient())

    with app_module.app.test_request_context(
        "/api/cron/live-lens-tick?date=2026-03-31&include_total=0&include_ats=0&include_ml=0&include_player_props=1&first_bet_only=0&max_props_per_game=1"
    ):
        resp = app_module.api_cron_live_lens_tick()

    payload = resp.get_json()
    assert payload["ok"] is True
    assert payload["wrote_signals"] == 1
    assert any(path.startswith("/api/live_player_lens") and "include_debug_rows=1" in path for path in requested_paths)


def test_load_recon_props_frame_backfills_from_live_lens_artifacts(tmp_path, monkeypatch):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)

    (processed / "live_lens_projections_2026-03-31.jsonl").write_text(
        json.dumps(
            {
                "date": "2026-03-31",
                "market": "player_prop",
                "game_id": "0000000003",
                "player": "Devin Booker",
                "name_key": "DEVIN BOOKER",
                "stat": "pts",
            }
        ) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(app_module, "DATA_PROCESSED_DIR", processed)
    monkeypatch.setenv("NBA_LIVE_LENS_DIR", str(processed))
    app_module._load_recon_props_frame.cache_clear()
    app_module._load_recon_props_lookup.cache_clear()

    import nba_betting.props_actuals as props_actuals_module

    monkeypatch.setattr(props_actuals_module, "fetch_prop_actuals_via_nba_cdn", lambda _date: pd.DataFrame())
    monkeypatch.setattr(props_actuals_module, "fetch_prop_actuals_via_nbaapi", lambda _date: pd.DataFrame())

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {
                "game": {
                    "homeTeam": {
                        "teamTricode": "PHX",
                        "players": [
                            {
                                "personId": 1626164,
                                "name": "Devin Booker",
                                "statistics": {
                                    "points": 27,
                                    "reboundsTotal": 4,
                                    "assists": 6,
                                    "threePointersMade": 3,
                                    "steals": 1,
                                    "blocks": 0,
                                    "turnovers": 2,
                                },
                            }
                        ],
                    },
                    "awayTeam": {"teamTricode": "UTA", "players": []},
                }
            }

    monkeypatch.setattr(app_module.requests, "get", lambda *args, **kwargs: _Resp())

    df, source_name = app_module._load_recon_props_frame("2026-03-31")

    assert source_name == "recon_props_2026-03-31.csv"
    assert len(df) == 1
    assert df.iloc[0]["player_name"] == "Devin Booker"
    assert (processed / "recon_props_2026-03-31.csv").exists()


def test_load_props_movement_callouts_exposes_player_id_photo_fallback(tmp_path, monkeypatch):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "player": "Jayson Tatum",
                "team": "Boston Celtics",
                "team_tricode": "BOS",
                "market": "pts",
                "side": "OVER",
                "line": 28.5,
                "price": -110,
                "open_line": 27.5,
                "open_price": -115,
                "line_move": 1.0,
                "implied_move": 0.03,
                "ev_pct": 2.4,
                "movement_tier": "fast",
            }
        ]
    ).to_csv(processed / "props_movement_signals_2026-03-15.csv", index=False)

    pd.DataFrame(
        [
            {
                "home_team": "Boston Celtics",
                "visitor_team": "Detroit Pistons",
            }
        ]
    ).to_csv(processed / "predictions_2026-03-15.csv", index=False)

    pd.DataFrame(
        [
            {
                "player_name": "Jayson Tatum",
                "team": "BOS",
                "player_id": 1628369,
                "pred_pts": 29.1,
            }
        ]
    ).to_csv(processed / "props_predictions_2026-03-15.csv", index=False)

    monkeypatch.setattr(app_module, "DATA_PROCESSED_DIR", processed)
    monkeypatch.setattr(app_module, "_maybe_fetch_remote_processed", lambda _name: None)

    items = app_module._load_props_movement_callouts(
        "2026-03-15",
        markets={"pts"},
        min_ev_pct=1.0,
        only_ev=True,
    )

    assert len(items) == 1
    assert items[0]["player"] == "Jayson Tatum"
    assert items[0]["player_id"] == 1628369
    assert items[0]["photo"] == "https://cdn.nba.com/headshots/nba/latest/1040x760/1628369.png"


def test_load_props_movement_callouts_rewrites_source_id_to_resolved_nba_headshot(tmp_path, monkeypatch):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "player": "Shai Gilgeous-Alexander",
                "team": "Oklahoma City Thunder",
                "team_tricode": "OKC",
                "market": "threes",
                "side": "UNDER",
                "line": 1.5,
                "price": 102,
                "open_line": 1.5,
                "open_price": -123,
                "line_move": 0.0,
                "implied_move": -0.05,
                "ev_pct": 4.8,
                "movement_tier": "fast",
                "player_id": 4278073,
                "photo": "https://cdn.nba.com/headshots/nba/latest/1040x760/4278073.png",
            }
        ]
    ).to_csv(processed / "props_movement_signals_2026-03-15.csv", index=False)

    pd.DataFrame(
        [
            {
                "home_team": "Oklahoma City Thunder",
                "visitor_team": "Minnesota Timberwolves",
            }
        ]
    ).to_csv(processed / "predictions_2026-03-15.csv", index=False)

    monkeypatch.setattr(app_module, "DATA_PROCESSED_DIR", processed)
    monkeypatch.setattr(app_module, "_maybe_fetch_remote_processed", lambda _name: None)
    monkeypatch.setattr(app_module, "_resolve_player_id", lambda player, team=None: 1628983 if player == "Shai Gilgeous-Alexander" else None)

    items = app_module._load_props_movement_callouts(
        "2026-03-15",
        markets={"threes"},
        min_ev_pct=1.0,
        only_ev=True,
    )

    assert len(items) == 1
    assert items[0]["player_id"] == 1628983
    assert items[0]["photo"] == "https://cdn.nba.com/headshots/nba/latest/1040x760/1628983.png"


def test_load_props_movement_callouts_falls_back_to_espn_headshot_for_source_id(tmp_path, monkeypatch):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "player": "Donovan Clingan",
                "team": "Portland Trail Blazers",
                "team_tricode": "POR",
                "market": "blk",
                "side": "UNDER",
                "line": 1.5,
                "price": 118,
                "open_line": 1.5,
                "open_price": -108,
                "line_move": 0.0,
                "implied_move": -0.06,
                "ev_pct": 3.2,
                "movement_tier": "fast",
                "player_id": 5105565,
            }
        ]
    ).to_csv(processed / "props_movement_signals_2026-03-15.csv", index=False)

    pd.DataFrame(
        [
            {
                "home_team": "Philadelphia 76ers",
                "visitor_team": "Portland Trail Blazers",
            }
        ]
    ).to_csv(processed / "predictions_2026-03-15.csv", index=False)

    monkeypatch.setattr(app_module, "DATA_PROCESSED_DIR", processed)
    monkeypatch.setattr(app_module, "_maybe_fetch_remote_processed", lambda _name: None)
    monkeypatch.setattr(app_module, "_resolve_player_id", lambda player, team=None: None)

    items = app_module._load_props_movement_callouts(
        "2026-03-15",
        markets={"blk"},
        min_ev_pct=1.0,
        only_ev=True,
    )

    assert len(items) == 1
    assert items[0]["player_id"] == 5105565
    assert items[0]["photo"] == "https://a.espncdn.com/i/headshots/nba/players/full/5105565.png"


def test_upload_props_refresh_artifacts_accepts_snapshot_only(tmp_path, monkeypatch):
    raw = tmp_path / "data" / "raw"
    processed = tmp_path / "data" / "processed"
    raw.mkdir(parents=True)
    processed.mkdir(parents=True)

    monkeypatch.setattr(app_module, "DATA_RAW_DIR", raw)
    monkeypatch.setattr(app_module, "DATA_PROCESSED_DIR", processed)
    monkeypatch.setattr(app_module, "_cron_auth_ok", lambda _req: True)
    monkeypatch.setattr(app_module, "_admin_auth_ok", lambda _req: True)

    client = app_module.app.test_client()
    response = client.post(
        "/api/cron/upload-props-refresh-artifacts",
        data={
            "date": "2026-03-18",
            "snapshot": (
                io.BytesIO(
                    (
                        "snapshot_ts,event_id,commence_time,bookmaker,market,outcome_name,player_name,point,price,home_team,away_team\n"
                        "2026-03-18T10:05:00Z,evt-1,2026-03-18T23:00:00Z,fanduel,player_points,Over,Jayson Tatum,27.5,-110,Boston Celtics,Miami Heat\n"
                        "2026-03-18T10:05:00Z,evt-1,2026-03-18T23:00:00Z,fanduel,player_points,Under,Jayson Tatum,27.5,-110,Boston Celtics,Miami Heat\n"
                    ).encode("utf-8")
                ),
                "odds_nba_player_props_2026-03-18.csv",
            ),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()

    assert payload["ok"] is True
    assert payload["upload_mode"] == "snapshot-only"
    assert payload["snapshot_rows"] == 2
    assert payload["predictions_rows"] == 0
    assert payload["edges_rows"] == 0
    assert payload["recs_rows"] == 0
    assert payload["opening_rows"] == 2
    assert payload["history_appended_rows"] == 2
    assert payload["uploaded_files"] == {
        "snapshot": True,
        "predictions": False,
        "edges": False,
        "recommendations": False,
    }
    assert (processed / "oddsapi_player_props_2026-03-18.csv").exists()
    assert (raw / "odds_nba_player_props_opening_2026-03-18.csv").exists()
    assert (raw / "odds_nba_player_props_history_2026-03-18.csv").exists()
    assert not (processed / "props_edges_2026-03-18.csv").exists()


def test_cards_shell_routes_use_single_main_page():
    client = app_module.app.test_client()

    root_response = client.get("/")
    pregame_response = client.get("/pregame")
    live_response = client.get("/live")
    betting_card_response = client.get("/betting-card")

    assert root_response.status_code == 200
    assert pregame_response.status_code == 404
    assert live_response.status_code == 302
    assert betting_card_response.status_code == 302

    root_html = root_response.get_data(as_text=True)

    assert 'NBA Game Cards' in root_html
    assert 'data-cards-base-path="/"' in root_html
    assert 'id="cardsPrevDateLink"' in root_html
    assert 'id="cardsNextDateLink"' in root_html
    assert 'id="cardsScoreboard"' in root_html
    assert 'id="cardsGrid"' in root_html
    assert live_response.headers["Location"].endswith("/")
    assert betting_card_response.headers["Location"].endswith("/")