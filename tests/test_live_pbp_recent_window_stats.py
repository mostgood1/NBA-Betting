from __future__ import annotations

import app as app_module


def test_live_pbp_recent_window_stats_reports_scoring_run_and_drought():
    actions = [
        {
            "period": 1,
            "clock": "11:30",
            "scoreHome": 2,
            "scoreAway": 0,
            "teamTricode": "BOS",
            "actionType": "Jump Shot",
            "isFieldGoal": 1,
            "shotValue": 2,
            "shotResult": "Made",
        },
        {
            "period": 1,
            "clock": "11:10",
            "scoreHome": 2,
            "scoreAway": 3,
            "teamTricode": "NYK",
            "actionType": "Jump Shot",
            "isFieldGoal": 1,
            "shotValue": 3,
            "shotResult": "Made",
        },
        {
            "period": 1,
            "clock": "10:55",
            "scoreHome": 2,
            "scoreAway": 5,
            "teamTricode": "NYK",
            "actionType": "Layup",
            "isFieldGoal": 1,
            "shotValue": 2,
            "shotResult": "Made",
        },
        {
            "period": 1,
            "clock": "10:20",
            "scoreHome": 2,
            "scoreAway": 7,
            "teamTricode": "NYK",
            "actionType": "Jump Shot",
            "isFieldGoal": 1,
            "shotValue": 2,
            "shotResult": "Made",
        },
        {
            "period": 1,
            "clock": "09:50",
            "scoreHome": 2,
            "scoreAway": 7,
            "teamTricode": "BOS",
            "actionType": "Turnover",
            "description": "Bad pass turnover",
        },
    ]

    stats = app_module._live_pbp_recent_window_stats(actions, window_sec=120)

    assert stats["current_scoring_run"]["team"] == "NYK"
    assert stats["current_scoring_run"]["points"] == 7
    assert stats["seconds_since_score"] == 30
    assert stats["points_total"] == 7