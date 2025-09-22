from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class EloConfig:
    k: float = 20.0
    home_adv: float = 60.0  # Elo points
    mov_multiplier: bool = True  # margin-of-victory multiplier (NBA typical)


class Elo:
    def __init__(self, config: EloConfig | None = None):
        self.config = config or EloConfig()
        self.ratings: Dict[str, float] = {}

    def get(self, team: str) -> float:
        return self.ratings.get(team, 1500.0)

    def expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def update_game(self, home: str, away: str, home_points: int, away_points: int) -> Tuple[float, float]:
        ra = self.get(home) + self.config.home_adv
        rb = self.get(away)
        sa = 1.0 if home_points > away_points else 0.0
        ea = self.expected(ra, rb)
        diff = abs(home_points - away_points)

        if self.config.mov_multiplier:
            # From FiveThirtyEight NBA Elo methodology
            mult = math.log(max(diff, 1) + 1) * (2.2 / (1 if sa == 1 else 2.2))
        else:
            mult = 1.0

        delta = self.config.k * mult * (sa - ea)
        # remove home adv when storing
        new_home = self.get(home) + delta
        new_away = self.get(away) - delta
        self.ratings[home] = new_home
        self.ratings[away] = new_away
        return new_home, new_away
