"""Playing time (PA/IP) projections by role.

Projects plate appearances for batters and innings pitched for pitchers
based on historical patterns, role expectations, and age-based injury risk.
"""

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PlayingTimeProjection:
    """Projected playing time for a player."""

    player_id: int
    name: str
    team: str
    role: str  # e.g., "Starter", "Bench", "SP", "RP", "CL"

    # Projections
    projected_pa: float = 0.0  # For batters
    projected_ip: float = 0.0  # For pitchers
    projected_games: float = 0.0

    # Confidence interval
    low_pa: float = 0.0
    high_pa: float = 0.0
    low_ip: float = 0.0
    high_ip: float = 0.0


# Typical playing time by role
ROLE_BASELINES = {
    # Batters
    "Everyday": {"pa": 600, "games": 150},
    "Platoon": {"pa": 350, "games": 110},
    "Bench": {"pa": 180, "games": 80},
    "Utility": {"pa": 280, "games": 100},
    # Pitchers
    "SP1": {"ip": 190, "games": 32},
    "SP2": {"ip": 175, "games": 31},
    "SP3": {"ip": 165, "games": 30},
    "SP4": {"ip": 150, "games": 28},
    "SP5": {"ip": 130, "games": 26},
    "RP_High": {"ip": 70, "games": 70},
    "RP_Mid": {"ip": 55, "games": 60},
    "RP_Low": {"ip": 40, "games": 45},
    "CL": {"ip": 60, "games": 60},
}


def project_playing_time(
    player_id: int,
    historical_stats: pd.DataFrame,
    role: str,
    age: int,
) -> PlayingTimeProjection:
    """Project playing time for a player.

    Args:
        player_id: FanGraphs player ID.
        historical_stats: DataFrame with player's historical PA/IP by year.
        role: Expected role for projection year.
        age: Player's age for projection year.

    Returns:
        PlayingTimeProjection with projected PA or IP.
    """
    # TODO: Implement based on historical patterns
    # TODO: Apply age-based injury risk adjustment
    raise NotImplementedError("Playing time projection not yet implemented")


def estimate_role(
    historical_stats: pd.DataFrame,
    player_type: Literal["batter", "pitcher"],
) -> str:
    """Estimate a player's role based on historical usage.

    Args:
        historical_stats: DataFrame with player's historical stats.
        player_type: "batter" or "pitcher".

    Returns:
        Role string (e.g., "Everyday", "SP1", "RP_High").
    """
    if player_type == "batter":
        avg_pa = historical_stats["PA"].mean()
        if avg_pa >= 550:
            return "Everyday"
        elif avg_pa >= 350:
            return "Platoon"
        elif avg_pa >= 200:
            return "Utility"
        else:
            return "Bench"
    else:
        # Pitcher
        avg_ip = historical_stats["IP"].mean()
        avg_gs = historical_stats.get("GS", pd.Series([0])).mean()

        if avg_gs >= 20:
            # Starter
            if avg_ip >= 180:
                return "SP1"
            elif avg_ip >= 160:
                return "SP2"
            elif avg_ip >= 140:
                return "SP3"
            elif avg_ip >= 120:
                return "SP4"
            else:
                return "SP5"
        else:
            # Reliever
            avg_sv = historical_stats.get("SV", pd.Series([0])).mean()
            if avg_sv >= 20:
                return "CL"
            elif avg_ip >= 60:
                return "RP_High"
            elif avg_ip >= 45:
                return "RP_Mid"
            else:
                return "RP_Low"


def apply_age_adjustment(
    baseline_playing_time: float,
    age: int,
    player_type: Literal["batter", "pitcher"],
) -> float:
    """Adjust projected playing time based on age/injury risk.

    Older players have higher injury risk and may see reduced playing time.

    Args:
        baseline_playing_time: Baseline PA or IP projection.
        age: Player's age.
        player_type: "batter" or "pitcher".

    Returns:
        Adjusted playing time projection.
    """
    # Age-based playing time multipliers (rough estimates)
    # Based on injury frequency by age
    if player_type == "batter":
        age_factors = {
            21: 0.90, 22: 0.93, 23: 0.96, 24: 0.98, 25: 1.00,
            26: 1.00, 27: 1.00, 28: 1.00, 29: 0.98, 30: 0.96,
            31: 0.94, 32: 0.92, 33: 0.90, 34: 0.87, 35: 0.84,
            36: 0.80, 37: 0.75, 38: 0.70, 39: 0.65, 40: 0.60,
        }
    else:
        # Pitchers have steeper decline
        age_factors = {
            21: 0.85, 22: 0.90, 23: 0.94, 24: 0.97, 25: 1.00,
            26: 1.00, 27: 1.00, 28: 0.98, 29: 0.95, 30: 0.92,
            31: 0.88, 32: 0.84, 33: 0.80, 34: 0.75, 35: 0.70,
            36: 0.65, 37: 0.58, 38: 0.50, 39: 0.42, 40: 0.35,
        }

    factor = age_factors.get(age, 0.50 if age > 40 else 0.85)
    return baseline_playing_time * factor
