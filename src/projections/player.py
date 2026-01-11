"""Individual player projection module.

Projects player performance based on weighted historical averages,
regression to the mean, and aging curves.
"""

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PlayerProjection:
    """Projected stats for a single player."""

    player_id: int
    name: str
    team: str
    position: Literal["Batter", "Pitcher"]

    # Projected playing time
    projected_pa: float = 0.0  # Plate appearances (batters)
    projected_ip: float = 0.0  # Innings pitched (pitchers)

    # Projected rate stats (batters)
    projected_wrc_plus: float = 100.0  # wRC+ (100 = league average)
    projected_obp: float = 0.320
    projected_slg: float = 0.400

    # Projected rate stats (pitchers)
    projected_fip: float = 4.00
    projected_k_per_9: float = 8.0
    projected_bb_per_9: float = 3.0

    # Projected runs contribution
    projected_batting_runs: float = 0.0
    projected_pitching_runs: float = 0.0

    # Uncertainty (standard deviation)
    uncertainty: float = 0.0


def project_player(
    player_id: int,
    historical_stats: pd.DataFrame,
    projection_year: int,
    weights: list[float] | None = None,
) -> PlayerProjection:
    """Generate projection for a single player.

    Args:
        player_id: FanGraphs player ID.
        historical_stats: DataFrame with player's historical stats by year.
        projection_year: Year to project for.
        weights: Weights for recent seasons [most_recent, ..., oldest].
                 Defaults to [5, 4, 3] for 3 most recent years.

    Returns:
        PlayerProjection with projected stats.
    """
    # TODO: Implement weighted averaging
    # TODO: Apply regression to the mean
    # TODO: Apply aging curve adjustment
    raise NotImplementedError("Player projection not yet implemented")


def calculate_weighted_average(
    stats: pd.DataFrame,
    stat_col: str,
    weight_col: str = "PA",
    year_weights: list[float] | None = None,
) -> float:
    """Calculate weighted average of a stat across seasons.

    Args:
        stats: DataFrame with yearly stats.
        stat_col: Column name of the stat to average.
        weight_col: Column to use for weighting (PA or IP).
        year_weights: Recency weights [most_recent, ..., oldest].

    Returns:
        Weighted average value.
    """
    if year_weights is None:
        year_weights = [5, 4, 3]  # Default: weight recent years more

    # Sort by year descending
    stats = stats.sort_values("Season", ascending=False).head(len(year_weights))

    if len(stats) == 0:
        return np.nan

    # Apply year weights and playing time weights
    total_weight = 0.0
    weighted_sum = 0.0

    for i, (_, row) in enumerate(stats.iterrows()):
        if i >= len(year_weights):
            break
        year_weight = year_weights[i]
        playing_time = row.get(weight_col, 1)
        combined_weight = year_weight * playing_time
        weighted_sum += row[stat_col] * combined_weight
        total_weight += combined_weight

    return weighted_sum / total_weight if total_weight > 0 else np.nan


def regress_to_mean(
    observed_value: float,
    sample_size: int,
    league_mean: float,
    regression_constant: int,
) -> float:
    """Apply regression to the mean for a stat.

    Uses the formula: regressed = (observed * n + mean * k) / (n + k)
    where k is the regression constant (sample size at which we trust
    the observed value 50%).

    Args:
        observed_value: Player's observed rate stat.
        sample_size: Number of PA/IP/etc.
        league_mean: League average for this stat.
        regression_constant: Sample size for 50% regression.

    Returns:
        Regressed value.
    """
    if sample_size <= 0:
        return league_mean

    weight = sample_size / (sample_size + regression_constant)
    return (observed_value * weight) + (league_mean * (1 - weight))


# Regression constants for common stats (approximate PA/IP needed for reliability)
REGRESSION_CONSTANTS = {
    # Batting
    "AVG": 910,
    "OBP": 460,
    "SLG": 320,
    "ISO": 160,
    "BABIP": 820,
    "K%": 60,
    "BB%": 120,
    "wRC+": 400,
    # Pitching
    "ERA": 600,  # IP-based
    "FIP": 200,
    "K/9": 70,
    "BB/9": 170,
    "HR/9": 300,
    "BABIP_P": 2000,
    "LOB%": 600,
}
