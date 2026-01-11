"""Aging curves, park factors, and regression adjustments.

Provides adjustment factors for player projections based on:
- Age-based performance curves
- Park factors for hitters changing teams
- Regression to the mean for small samples
"""

import logging
from typing import List, Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Aging Curves
# ============================================================================

def get_aging_adjustment(
    current_age: int,
    projection_age: int,
    player_type: Literal["batter", "pitcher"],
    stat_type: Literal["rate", "counting"] = "rate",
) -> float:
    """Get aging curve adjustment factor.

    Based on research showing peak age ~27 for batters, ~26 for pitchers,
    with gradual decline afterward.

    Args:
        current_age: Player's age in the baseline year.
        projection_age: Player's age in the projection year.
        player_type: "batter" or "pitcher".
        stat_type: "rate" for rate stats, "counting" for counting stats.

    Returns:
        Multiplier to apply to the stat (e.g., 0.98 for slight decline).
    """
    age_diff = projection_age - current_age

    if player_type == "batter":
        # Batter aging curve (wRC+ based)
        # Peak at 27, ~0.5% decline per year after 30
        aging_curve = {
            20: 0.92, 21: 0.94, 22: 0.96, 23: 0.98, 24: 0.99,
            25: 1.00, 26: 1.01, 27: 1.01, 28: 1.00, 29: 0.99,
            30: 0.98, 31: 0.97, 32: 0.95, 33: 0.93, 34: 0.91,
            35: 0.88, 36: 0.85, 37: 0.82, 38: 0.78, 39: 0.74,
            40: 0.70, 41: 0.66, 42: 0.62,
        }
    else:
        # Pitcher aging curve (FIP based, inverted so lower = better)
        # Peak at 26, steeper decline after 32
        aging_curve = {
            20: 1.08, 21: 1.05, 22: 1.03, 23: 1.01, 24: 1.00,
            25: 0.99, 26: 0.99, 27: 1.00, 28: 1.01, 29: 1.02,
            30: 1.03, 31: 1.05, 32: 1.07, 33: 1.10, 34: 1.13,
            35: 1.17, 36: 1.22, 37: 1.28, 38: 1.35, 39: 1.43,
            40: 1.52, 41: 1.62, 42: 1.73,
        }

    current_factor = aging_curve.get(current_age, 1.0)
    projection_factor = aging_curve.get(projection_age, 1.0)

    # Return the relative change
    if player_type == "batter":
        return projection_factor / current_factor
    else:
        # For pitchers, we want the inverse (lower FIP is better)
        return projection_factor / current_factor


# ============================================================================
# Park Factors
# ============================================================================

# FanGraphs-style park factors (100 = neutral)
# Values > 100 favor hitters, < 100 favor pitchers
PARK_FACTORS = {
    # Hitter-friendly
    "COL": 115,  # Coors Field
    "CIN": 106,  # Great American
    "TEX": 105,  # Globe Life
    "ARI": 104,  # Chase Field
    "CHC": 103,  # Wrigley
    "PHI": 102,  # Citizens Bank
    "BAL": 102,  # Camden Yards

    # Neutral
    "MIL": 101,
    "ATL": 100,
    "BOS": 100,  # Fenway (complex park)
    "STL": 100,
    "MIN": 100,
    "CHW": 100,
    "HOU": 99,
    "KCR": 99,
    "DET": 99,
    "TOR": 99,

    # Pitcher-friendly
    "LAD": 98,
    "NYY": 98,
    "WSN": 98,
    "LAA": 97,
    "PIT": 97,
    "SDP": 96,
    "CLE": 96,
    "TBR": 96,
    "NYM": 95,
    "SFG": 95,
    "SEA": 94,
    "MIA": 93,
    "OAK": 93,
}


def get_park_factor(team: str) -> int:
    """Get park factor for a team's home stadium.

    Args:
        team: Team abbreviation.

    Returns:
        Park factor (100 = neutral).
    """
    return PARK_FACTORS.get(team, 100)


def adjust_for_park_change(
    stat_value: float,
    old_team: str,
    new_team: str,
    stat_type: Literal["batting", "pitching"],
) -> float:
    """Adjust a stat for a player changing teams (park factors).

    Args:
        stat_value: Original stat value.
        old_team: Team the player is leaving.
        new_team: Team the player is joining.
        stat_type: "batting" or "pitching".

    Returns:
        Adjusted stat value.
    """
    old_pf = get_park_factor(old_team)
    new_pf = get_park_factor(new_team)

    if old_pf == new_pf:
        return stat_value

    # Park factors affect ~50% of a player's games (home games)
    # So the adjustment is halved
    pf_ratio = new_pf / old_pf
    adjustment = 1 + (pf_ratio - 1) * 0.5

    if stat_type == "batting":
        # Higher park factor = more runs = higher stats
        return stat_value * adjustment
    else:
        # For pitchers (ERA/FIP), higher park factor = worse stats
        return stat_value * adjustment


def neutralize_stats(
    stats: pd.DataFrame,
    team_col: str = "Team",
    stat_cols: Optional[List[str]] = None,
    stat_type: Literal["batting", "pitching"] = "batting",
) -> pd.DataFrame:
    """Neutralize stats to a neutral park environment.

    Args:
        stats: DataFrame with player stats.
        team_col: Column containing team abbreviation.
        stat_cols: Columns to adjust. If None, adjusts common rate stats.
        stat_type: "batting" or "pitching".

    Returns:
        DataFrame with park-neutralized stats.
    """
    if stat_cols is None:
        if stat_type == "batting":
            stat_cols = ["AVG", "OBP", "SLG", "wOBA", "ISO"]
        else:
            stat_cols = ["ERA", "FIP", "xFIP"]

    result = stats.copy()

    for col in stat_cols:
        if col not in result.columns:
            continue

        # Neutralize to park factor of 100
        for team in result[team_col].unique():
            mask = result[team_col] == team
            pf = get_park_factor(team)
            adjustment = 100 / pf

            if stat_type == "pitching":
                # For pitchers, lower is better, so inverse adjustment
                result.loc[mask, col] = result.loc[mask, col] * adjustment
            else:
                result.loc[mask, col] = result.loc[mask, col] * adjustment

    return result


# ============================================================================
# League Constants
# ============================================================================

# 2024 approximate league averages
LEAGUE_AVERAGES = {
    # Batting
    "AVG": 0.248,
    "OBP": 0.315,
    "SLG": 0.399,
    "wOBA": 0.310,
    "wRC+": 100,
    "ISO": 0.151,
    "BABIP": 0.296,
    "K%": 0.227,
    "BB%": 0.085,

    # Pitching
    "ERA": 4.17,
    "FIP": 4.12,
    "xFIP": 4.15,
    "K/9": 8.6,
    "BB/9": 3.2,
    "HR/9": 1.22,
    "WHIP": 1.28,
    "BABIP_P": 0.290,
    "LOB%": 0.72,
}


def get_league_average(stat: str) -> float:
    """Get league average for a stat.

    Args:
        stat: Stat name.

    Returns:
        League average value.
    """
    return LEAGUE_AVERAGES.get(stat, 0.0)
