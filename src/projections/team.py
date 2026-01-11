"""Aggregate player projections to team level.

Converts individual player projections to team runs scored/allowed
and applies Pythagorean expectation for win projections.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# League average runs per team per season (approximate)
LEAGUE_AVG_RUNS = 700


@dataclass
class TeamProjection:
    """Projected season outcomes for a team."""

    team: str
    division: str

    # Run projections
    projected_runs_scored: float
    projected_runs_allowed: float

    # Win projections
    projected_wins: float
    projected_losses: float

    # Confidence intervals
    wins_10th_percentile: float = 0.0
    wins_90th_percentile: float = 0.0

    # Component breakdowns
    batting_runs_above_avg: float = 0.0
    pitching_runs_above_avg: float = 0.0


# Division mappings
DIVISIONS = {
    "AL East": ["NYY", "BOS", "TOR", "TBR", "BAL"],
    "AL Central": ["CLE", "MIN", "CHW", "DET", "KCR"],
    "AL West": ["HOU", "TEX", "SEA", "LAA", "OAK"],
    "NL East": ["ATL", "PHI", "NYM", "MIA", "WSN"],
    "NL Central": ["MIL", "CHC", "STL", "PIT", "CIN"],
    "NL West": ["LAD", "SDP", "ARI", "SFG", "COL"],
}

TEAM_TO_DIVISION = {
    team: div for div, teams in DIVISIONS.items() for team in teams
}


def pythagorean_wins(
    runs_scored: float,
    runs_allowed: float,
    exponent: float = 1.83,
    games: int = 162,
) -> float:
    """Calculate expected wins using Pythagorean expectation.

    Formula: Win% = RS^exp / (RS^exp + RA^exp)

    Args:
        runs_scored: Team's projected runs scored.
        runs_allowed: Team's projected runs allowed.
        exponent: Pythagorean exponent (1.83 is standard).
        games: Number of games in season.

    Returns:
        Expected wins.
    """
    if runs_allowed <= 0:
        return float(games)
    if runs_scored <= 0:
        return 0.0

    rs_exp = runs_scored ** exponent
    ra_exp = runs_allowed ** exponent
    win_pct = rs_exp / (rs_exp + ra_exp)

    return win_pct * games


def project_team(
    team: str,
    batter_projections: pd.DataFrame,
    pitcher_projections: pd.DataFrame,
) -> TeamProjection:
    """Generate team projection from player projections.

    Args:
        team: Team abbreviation.
        batter_projections: DataFrame with batter projections for this team.
        pitcher_projections: DataFrame with pitcher projections for this team.

    Returns:
        TeamProjection with projected wins/losses.
    """
    # TODO: Sum batting runs contributions
    # TODO: Sum pitching runs contributions
    # TODO: Convert to team runs scored/allowed
    # TODO: Apply Pythagorean expectation
    raise NotImplementedError("Team projection not yet implemented")


def calculate_team_runs_scored(
    batters: pd.DataFrame,
    league_wrc_plus: float = 100.0,
    league_runs_per_pa: float = 0.11,
) -> float:
    """Calculate projected team runs scored from batter projections.

    Uses wRC+ and projected PA to estimate runs above/below average,
    then adds to league average baseline.

    Args:
        batters: DataFrame with columns: projected_pa, projected_wrc_plus
        league_wrc_plus: League average wRC+ (always 100).
        league_runs_per_pa: Average runs per PA (for conversion).

    Returns:
        Projected total runs scored.
    """
    total_pa = batters["projected_pa"].sum()
    if total_pa == 0:
        return LEAGUE_AVG_RUNS

    # Weight wRC+ by PA
    weighted_wrc_plus = (
        (batters["projected_wrc_plus"] * batters["projected_pa"]).sum()
        / total_pa
    )

    # Convert to runs
    # wRC+ of 100 = league average, each point = ~0.1% runs
    runs_multiplier = weighted_wrc_plus / 100.0
    projected_runs = LEAGUE_AVG_RUNS * runs_multiplier

    return projected_runs


def calculate_runs_from_wrc(
    batters: pd.DataFrame,
    wrc_to_runs: float = 1.0,
) -> float:
    """Calculate runs from wRC (weighted runs created).

    Args:
        batters: DataFrame with columns: projected_wrc (runs created)
        wrc_to_runs: Conversion factor (typically 1.0).

    Returns:
        Total runs scored.
    """
    if "projected_wrc" not in batters.columns:
        raise ValueError("batters DataFrame must have 'projected_wrc' column")

    return batters["projected_wrc"].sum() * wrc_to_runs
