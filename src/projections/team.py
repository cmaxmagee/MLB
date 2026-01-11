"""Aggregate player projections to team level.

Converts individual player projections to team runs scored/allowed
and applies Pythagorean expectation for win projections.
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .player import BatterProjection, PitcherProjection

logger = logging.getLogger(__name__)

# League average runs per team per season (approximate)
LEAGUE_AVG_RUNS = 700
GAMES_PER_SEASON = 162


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
    baserunning_runs: float = 0.0
    fielding_runs: float = 0.0

    # Roster strength indicators
    lineup_wrc_plus: float = 100.0
    rotation_fip: float = 4.00
    bullpen_fip: float = 4.00

    # Variance for simulation
    projected_wins_std: float = 7.0  # Historical std ~6-8 wins


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

TEAM_FULL_NAMES = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles", "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs", "CHW": "Chicago White Sox",
    "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies", "DET": "Detroit Tigers",
    "HOU": "Houston Astros", "KCR": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins", "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees", "OAK": "Oakland Athletics",
    "PHI": "Philadelphia Phillies", "PIT": "Pittsburgh Pirates",
    "SDP": "San Diego Padres", "SEA": "Seattle Mariners",
    "SFG": "San Francisco Giants", "STL": "St. Louis Cardinals",
    "TBR": "Tampa Bay Rays", "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays", "WSN": "Washington Nationals",
}


def pythagorean_wins(
    runs_scored: float,
    runs_allowed: float,
    exponent: float = 1.83,
    games: int = GAMES_PER_SEASON,
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


def pythag_exponent(runs_per_game: float) -> float:
    """Calculate optimal Pythagorean exponent based on run environment.

    Uses Pythagenpat formula: exponent = (runs_per_game) ^ 0.287

    Args:
        runs_per_game: Average runs per game in the league.

    Returns:
        Optimal exponent.
    """
    return runs_per_game ** 0.287


def project_team(
    team: str,
    batter_projections: List["BatterProjection"],
    pitcher_projections: List["PitcherProjection"],
) -> TeamProjection:
    """Generate team projection from player projections.

    Args:
        team: Team abbreviation.
        batter_projections: List of BatterProjection objects.
        pitcher_projections: List of PitcherProjection objects.

    Returns:
        TeamProjection with projected wins/losses.
    """
    division = TEAM_TO_DIVISION.get(team, "Unknown")

    # Calculate batting runs above average
    batting_runs = sum(b.projected_batting_runs for b in batter_projections)
    baserunning_runs = sum(b.projected_baserunning_runs for b in batter_projections)

    # Calculate pitching runs above average (positive = good for team)
    pitching_runs = sum(p.projected_pitching_runs for p in pitcher_projections)

    # Convert to team runs
    projected_runs_scored = LEAGUE_AVG_RUNS + batting_runs + baserunning_runs
    projected_runs_allowed = LEAGUE_AVG_RUNS - pitching_runs

    # Ensure reasonable bounds
    projected_runs_scored = max(500, min(950, projected_runs_scored))
    projected_runs_allowed = max(500, min(950, projected_runs_allowed))

    # Calculate wins
    projected_wins = pythagorean_wins(projected_runs_scored, projected_runs_allowed)
    projected_losses = GAMES_PER_SEASON - projected_wins

    # Calculate lineup strength (PA-weighted wRC+)
    total_pa = sum(b.projected_pa for b in batter_projections)
    if total_pa > 0:
        lineup_wrc_plus = sum(
            b.projected_wrc_plus * b.projected_pa for b in batter_projections
        ) / total_pa
    else:
        lineup_wrc_plus = 100.0

    # Calculate rotation/bullpen FIP
    starters = [p for p in pitcher_projections if p.role.startswith("SP")]
    relievers = [p for p in pitcher_projections if not p.role.startswith("SP")]

    starter_ip = sum(p.projected_ip for p in starters)
    if starter_ip > 0:
        rotation_fip = sum(
            p.projected_fip * p.projected_ip for p in starters
        ) / starter_ip
    else:
        rotation_fip = 4.00

    reliever_ip = sum(p.projected_ip for p in relievers)
    if reliever_ip > 0:
        bullpen_fip = sum(
            p.projected_fip * p.projected_ip for p in relievers
        ) / reliever_ip
    else:
        bullpen_fip = 4.00

    # Estimate variance based on roster depth and consistency
    # Teams with more high-confidence projections have lower variance
    if batter_projections:
        avg_confidence = np.mean([b.projection_confidence for b in batter_projections])
    else:
        avg_confidence = 0.5

    # Base std is ~7 wins, adjust based on projection confidence
    projected_wins_std = 7.0 + (1.0 - avg_confidence) * 3.0

    return TeamProjection(
        team=team,
        division=division,
        projected_runs_scored=projected_runs_scored,
        projected_runs_allowed=projected_runs_allowed,
        projected_wins=projected_wins,
        projected_losses=projected_losses,
        batting_runs_above_avg=batting_runs,
        pitching_runs_above_avg=pitching_runs,
        baserunning_runs=baserunning_runs,
        lineup_wrc_plus=lineup_wrc_plus,
        rotation_fip=rotation_fip,
        bullpen_fip=bullpen_fip,
        projected_wins_std=projected_wins_std,
    )


def project_team_from_dataframe(
    team: str,
    batters: pd.DataFrame,
    pitchers: pd.DataFrame,
) -> TeamProjection:
    """Generate team projection from DataFrames.

    Args:
        team: Team abbreviation.
        batters: DataFrame with columns: projected_pa, projected_wrc_plus,
                 projected_batting_runs
        pitchers: DataFrame with columns: projected_ip, projected_fip,
                  projected_pitching_runs, role

    Returns:
        TeamProjection with projected wins/losses.
    """
    division = TEAM_TO_DIVISION.get(team, "Unknown")

    # Sum runs contributions
    batting_runs = batters["projected_batting_runs"].sum() if len(batters) > 0 else 0
    pitching_runs = pitchers["projected_pitching_runs"].sum() if len(pitchers) > 0 else 0

    # Convert to team runs
    projected_runs_scored = LEAGUE_AVG_RUNS + batting_runs
    projected_runs_allowed = LEAGUE_AVG_RUNS - pitching_runs

    # Bounds check
    projected_runs_scored = max(500, min(950, projected_runs_scored))
    projected_runs_allowed = max(500, min(950, projected_runs_allowed))

    # Wins
    projected_wins = pythagorean_wins(projected_runs_scored, projected_runs_allowed)

    # Lineup wRC+
    total_pa = batters["projected_pa"].sum() if len(batters) > 0 else 0
    if total_pa > 0:
        lineup_wrc_plus = (
            batters["projected_wrc_plus"] * batters["projected_pa"]
        ).sum() / total_pa
    else:
        lineup_wrc_plus = 100.0

    return TeamProjection(
        team=team,
        division=division,
        projected_runs_scored=projected_runs_scored,
        projected_runs_allowed=projected_runs_allowed,
        projected_wins=projected_wins,
        projected_losses=GAMES_PER_SEASON - projected_wins,
        batting_runs_above_avg=batting_runs,
        pitching_runs_above_avg=pitching_runs,
        lineup_wrc_plus=lineup_wrc_plus,
    )


def calculate_team_runs_scored(
    batters: pd.DataFrame,
    league_wrc_plus: float = 100.0,
) -> float:
    """Calculate projected team runs scored from batter projections.

    Args:
        batters: DataFrame with columns: projected_pa, projected_wrc_plus
        league_wrc_plus: League average wRC+ (always 100).

    Returns:
        Projected total runs scored.
    """
    if len(batters) == 0 or batters["projected_pa"].sum() == 0:
        return LEAGUE_AVG_RUNS

    total_pa = batters["projected_pa"].sum()

    # Weight wRC+ by PA
    weighted_wrc_plus = (
        (batters["projected_wrc_plus"] * batters["projected_pa"]).sum()
        / total_pa
    )

    # Convert to runs: wRC+ of 100 = league average
    runs_multiplier = weighted_wrc_plus / league_wrc_plus
    projected_runs = LEAGUE_AVG_RUNS * runs_multiplier

    return projected_runs


def calculate_team_runs_allowed(
    pitchers: pd.DataFrame,
    league_fip: float = 4.00,
) -> float:
    """Calculate projected team runs allowed from pitcher projections.

    Args:
        pitchers: DataFrame with columns: projected_ip, projected_fip
        league_fip: League average FIP.

    Returns:
        Projected total runs allowed.
    """
    if len(pitchers) == 0 or pitchers["projected_ip"].sum() == 0:
        return LEAGUE_AVG_RUNS

    total_ip = pitchers["projected_ip"].sum()

    # Weight FIP by IP
    weighted_fip = (
        (pitchers["projected_fip"] * pitchers["projected_ip"]).sum()
        / total_ip
    )

    # Convert FIP to runs: RA = IP * FIP / 9
    # But we want total runs, so scale to full season
    expected_ip = 1458  # 162 * 9
    fip_ratio = weighted_fip / league_fip
    projected_runs = LEAGUE_AVG_RUNS * fip_ratio

    return projected_runs


def create_standings_dataframe(
    team_projections: Dict[str, TeamProjection],
) -> pd.DataFrame:
    """Create standings DataFrame from team projections.

    Args:
        team_projections: Dict mapping team abbreviation to TeamProjection.

    Returns:
        DataFrame with standings sorted by projected wins.
    """
    rows = []
    for team, proj in team_projections.items():
        rows.append({
            "Team": team,
            "Full Name": TEAM_FULL_NAMES.get(team, team),
            "Division": proj.division,
            "W": round(proj.projected_wins, 1),
            "L": round(proj.projected_losses, 1),
            "RS": round(proj.projected_runs_scored),
            "RA": round(proj.projected_runs_allowed),
            "Run Diff": round(proj.projected_runs_scored - proj.projected_runs_allowed),
            "wRC+": round(proj.lineup_wrc_plus, 1),
            "Rotation FIP": round(proj.rotation_fip, 2),
            "Bullpen FIP": round(proj.bullpen_fip, 2),
        })

    df = pd.DataFrame(rows)
    return df.sort_values("W", ascending=False).reset_index(drop=True)
