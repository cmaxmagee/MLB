"""Compute standings and playoff odds from simulations."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .monte_carlo import SeasonSimulation, SimulationResult

logger = logging.getLogger(__name__)


@dataclass
class DivisionStandings:
    """Projected standings for a division."""

    division: str
    teams: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        # Sort by projected wins descending
        self.teams = sorted(self.teams, key=lambda x: x["mean_wins"], reverse=True)


@dataclass
class LeagueStandings:
    """Projected standings for a league."""

    league: str  # "AL" or "NL"
    divisions: Dict[str, DivisionStandings] = field(default_factory=dict)
    wild_card_race: List[Dict] = field(default_factory=list)


def generate_standings(simulation: SeasonSimulation) -> Dict[str, LeagueStandings]:
    """Generate projected standings from simulation results.

    Args:
        simulation: Completed season simulation.

    Returns:
        Dictionary with "AL" and "NL" LeagueStandings.
    """
    from ..projections.team import TEAM_TO_DIVISION

    # Group results by division
    divisions = {}
    for team, result in simulation.team_results.items():
        div = TEAM_TO_DIVISION.get(team, "Unknown")
        if div not in divisions:
            divisions[div] = []
        divisions[div].append({
            "team": team,
            "mean_wins": result.mean_wins,
            "wins_10th_pct": result.wins_10th_pct,
            "wins_90th_pct": result.wins_90th_pct,
            "division_winner_pct": result.division_winner_pct,
            "playoff_pct": result.playoff_pct,
        })

    # Create standings objects
    standings = {
        "AL": LeagueStandings(league="AL"),
        "NL": LeagueStandings(league="NL"),
    }

    for div, teams in divisions.items():
        league = "AL" if "AL" in div else "NL"
        standings[league].divisions[div] = DivisionStandings(
            division=div,
            teams=teams,
        )

    # Generate wild card races
    for league in ["AL", "NL"]:
        all_teams = []
        for div_standings in standings[league].divisions.values():
            # Exclude likely division winners for wild card race
            for i, team in enumerate(div_standings.teams):
                if i > 0:  # Skip division leader
                    all_teams.append(team)

        # Sort by playoff odds
        all_teams.sort(key=lambda x: x["playoff_pct"], reverse=True)
        standings[league].wild_card_race = all_teams[:6]

    return standings


def standings_to_dataframe(standings: Dict[str, LeagueStandings]) -> pd.DataFrame:
    """Convert standings to a flat DataFrame.

    Args:
        standings: League standings from generate_standings.

    Returns:
        DataFrame with all teams and their projections.
    """
    rows = []
    for league, league_standings in standings.items():
        for div, div_standings in league_standings.divisions.items():
            for rank, team in enumerate(div_standings.teams, 1):
                rows.append({
                    "League": league,
                    "Division": div,
                    "Rank": rank,
                    "Team": team["team"],
                    "Projected Wins": round(team["mean_wins"], 1),
                    "10th Pct": round(team["wins_10th_pct"], 1),
                    "90th Pct": round(team["wins_90th_pct"], 1),
                    "Division %": round(team["division_winner_pct"], 1),
                    "Playoff %": round(team["playoff_pct"], 1),
                })

    df = pd.DataFrame(rows)
    return df.sort_values(["League", "Division", "Rank"])


def calculate_head_to_head_odds(
    team1: str,
    team2: str,
    simulation: SeasonSimulation,
) -> dict:
    """Calculate odds of one team finishing ahead of another.

    Args:
        team1: First team abbreviation.
        team2: Second team abbreviation.
        simulation: Completed simulation.

    Returns:
        Dictionary with comparison results.
    """
    if team1 not in simulation.team_results or team2 not in simulation.team_results:
        raise ValueError(f"Team not found in simulation results")

    wins1 = simulation.team_results[team1].win_distribution
    wins2 = simulation.team_results[team2].win_distribution

    team1_better = np.sum(wins1 > wins2)
    team2_better = np.sum(wins2 > wins1)
    tie = np.sum(wins1 == wins2)

    total = len(wins1)

    return {
        "team1": team1,
        "team2": team2,
        "team1_wins_more_pct": team1_better / total * 100,
        "team2_wins_more_pct": team2_better / total * 100,
        "tie_pct": tie / total * 100,
        "avg_difference": float(np.mean(wins1 - wins2)),
    }


def calculate_wins_over_threshold(
    team: str,
    threshold: int,
    simulation: SeasonSimulation,
) -> float:
    """Calculate probability of a team winning at least N games.

    Args:
        team: Team abbreviation.
        threshold: Win threshold.
        simulation: Completed simulation.

    Returns:
        Probability (0-100) of reaching threshold.
    """
    if team not in simulation.team_results:
        raise ValueError(f"Team {team} not found in simulation results")

    wins = simulation.team_results[team].win_distribution
    return float(np.sum(wins >= threshold) / len(wins) * 100)
