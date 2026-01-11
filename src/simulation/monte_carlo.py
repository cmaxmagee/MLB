"""Monte Carlo simulation for season outcomes.

Runs thousands of simulated seasons to generate probability distributions
for wins, division standings, and playoff odds.
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..projections.team import pythagorean_wins, DIVISIONS, TEAM_TO_DIVISION

if TYPE_CHECKING:
    from ..projections.player import BatterProjection, PitcherProjection
    from ..projections.projector import TeamProjectionSet

logger = logging.getLogger(__name__)

# Default simulation parameters
DEFAULT_ITERATIONS = 10_000
GAMES_PER_SEASON = 162

# Historical standard deviation of team wins around projection
# Teams typically deviate 6-8 wins from their "true" talent level
TEAM_WIN_STDEV = 7.0

# Player variance scaling (how much individual performance varies year-to-year)
BATTER_WRC_PLUS_STDEV = 15.0  # wRC+ standard deviation
PITCHER_FIP_STDEV = 0.50  # FIP standard deviation


@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation for one team."""

    team: str
    iterations: int

    # Win distribution
    mean_wins: float
    median_wins: float
    std_wins: float
    wins_10th_pct: float
    wins_25th_pct: float
    wins_75th_pct: float
    wins_90th_pct: float

    # Playoff odds
    division_winner_pct: float = 0.0
    wild_card_pct: float = 0.0
    playoff_pct: float = 0.0
    world_series_pct: float = 0.0

    # Additional metrics
    wins_over_500_pct: float = 0.0
    wins_90_plus_pct: float = 0.0
    wins_100_plus_pct: float = 0.0
    last_place_pct: float = 0.0

    # Full distribution for analysis
    win_distribution: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class SeasonSimulation:
    """Complete season simulation results."""

    iterations: int
    random_seed: int | None = None
    team_results: dict[str, SimulationResult] = field(default_factory=dict)
    division_winner_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    wild_card_counts: dict[str, int] = field(default_factory=dict)

    def get_playoff_odds_df(self) -> pd.DataFrame:
        """Get playoff odds as a sorted DataFrame."""
        rows = []
        for team, result in self.team_results.items():
            rows.append({
                "Team": team,
                "Proj W": round(result.mean_wins, 1),
                "10th": round(result.wins_10th_pct),
                "90th": round(result.wins_90th_pct),
                "Div %": round(result.division_winner_pct, 1),
                "WC %": round(result.wild_card_pct, 1),
                "Playoff %": round(result.playoff_pct, 1),
            })
        df = pd.DataFrame(rows)
        return df.sort_values("Playoff %", ascending=False).reset_index(drop=True)


def run_simulation(
    team_projections: pd.DataFrame | dict[str, "TeamProjectionSet"],
    iterations: int = DEFAULT_ITERATIONS,
    team_variance: float = TEAM_WIN_STDEV,
    random_seed: int | None = None,
    show_progress: bool = True,
) -> SeasonSimulation:
    """Run Monte Carlo simulation of the season.

    Args:
        team_projections: DataFrame with columns (team, division,
            projected_runs_scored, projected_runs_allowed) OR dict of
            TeamProjectionSet objects.
        iterations: Number of seasons to simulate.
        team_variance: Standard deviation of team wins (default 7).
        random_seed: Random seed for reproducibility.
        show_progress: Whether to show progress bar.

    Returns:
        SeasonSimulation with full results.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Handle both DataFrame and dict input
    if isinstance(team_projections, dict):
        # Convert TeamProjectionSet dict to DataFrame
        rows = []
        for team, proj_set in team_projections.items():
            rows.append({
                "team": team,
                "division": proj_set.division,
                "projected_runs_scored": proj_set.projected_runs_scored,
                "projected_runs_allowed": proj_set.projected_runs_allowed,
                "projected_wins": proj_set.projected_wins,
            })
        team_projections = pd.DataFrame(rows)

    teams = team_projections["team"].tolist()

    # Pre-calculate base win projections
    base_wins = {}
    team_std = {}
    for _, row in team_projections.iterrows():
        if "projected_wins" in row and pd.notna(row["projected_wins"]):
            wins = row["projected_wins"]
        else:
            wins = pythagorean_wins(
                row["projected_runs_scored"],
                row["projected_runs_allowed"],
            )
        base_wins[row["team"]] = wins
        # Use team-specific variance if available
        team_std[row["team"]] = row.get("projected_wins_std", team_variance)

    # Simulate seasons - vectorized for speed
    logger.info(f"Running {iterations:,} season simulations...")

    # Generate all random noise at once
    noise = {
        team: np.random.normal(0, team_std[team], iterations)
        for team in teams
    }

    # Calculate simulated wins for all teams
    simulated_wins = {
        team: np.clip(base_wins[team] + noise[team], 0, GAMES_PER_SEASON)
        for team in teams
    }

    # Calculate results
    simulation = SeasonSimulation(iterations=iterations, random_seed=random_seed)

    for team in teams:
        wins = simulated_wins[team]
        simulation.team_results[team] = SimulationResult(
            team=team,
            iterations=iterations,
            mean_wins=float(np.mean(wins)),
            median_wins=float(np.median(wins)),
            std_wins=float(np.std(wins)),
            wins_10th_pct=float(np.percentile(wins, 10)),
            wins_25th_pct=float(np.percentile(wins, 25)),
            wins_75th_pct=float(np.percentile(wins, 75)),
            wins_90th_pct=float(np.percentile(wins, 90)),
            wins_over_500_pct=float(np.mean(wins >= 81) * 100),
            wins_90_plus_pct=float(np.mean(wins >= 90) * 100),
            wins_100_plus_pct=float(np.mean(wins >= 100) * 100),
            win_distribution=wins,
        )

    # Calculate playoff odds
    _calculate_playoff_odds(simulation, team_projections, simulated_wins, iterations)

    return simulation


def run_simulation_with_player_variance(
    team_projection_sets: dict[str, "TeamProjectionSet"],
    iterations: int = DEFAULT_ITERATIONS,
    team_residual_std: float = 4.0,
    random_seed: int | None = None,
    show_progress: bool = True,
) -> SeasonSimulation:
    """Run simulation with player-level variance modeling.

    This is a more sophisticated simulation that models individual
    player performance variance and its impact on team outcomes.

    Two variance components:
    1. Player-level: Each player's stats vary around their projection
    2. Team-level residual: Unexplained team variance (~4-5 wins)

    Args:
        team_projection_sets: Dict mapping team to TeamProjectionSet.
        iterations: Number of simulations.
        team_residual_std: Residual team-level std after player variance.
        random_seed: Random seed for reproducibility.
        show_progress: Whether to show progress bar.

    Returns:
        SeasonSimulation with results.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    teams = list(team_projection_sets.keys())
    logger.info(f"Running {iterations:,} simulations with player variance...")

    # Pre-allocate results arrays
    simulated_wins = {team: np.zeros(iterations) for team in teams}

    # For each simulation, sample player performances and recalculate team wins
    iterator = range(iterations)
    if show_progress:
        iterator = tqdm(iterator, desc="Simulating with player variance")

    for i in iterator:
        for team in teams:
            proj_set = team_projection_sets[team]

            # Sample batter performances
            sampled_batting_runs = 0.0
            for batter in proj_set.batters:
                # wRC+ varies with std ~15 points
                # Confidence reduces variance
                batter_std = BATTER_WRC_PLUS_STDEV * (2.0 - batter.projection_confidence)
                sampled_wrc_plus = np.random.normal(
                    batter.projected_wrc_plus, batter_std
                )
                # Convert sampled wRC+ to batting runs
                sampled_runs = (sampled_wrc_plus - 100) / 100 * batter.projected_pa * 0.12
                sampled_batting_runs += sampled_runs

            # Sample pitcher performances
            sampled_pitching_runs = 0.0
            for pitcher in proj_set.pitchers:
                # FIP varies with std ~0.5
                pitcher_std = PITCHER_FIP_STDEV * (2.0 - pitcher.projection_confidence)
                sampled_fip = np.random.normal(pitcher.projected_fip, pitcher_std)
                # Convert sampled FIP to pitching runs (negative FIP diff = good)
                league_fip = 4.00
                sampled_runs = -((sampled_fip - league_fip) / 9 * pitcher.projected_ip)
                sampled_pitching_runs += sampled_runs

            # Calculate team runs
            league_avg = 700
            runs_scored = league_avg + sampled_batting_runs
            runs_allowed = league_avg - sampled_pitching_runs

            # Bounds
            runs_scored = max(500, min(950, runs_scored))
            runs_allowed = max(500, min(950, runs_allowed))

            # Calculate Pythagorean wins
            base_wins = pythagorean_wins(runs_scored, runs_allowed)

            # Add team-level residual noise
            team_noise = np.random.normal(0, team_residual_std)
            final_wins = np.clip(base_wins + team_noise, 0, GAMES_PER_SEASON)

            simulated_wins[team][i] = final_wins

    # Calculate results
    simulation = SeasonSimulation(iterations=iterations, random_seed=random_seed)

    for team in teams:
        wins = simulated_wins[team]
        simulation.team_results[team] = SimulationResult(
            team=team,
            iterations=iterations,
            mean_wins=float(np.mean(wins)),
            median_wins=float(np.median(wins)),
            std_wins=float(np.std(wins)),
            wins_10th_pct=float(np.percentile(wins, 10)),
            wins_25th_pct=float(np.percentile(wins, 25)),
            wins_75th_pct=float(np.percentile(wins, 75)),
            wins_90th_pct=float(np.percentile(wins, 90)),
            wins_over_500_pct=float(np.mean(wins >= 81) * 100),
            wins_90_plus_pct=float(np.mean(wins >= 90) * 100),
            wins_100_plus_pct=float(np.mean(wins >= 100) * 100),
            win_distribution=wins,
        )

    # Build DataFrame for playoff calculation
    team_df = pd.DataFrame([
        {"team": team, "division": team_projection_sets[team].division}
        for team in teams
    ])

    _calculate_playoff_odds(simulation, team_df, simulated_wins, iterations)

    return simulation


def _calculate_playoff_odds(
    simulation: SeasonSimulation,
    team_projections: pd.DataFrame,
    simulated_wins: dict[str, np.ndarray],
    iterations: int,
) -> None:
    """Calculate division winner and wild card odds."""

    # Group teams by division
    divisions: dict[str, list[str]] = {}
    for _, row in team_projections.iterrows():
        div = row["division"]
        if div not in divisions:
            divisions[div] = []
        divisions[div].append(row["team"])

    # Initialize counters
    division_win_counts = {team: 0 for team in simulated_wins}
    wild_card_counts = {team: 0 for team in simulated_wins}
    last_place_counts = {team: 0 for team in simulated_wins}

    # Vectorized playoff calculation
    for i in range(iterations):
        # Division winners
        sim_division_winners = set()
        for div, div_teams in divisions.items():
            wins_this_sim = {t: simulated_wins[t][i] for t in div_teams}
            winner = max(wins_this_sim, key=wins_this_sim.get)
            division_win_counts[winner] += 1
            sim_division_winners.add(winner)

            # Last place
            loser = min(wins_this_sim, key=wins_this_sim.get)
            last_place_counts[loser] += 1

        # Wild cards - 3 per league
        for league in ["AL", "NL"]:
            league_non_winners = [
                t for div, teams in divisions.items()
                if league in div
                for t in teams
                if t not in sim_division_winners
            ]
            # Sort by wins this simulation
            league_non_winners.sort(
                key=lambda t: simulated_wins[t][i], reverse=True
            )
            # Top 3 get wild card spots
            for wc_team in league_non_winners[:3]:
                wild_card_counts[wc_team] += 1

    # Convert to percentages
    for team in simulated_wins:
        result = simulation.team_results[team]
        result.division_winner_pct = division_win_counts[team] / iterations * 100
        result.wild_card_pct = wild_card_counts[team] / iterations * 100
        result.playoff_pct = result.division_winner_pct + result.wild_card_pct
        result.last_place_pct = last_place_counts[team] / iterations * 100


def calculate_matchup_probability(
    team1: str,
    team2: str,
    simulation: SeasonSimulation,
) -> dict:
    """Calculate probability of team1 finishing ahead of team2.

    Args:
        team1: First team abbreviation.
        team2: Second team abbreviation.
        simulation: Completed simulation.

    Returns:
        Dict with matchup probabilities.
    """
    wins1 = simulation.team_results[team1].win_distribution
    wins2 = simulation.team_results[team2].win_distribution

    team1_ahead = np.sum(wins1 > wins2)
    team2_ahead = np.sum(wins2 > wins1)
    ties = np.sum(wins1 == wins2)

    total = len(wins1)

    return {
        "team1": team1,
        "team2": team2,
        f"{team1}_wins_more": round(team1_ahead / total * 100, 1),
        f"{team2}_wins_more": round(team2_ahead / total * 100, 1),
        "tie_pct": round(ties / total * 100, 1),
        "avg_margin": round(float(np.mean(wins1 - wins2)), 1),
    }


def probability_of_n_wins(
    team: str,
    n_wins: int,
    simulation: SeasonSimulation,
    at_least: bool = True,
) -> float:
    """Calculate probability of a team winning at least (or exactly) N games.

    Args:
        team: Team abbreviation.
        n_wins: Win threshold.
        simulation: Completed simulation.
        at_least: If True, P(wins >= n). If False, P(wins == n).

    Returns:
        Probability as percentage.
    """
    wins = simulation.team_results[team].win_distribution
    if at_least:
        return float(np.mean(wins >= n_wins) * 100)
    else:
        return float(np.mean(np.floor(wins) == n_wins) * 100)
