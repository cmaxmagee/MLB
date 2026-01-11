"""Monte Carlo simulation for season outcomes.

Runs thousands of simulated seasons to generate probability distributions
for wins, division standings, and playoff odds.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..projections.team import pythagorean_wins

logger = logging.getLogger(__name__)

# Default simulation parameters
DEFAULT_ITERATIONS = 10_000
GAMES_PER_SEASON = 162

# Historical standard deviation of team wins around projection
# Teams typically deviate 6-8 wins from their "true" talent level
TEAM_WIN_STDEV = 7.0


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

    # Full distribution for analysis
    win_distribution: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class SeasonSimulation:
    """Complete season simulation results."""

    iterations: int
    team_results: dict[str, SimulationResult] = field(default_factory=dict)
    division_winner_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    wild_card_counts: dict[str, int] = field(default_factory=dict)


def run_simulation(
    team_projections: pd.DataFrame,
    iterations: int = DEFAULT_ITERATIONS,
    player_variance: pd.DataFrame | None = None,
    team_variance: float = TEAM_WIN_STDEV,
    random_seed: int | None = None,
) -> SeasonSimulation:
    """Run Monte Carlo simulation of the season.

    Args:
        team_projections: DataFrame with columns:
            team, division, projected_runs_scored, projected_runs_allowed
        iterations: Number of seasons to simulate.
        player_variance: Optional player-level variance adjustments.
        team_variance: Standard deviation of team wins (default 7).
        random_seed: Random seed for reproducibility.

    Returns:
        SeasonSimulation with full results.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    teams = team_projections["team"].tolist()
    n_teams = len(teams)

    # Pre-calculate base win projections using Pythagorean expectation
    base_wins = {}
    for _, row in team_projections.iterrows():
        wins = pythagorean_wins(
            row["projected_runs_scored"],
            row["projected_runs_allowed"],
        )
        base_wins[row["team"]] = wins

    # Simulate seasons
    logger.info(f"Running {iterations:,} season simulations...")
    simulated_wins = {team: np.zeros(iterations) for team in teams}

    for i in tqdm(range(iterations), desc="Simulating seasons"):
        for team in teams:
            # Add variance components
            # 1. Team-level random variance
            team_noise = np.random.normal(0, team_variance)

            # 2. Clamp to valid range [0, 162]
            wins = np.clip(base_wins[team] + team_noise, 0, GAMES_PER_SEASON)
            simulated_wins[team][i] = wins

    # Calculate results
    simulation = SeasonSimulation(iterations=iterations)

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
            win_distribution=wins,
        )

    # Calculate playoff odds
    _calculate_playoff_odds(simulation, team_projections, simulated_wins, iterations)

    return simulation


def _calculate_playoff_odds(
    simulation: SeasonSimulation,
    team_projections: pd.DataFrame,
    simulated_wins: dict[str, np.ndarray],
    iterations: int,
) -> None:
    """Calculate division winner and wild card odds."""

    # Group teams by division
    divisions = {}
    for _, row in team_projections.iterrows():
        div = row["division"]
        if div not in divisions:
            divisions[div] = []
        divisions[div].append(row["team"])

    # Track division winners and wild cards per simulation
    division_winners = {div: [] for div in divisions}
    wild_cards = {"AL": [], "NL": []}

    for i in range(iterations):
        # Find division winner for each division
        sim_division_winners = {}
        for div, teams in divisions.items():
            best_team = max(teams, key=lambda t: simulated_wins[t][i])
            sim_division_winners[div] = best_team
            division_winners[div].append(best_team)

        # Find wild cards (next 3 best teams per league)
        for league in ["AL", "NL"]:
            league_teams = [
                t for div, teams in divisions.items()
                if league in div
                for t in teams
                if t != sim_division_winners.get(div)
            ]
            # Sort by wins this simulation
            league_teams.sort(key=lambda t: simulated_wins[t][i], reverse=True)
            wild_cards[league].append(league_teams[:3])

    # Calculate percentages
    for div, winners in division_winners.items():
        for team in divisions[div]:
            count = winners.count(team)
            pct = count / iterations * 100
            simulation.team_results[team].division_winner_pct = pct

    # Wild card percentages
    for league in ["AL", "NL"]:
        for wc_list in wild_cards[league]:
            for team in wc_list:
                if team in simulation.team_results:
                    simulation.team_results[team].wild_card_pct += 100 / iterations

    # Total playoff percentage
    for team in simulation.team_results:
        result = simulation.team_results[team]
        result.playoff_pct = result.division_winner_pct + result.wild_card_pct


def simulate_with_player_variance(
    team_projections: pd.DataFrame,
    player_projections: pd.DataFrame,
    iterations: int = DEFAULT_ITERATIONS,
) -> SeasonSimulation:
    """Run simulation with player-level variance modeling.

    This is a more sophisticated simulation that models individual
    player performance variance and its impact on team outcomes.

    Args:
        team_projections: Team-level projections.
        player_projections: Player projections with uncertainty.
        iterations: Number of simulations.

    Returns:
        SeasonSimulation with results.
    """
    # TODO: Implement player-level variance
    # For each simulation:
    # 1. Sample each player's performance from their distribution
    # 2. Re-aggregate to team level
    # 3. Calculate team wins
    raise NotImplementedError("Player-level variance not yet implemented")
