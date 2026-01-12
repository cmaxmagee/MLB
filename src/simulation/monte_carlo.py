"""Monte Carlo simulation for season outcomes.

Runs thousands of simulated seasons to generate probability distributions
for wins, division standings, and playoff odds.
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Union

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

# Playing time variance by age (injury risk modeling)
# Older players have more variance in playing time due to injury risk
# Values are coefficient of variation (std_dev / mean)
PLAYING_TIME_CV_BY_AGE = {
    "young": 0.08,    # Under 26: ~8% variance (most durable)
    "prime": 0.12,    # 26-30: ~12% variance
    "veteran": 0.18,  # 31-34: ~18% variance
    "old": 0.25,      # 35+: ~25% variance (highest injury risk)
}

# Young player upside: wider distributions for breakout candidates
YOUNG_PLAYER_AGE_THRESHOLD = 26
YOUNG_PLAYER_EXPERIENCE_THRESHOLD = 2  # seasons_of_data
YOUNG_PLAYER_VARIANCE_MULTIPLIER = 1.5  # 50% wider distributions


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
    pennant_pct: float = 0.0  # League championship (ALCS/NLCS winner)
    world_series_pct: float = 0.0  # World Series appearance (same as pennant)
    champion_pct: float = 0.0  # World Series winner

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
    random_seed: Optional[int] = None
    team_results: Dict[str, SimulationResult] = field(default_factory=dict)
    division_winner_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    wild_card_counts: Dict[str, int] = field(default_factory=dict)

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
                "Pennant %": round(result.pennant_pct, 1),
                "WS Champ %": round(result.champion_pct, 1),
            })
        df = pd.DataFrame(rows)
        return df.sort_values("Playoff %", ascending=False).reset_index(drop=True)


def get_playing_time_variance(age: int) -> float:
    """Get the coefficient of variation for playing time based on age.

    Older players have higher variance due to increased injury risk.

    Args:
        age: Player's age for the projection year.

    Returns:
        Coefficient of variation (std_dev / mean) for playing time.
    """
    if age < 26:
        return PLAYING_TIME_CV_BY_AGE["young"]
    elif age <= 30:
        return PLAYING_TIME_CV_BY_AGE["prime"]
    elif age <= 34:
        return PLAYING_TIME_CV_BY_AGE["veteran"]
    else:
        return PLAYING_TIME_CV_BY_AGE["old"]


def is_young_player_with_upside(
    age: int,
    seasons_of_data: int,
) -> bool:
    """Check if a player qualifies for the young player upside boost.

    Young players with limited experience have wider potential outcomes,
    both up and down. This function identifies candidates for increased
    simulation variance.

    Args:
        age: Player's age for the projection year.
        seasons_of_data: Number of MLB seasons with significant playing time.

    Returns:
        True if player qualifies for upside variance boost.
    """
    return (
        age < YOUNG_PLAYER_AGE_THRESHOLD
        and seasons_of_data < YOUNG_PLAYER_EXPERIENCE_THRESHOLD
    )


def sample_playing_time(
    projected_pa_or_ip: float,
    age: int,
    is_pitcher: bool = False,
) -> float:
    """Sample playing time with age-based injury variance.

    Instead of using deterministic playing time projections, this samples
    from a distribution where variance increases with age (injury risk).

    Args:
        projected_pa_or_ip: Projected PA (batters) or IP (pitchers).
        age: Player's age for the projection year.
        is_pitcher: Whether this is a pitcher (affects bounds).

    Returns:
        Sampled playing time value.
    """
    cv = get_playing_time_variance(age)
    std_dev = projected_pa_or_ip * cv

    sampled = np.random.normal(projected_pa_or_ip, std_dev)

    # Apply reasonable bounds
    if is_pitcher:
        # IP bounds: minimum 0, max depends on starter vs reliever
        sampled = max(0, min(sampled, projected_pa_or_ip * 1.3))
    else:
        # PA bounds: minimum 0, max ~720 (everyday player full season)
        sampled = max(0, min(sampled, 720))

    return sampled


def run_simulation(
    team_projections: Union[pd.DataFrame, Dict[str, "TeamProjectionSet"]],
    iterations: int = DEFAULT_ITERATIONS,
    team_variance: float = TEAM_WIN_STDEV,
    random_seed: Optional[int] = None,
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
    team_projection_sets: Dict[str, "TeamProjectionSet"],
    iterations: int = DEFAULT_ITERATIONS,
    team_residual_std: float = 4.0,
    random_seed: Optional[int] = None,
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
                # Sample playing time with injury variance
                sampled_pa = sample_playing_time(
                    batter.projected_pa,
                    batter.age,
                    is_pitcher=False,
                )

                # wRC+ varies with std ~15 points
                # Confidence reduces variance
                batter_std = BATTER_WRC_PLUS_STDEV * (2.0 - batter.projection_confidence)

                # Young player upside: wider distribution for breakout candidates
                if is_young_player_with_upside(batter.age, batter.seasons_of_data):
                    batter_std *= YOUNG_PLAYER_VARIANCE_MULTIPLIER

                sampled_wrc_plus = np.random.normal(
                    batter.projected_wrc_plus, batter_std
                )
                # Convert sampled wRC+ to batting runs using sampled PA
                sampled_runs = (sampled_wrc_plus - 100) / 100 * sampled_pa * 0.12
                sampled_batting_runs += sampled_runs

            # Sample pitcher performances
            sampled_pitching_runs = 0.0
            for pitcher in proj_set.pitchers:
                # Sample playing time with injury variance
                sampled_ip = sample_playing_time(
                    pitcher.projected_ip,
                    pitcher.age,
                    is_pitcher=True,
                )

                # FIP varies with std ~0.5
                pitcher_std = PITCHER_FIP_STDEV * (2.0 - pitcher.projection_confidence)

                # Young player upside: wider distribution for breakout candidates
                if is_young_player_with_upside(pitcher.age, pitcher.seasons_of_data):
                    pitcher_std *= YOUNG_PLAYER_VARIANCE_MULTIPLIER

                sampled_fip = np.random.normal(pitcher.projected_fip, pitcher_std)
                # Convert sampled FIP to pitching runs using sampled IP
                league_fip = 4.00
                sampled_runs = -((sampled_fip - league_fip) / 9 * sampled_ip)
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


def _simulate_series(
    team_a_wins: float,
    team_b_wins: float,
    best_of: int = 7,
) -> float:
    """Calculate probability of team A winning a playoff series using log5.

    Uses the log5 formula to convert win totals to head-to-head probability,
    then calculates series win probability.

    Args:
        team_a_wins: Team A's season win total.
        team_b_wins: Team B's season win total.
        best_of: Series length (3, 5, or 7).

    Returns:
        Probability of team A winning the series (0-1).
    """
    # Convert wins to win percentage
    pct_a = team_a_wins / 162
    pct_b = team_b_wins / 162

    # Log5 formula for head-to-head probability
    # P(A beats B) = (pA * (1-pB)) / (pA * (1-pB) + pB * (1-pA))
    if pct_a + pct_b == 0:
        p_a_wins_game = 0.5
    else:
        numerator = pct_a * (1 - pct_b)
        denominator = pct_a * (1 - pct_b) + pct_b * (1 - pct_a)
        p_a_wins_game = numerator / denominator if denominator > 0 else 0.5

    # Add small home field advantage for higher seed (~2%)
    if team_a_wins > team_b_wins:
        p_a_wins_game = min(0.95, p_a_wins_game + 0.02)
    elif team_b_wins > team_a_wins:
        p_a_wins_game = max(0.05, p_a_wins_game - 0.02)

    # Calculate series win probability using binomial
    wins_needed = (best_of // 2) + 1
    p_a_wins_series = 0.0

    # Sum probability of A winning in exactly wins_needed + k games
    # where k is extra games (0 to wins_needed - 1)
    from math import comb
    for losses in range(wins_needed):
        # A wins 'wins_needed' games and loses 'losses' games
        # Last game must be a win for A
        games_before_last = wins_needed - 1 + losses
        ways = comb(games_before_last, losses)
        prob = ways * (p_a_wins_game ** wins_needed) * ((1 - p_a_wins_game) ** losses)
        p_a_wins_series += prob

    return p_a_wins_series


def _calculate_playoff_odds(
    simulation: SeasonSimulation,
    team_projections: pd.DataFrame,
    simulated_wins: Dict[str, np.ndarray],
    iterations: int,
) -> None:
    """Calculate division winner, wild card, pennant, and championship odds."""

    # Group teams by division
    divisions: Dict[str, List[str]] = {}
    for _, row in team_projections.iterrows():
        div = row["division"]
        if div not in divisions:
            divisions[div] = []
        divisions[div].append(row["team"])

    # Initialize counters
    division_win_counts = {team: 0 for team in simulated_wins}
    wild_card_counts = {team: 0 for team in simulated_wins}
    last_place_counts = {team: 0 for team in simulated_wins}
    pennant_counts = {team: 0 for team in simulated_wins}
    champion_counts = {team: 0 for team in simulated_wins}

    # Playoff simulation for each iteration
    for i in range(iterations):
        # Division winners
        sim_division_winners = {}  # {division: (winner, wins)}
        for div, div_teams in divisions.items():
            wins_this_sim = {t: simulated_wins[t][i] for t in div_teams}
            winner = max(wins_this_sim, key=wins_this_sim.get)
            division_win_counts[winner] += 1
            sim_division_winners[div] = (winner, wins_this_sim[winner])

            # Last place
            loser = min(wins_this_sim, key=wins_this_sim.get)
            last_place_counts[loser] += 1

        # Build playoff brackets for each league
        league_champions = {}

        for league in ["AL", "NL"]:
            # Get division winners for this league, sorted by wins
            league_div_winners = [
                (team, wins) for div, (team, wins) in sim_division_winners.items()
                if league in div
            ]
            league_div_winners.sort(key=lambda x: x[1], reverse=True)

            # Get wild card teams
            league_non_winners = [
                (t, simulated_wins[t][i])
                for div, teams in divisions.items()
                if league in div
                for t in teams
                if t != sim_division_winners[div][0]
            ]
            league_non_winners.sort(key=lambda x: x[1], reverse=True)
            wild_cards = league_non_winners[:3]

            # Record wild card counts
            for wc_team, _ in wild_cards:
                wild_card_counts[wc_team] += 1

            # Seeds: 1-3 are division winners, 4-6 are wild cards
            seeds = league_div_winners + wild_cards  # [(team, wins), ...]

            # Wild Card Round (best of 3)
            # #3 vs #6, #4 vs #5
            wc_game1_winner = _pick_series_winner(seeds[2], seeds[5], 3)
            wc_game2_winner = _pick_series_winner(seeds[3], seeds[4], 3)

            # Division Series (best of 5)
            # #1 vs lowest remaining seed, #2 vs other
            # Determine matchups based on seeds
            wc_winners = sorted([wc_game1_winner, wc_game2_winner], key=lambda x: x[1], reverse=True)

            # #1 plays lowest remaining, #2 plays highest remaining
            ds1_winner = _pick_series_winner(seeds[0], wc_winners[1], 5)  # #1 vs lower
            ds2_winner = _pick_series_winner(seeds[1], wc_winners[0], 5)  # #2 vs higher

            # League Championship Series (best of 7)
            lcs_winner = _pick_series_winner(ds1_winner, ds2_winner, 7)
            pennant_counts[lcs_winner[0]] += 1
            league_champions[league] = lcs_winner

        # World Series (best of 7)
        al_champ = league_champions["AL"]
        nl_champ = league_champions["NL"]
        ws_winner = _pick_series_winner(al_champ, nl_champ, 7)
        champion_counts[ws_winner[0]] += 1

    # Convert to percentages
    for team in simulated_wins:
        result = simulation.team_results[team]
        result.division_winner_pct = division_win_counts[team] / iterations * 100
        result.wild_card_pct = wild_card_counts[team] / iterations * 100
        result.playoff_pct = result.division_winner_pct + result.wild_card_pct
        result.pennant_pct = pennant_counts[team] / iterations * 100
        result.world_series_pct = result.pennant_pct  # Same as pennant (WS appearance)
        result.champion_pct = champion_counts[team] / iterations * 100
        result.last_place_pct = last_place_counts[team] / iterations * 100


def _pick_series_winner(
    team_a: tuple,
    team_b: tuple,
    best_of: int,
) -> tuple:
    """Randomly pick a series winner based on probability.

    Args:
        team_a: (team_name, wins) tuple.
        team_b: (team_name, wins) tuple.
        best_of: Series length.

    Returns:
        Winning team's (team_name, wins) tuple.
    """
    p_a_wins = _simulate_series(team_a[1], team_b[1], best_of)
    if np.random.random() < p_a_wins:
        return team_a
    return team_b


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
