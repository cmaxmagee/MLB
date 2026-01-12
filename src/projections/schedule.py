"""Strength of Schedule calculations.

Calculates schedule difficulty based on opponent team projections
and adjusts win projections accordingly.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .projector import TeamProjectionSet

from .team import DIVISIONS, TEAM_TO_DIVISION

logger = logging.getLogger(__name__)

# Games per season
GAMES_PER_SEASON = 162

# MLB Schedule Structure (2023+ format)
# Teams play more games against division rivals
DIVISIONAL_GAMES = 13  # Games vs each of 4 division rivals = 52 total
NON_DIVISIONAL_LEAGUE_GAMES = 6.6  # ~66 games vs 10 other teams in league
INTERLEAGUE_GAMES_PER_TEAM = 2.93  # ~44 games vs 15 interleague teams


@dataclass
class ScheduleBreakdown:
    """Breakdown of a team's schedule by opponent type."""

    team: str
    division: str

    # Games breakdown
    divisional_games: int = 52  # 13 * 4 division rivals
    non_divisional_league_games: int = 66  # ~6.6 * 10 teams
    interleague_games: int = 44  # ~3 * 15 teams

    # Opponent strength (avg projected win% of opponents)
    divisional_opponent_strength: float = 0.500
    non_divisional_opponent_strength: float = 0.500
    interleague_opponent_strength: float = 0.500

    # Overall SOS
    overall_sos: float = 0.500  # 0.500 = neutral
    sos_win_adjustment: float = 0.0  # Adjustment to apply to projected wins


def get_division_opponents(team: str) -> List[str]:
    """Get a team's division opponents.

    Args:
        team: Team abbreviation.

    Returns:
        List of 4 division opponent abbreviations.
    """
    division = TEAM_TO_DIVISION.get(team)
    if not division:
        return []

    return [t for t in DIVISIONS.get(division, []) if t != team]


def get_league_opponents(team: str, exclude_division: bool = True) -> List[str]:
    """Get a team's same-league opponents.

    Args:
        team: Team abbreviation.
        exclude_division: If True, excludes division rivals.

    Returns:
        List of opponent abbreviations.
    """
    division = TEAM_TO_DIVISION.get(team)
    if not division:
        return []

    league = "AL" if "AL" in division else "NL"
    opponents = []

    for div, teams in DIVISIONS.items():
        if league in div:
            if exclude_division and div == division:
                continue
            opponents.extend([t for t in teams if t != team])

    return opponents


def get_interleague_opponents(team: str) -> List[str]:
    """Get a team's interleague opponents.

    Args:
        team: Team abbreviation.

    Returns:
        List of 15 interleague opponent abbreviations.
    """
    division = TEAM_TO_DIVISION.get(team)
    if not division:
        return []

    league = "AL" if "AL" in division else "NL"
    other_league = "NL" if league == "AL" else "AL"

    opponents = []
    for div, teams in DIVISIONS.items():
        if other_league in div:
            opponents.extend(teams)

    return opponents


def calculate_opponent_strength(
    opponents: List[str],
    team_win_pcts: Dict[str, float],
) -> float:
    """Calculate average win percentage of opponents.

    Args:
        opponents: List of opponent team abbreviations.
        team_win_pcts: Dict mapping team to projected win percentage.

    Returns:
        Average opponent win percentage.
    """
    if not opponents:
        return 0.500

    total_win_pct = sum(
        team_win_pcts.get(opp, 0.500)
        for opp in opponents
    )

    return total_win_pct / len(opponents)


def calculate_sos(
    team: str,
    team_projections: Dict[str, "TeamProjectionSet"],
) -> ScheduleBreakdown:
    """Calculate Strength of Schedule for a team.

    Args:
        team: Team abbreviation.
        team_projections: Dict of all team projections.

    Returns:
        ScheduleBreakdown with SOS calculations.
    """
    division = TEAM_TO_DIVISION.get(team, "Unknown")

    # Convert projections to win percentages
    team_win_pcts = {
        t: proj.projected_wins / GAMES_PER_SEASON
        for t, proj in team_projections.items()
    }

    # Get opponent groups
    div_opponents = get_division_opponents(team)
    league_opponents = get_league_opponents(team, exclude_division=True)
    interleague_opponents = get_interleague_opponents(team)

    # Calculate strength of each opponent group
    div_strength = calculate_opponent_strength(div_opponents, team_win_pcts)
    league_strength = calculate_opponent_strength(league_opponents, team_win_pcts)
    interleague_strength = calculate_opponent_strength(interleague_opponents, team_win_pcts)

    # Calculate weighted overall SOS
    # Weights based on games played against each group
    div_games = 52
    league_games = 66
    inter_games = 44
    total_games = div_games + league_games + inter_games

    overall_sos = (
        div_strength * div_games +
        league_strength * league_games +
        interleague_strength * inter_games
    ) / total_games

    # Calculate win adjustment
    # A team facing tougher opponents should have their win projection adjusted
    # The adjustment is based on how much harder/easier their schedule is vs average
    # Research suggests ~0.4-0.5 wins per 0.01 SOS difference
    sos_diff = overall_sos - 0.500
    # Negative adjustment for tough schedules, positive for easy schedules
    # Multiply by adjustment factor (empirically ~65-70 wins worth of impact)
    win_adjustment = -sos_diff * 65

    return ScheduleBreakdown(
        team=team,
        division=division,
        divisional_games=div_games,
        non_divisional_league_games=league_games,
        interleague_games=inter_games,
        divisional_opponent_strength=div_strength,
        non_divisional_opponent_strength=league_strength,
        interleague_opponent_strength=interleague_strength,
        overall_sos=overall_sos,
        sos_win_adjustment=win_adjustment,
    )


def calculate_all_sos(
    team_projections: Dict[str, "TeamProjectionSet"],
) -> Dict[str, ScheduleBreakdown]:
    """Calculate Strength of Schedule for all teams.

    Args:
        team_projections: Dict of all team projections.

    Returns:
        Dict mapping team abbreviation to ScheduleBreakdown.
    """
    sos_results = {}

    for team in team_projections:
        sos_results[team] = calculate_sos(team, team_projections)

    return sos_results


def apply_sos_adjustments(
    team_projections: Dict[str, "TeamProjectionSet"],
    sos_results: Optional[Dict[str, ScheduleBreakdown]] = None,
) -> Dict[str, "TeamProjectionSet"]:
    """Apply Strength of Schedule adjustments to team projections.

    Modifies the team projections in place and returns them.

    Args:
        team_projections: Dict of team projections.
        sos_results: Pre-calculated SOS results (will calculate if not provided).

    Returns:
        Updated team projections with SOS adjustments applied.
    """
    if sos_results is None:
        sos_results = calculate_all_sos(team_projections)

    for team, proj in team_projections.items():
        if team in sos_results:
            sos = sos_results[team]

            # Store SOS info on projection
            proj.sos = sos.overall_sos
            proj.sos_win_adjustment = sos.sos_win_adjustment

            # Apply adjustment to projected wins
            proj.projected_wins_pre_sos = proj.projected_wins
            proj.projected_wins = proj.projected_wins + sos.sos_win_adjustment

            # Bound wins to reasonable range
            proj.projected_wins = max(30, min(130, proj.projected_wins))

            if abs(sos.sos_win_adjustment) >= 0.5:
                logger.debug(
                    f"{team}: SOS={sos.overall_sos:.3f}, "
                    f"adjustment={sos.sos_win_adjustment:+.1f} wins"
                )

    return team_projections


def get_division_sos_summary(
    sos_results: Dict[str, ScheduleBreakdown],
) -> Dict[str, Dict[str, float]]:
    """Get SOS summary by division.

    Args:
        sos_results: Dict of ScheduleBreakdown by team.

    Returns:
        Dict mapping division to summary stats.
    """
    division_sos = {}

    for div, teams in DIVISIONS.items():
        div_sos_values = [
            sos_results[t].overall_sos
            for t in teams
            if t in sos_results
        ]

        if div_sos_values:
            division_sos[div] = {
                "avg_sos": sum(div_sos_values) / len(div_sos_values),
                "min_sos": min(div_sos_values),
                "max_sos": max(div_sos_values),
            }

    return division_sos


def format_sos_report(
    sos_results: Dict[str, ScheduleBreakdown],
) -> str:
    """Format SOS results as a readable report.

    Args:
        sos_results: Dict of ScheduleBreakdown by team.

    Returns:
        Formatted string report.
    """
    lines = ["STRENGTH OF SCHEDULE REPORT", "=" * 50, ""]

    # Sort by SOS (hardest to easiest)
    sorted_teams = sorted(
        sos_results.keys(),
        key=lambda t: sos_results[t].overall_sos,
        reverse=True,
    )

    lines.append(f"{'Team':<6} {'SOS':>6} {'Div Opp':>8} {'League':>8} {'IL':>8} {'Adj':>6}")
    lines.append("-" * 50)

    for team in sorted_teams:
        sos = sos_results[team]
        lines.append(
            f"{team:<6} {sos.overall_sos:>6.3f} "
            f"{sos.divisional_opponent_strength:>8.3f} "
            f"{sos.non_divisional_opponent_strength:>8.3f} "
            f"{sos.interleague_opponent_strength:>8.3f} "
            f"{sos.sos_win_adjustment:>+5.1f}"
        )

    lines.append("")
    lines.append("Division Summary:")
    lines.append("-" * 30)

    div_summary = get_division_sos_summary(sos_results)
    for div in sorted(div_summary.keys()):
        stats = div_summary[div]
        lines.append(f"  {div}: avg {stats['avg_sos']:.3f}")

    return "\n".join(lines)
