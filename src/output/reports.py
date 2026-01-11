"""Generate summary reports from projections and simulations."""

import logging
from datetime import datetime
from pathlib import Path
from typing import TextIO

import pandas as pd

from ..simulation.monte_carlo import SeasonSimulation
from ..simulation.standings import generate_standings, standings_to_dataframe

logger = logging.getLogger(__name__)


def generate_season_report(
    simulation: SeasonSimulation,
    output_path: Path | str | None = None,
) -> str:
    """Generate a text report of season projections.

    Args:
        simulation: Completed season simulation.
        output_path: Optional file path to save report.

    Returns:
        Report as a string.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("MLB SEASON PROJECTION REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Simulations: {simulation.iterations:,}")
    lines.append("=" * 70)
    lines.append("")

    # Generate standings
    standings = generate_standings(simulation)

    for league in ["AL", "NL"]:
        lines.append(f"\n{'='*35}")
        lines.append(f"  {league} STANDINGS")
        lines.append(f"{'='*35}\n")

        for div_name, div_standings in sorted(standings[league].divisions.items()):
            lines.append(f"\n{div_name}")
            lines.append("-" * 50)
            lines.append(
                f"{'Team':<6} {'W':>5} {'Range':>12} {'Div%':>6} {'Playoff%':>8}"
            )
            lines.append("-" * 50)

            for team in div_standings.teams:
                win_range = f"{team['wins_10th_pct']:.0f}-{team['wins_90th_pct']:.0f}"
                lines.append(
                    f"{team['team']:<6} {team['mean_wins']:>5.1f} "
                    f"{win_range:>12} {team['division_winner_pct']:>5.1f}% "
                    f"{team['playoff_pct']:>7.1f}%"
                )

        # Wild card race
        lines.append(f"\n{league} Wild Card Race")
        lines.append("-" * 40)
        for team in standings[league].wild_card_race[:6]:
            lines.append(
                f"  {team['team']:<6} {team['mean_wins']:>5.1f} wins  "
                f"Playoff: {team['playoff_pct']:>5.1f}%"
            )

    # Summary stats
    lines.append("\n" + "=" * 70)
    lines.append("TOP PROJECTED TEAMS")
    lines.append("=" * 70)

    all_teams = [
        (team, result.mean_wins, result.playoff_pct)
        for team, result in simulation.team_results.items()
    ]
    all_teams.sort(key=lambda x: x[1], reverse=True)

    lines.append("\nTop 10 by Projected Wins:")
    for i, (team, wins, playoff_pct) in enumerate(all_teams[:10], 1):
        lines.append(f"  {i:2}. {team:<6} {wins:>5.1f} wins  ({playoff_pct:.0f}% playoff)")

    lines.append("\nBottom 5 by Projected Wins:")
    for team, wins, playoff_pct in all_teams[-5:]:
        lines.append(f"      {team:<6} {wins:>5.1f} wins")

    report = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        logger.info(f"Report saved to {output_path}")

    return report


def generate_team_report(
    team: str,
    simulation: SeasonSimulation,
) -> str:
    """Generate a detailed report for a single team.

    Args:
        team: Team abbreviation.
        simulation: Completed season simulation.

    Returns:
        Team report as a string.
    """
    if team not in simulation.team_results:
        return f"Team {team} not found in simulation results"

    result = simulation.team_results[team]

    lines = []
    lines.append(f"\n{team} SEASON PROJECTION")
    lines.append("=" * 40)
    lines.append(f"\nProjected Record: {result.mean_wins:.0f}-{162 - result.mean_wins:.0f}")
    lines.append(f"Median Wins: {result.median_wins:.0f}")
    lines.append(f"Standard Deviation: {result.std_wins:.1f} wins")
    lines.append(f"\nWin Distribution:")
    lines.append(f"  10th percentile: {result.wins_10th_pct:.0f} wins")
    lines.append(f"  25th percentile: {result.wins_25th_pct:.0f} wins")
    lines.append(f"  75th percentile: {result.wins_75th_pct:.0f} wins")
    lines.append(f"  90th percentile: {result.wins_90th_pct:.0f} wins")
    lines.append(f"\nPlayoff Odds:")
    lines.append(f"  Division Winner: {result.division_winner_pct:.1f}%")
    lines.append(f"  Wild Card: {result.wild_card_pct:.1f}%")
    lines.append(f"  Make Playoffs: {result.playoff_pct:.1f}%")

    return "\n".join(lines)


def export_to_csv(
    simulation: SeasonSimulation,
    output_path: Path | str,
) -> None:
    """Export simulation results to CSV.

    Args:
        simulation: Completed season simulation.
        output_path: Path for output CSV.
    """
    standings = generate_standings(simulation)
    df = standings_to_dataframe(standings)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Results exported to {output_path}")
