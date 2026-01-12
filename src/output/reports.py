"""Generate summary reports from projections and simulations."""

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import pandas as pd

from ..simulation.monte_carlo import SeasonSimulation
from ..projections.team import TEAM_TO_DIVISION, TEAM_FULL_NAMES, DIVISIONS

if TYPE_CHECKING:
    from ..projections.projector import TeamProjectionSet

logger = logging.getLogger(__name__)


def generate_season_report(
    simulation: SeasonSimulation,
    output_path: Optional[Union[Path, str]] = None,
    projection_year: Optional[int] = None,
) -> str:
    """Generate a text report of season projections.

    Args:
        simulation: Completed season simulation.
        output_path: Optional file path to save report.
        projection_year: Year being projected.

    Returns:
        Report as a string.
    """
    lines = []
    year_str = str(projection_year) if projection_year else ""

    lines.append("=" * 72)
    lines.append(f"  MLB {year_str} SEASON PROJECTION REPORT".center(72))
    lines.append("=" * 72)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Simulations: {simulation.iterations:,}")
    lines.append("")

    # Generate standings by division
    for league in ["AL", "NL"]:
        lines.append("")
        lines.append("=" * 72)
        lines.append(f"  {league} STANDINGS".center(72))
        lines.append("=" * 72)

        for div in sorted(DIVISIONS.keys()):
            if league not in div:
                continue

            lines.append(f"\n{div}")
            lines.append("-" * 72)
            lines.append(
                f"{'Team':<5} {'Full Name':<24} {'W':>5} {'L':>5} "
                f"{'Range':>9} {'Div%':>6} {'WC%':>5} {'Playoff':>7}"
            )
            lines.append("-" * 72)

            # Get teams in this division
            div_teams = DIVISIONS[div]
            div_results = [
                (team, simulation.team_results[team])
                for team in div_teams
                if team in simulation.team_results
            ]
            div_results.sort(key=lambda x: x[1].mean_wins, reverse=True)

            for team, result in div_results:
                full_name = TEAM_FULL_NAMES.get(team, team)[:24]
                losses = 162 - result.mean_wins
                win_range = f"{result.wins_10th_pct:.0f}-{result.wins_90th_pct:.0f}"
                lines.append(
                    f"{team:<5} {full_name:<24} {result.mean_wins:>5.1f} {losses:>5.1f} "
                    f"{win_range:>9} {result.division_winner_pct:>5.1f}% "
                    f"{result.wild_card_pct:>4.1f}% {result.playoff_pct:>6.1f}%"
                )

    # Overall rankings
    lines.append("")
    lines.append("=" * 72)
    lines.append("  OVERALL POWER RANKINGS".center(72))
    lines.append("=" * 72)

    all_teams = [
        (team, result)
        for team, result in simulation.team_results.items()
    ]
    all_teams.sort(key=lambda x: x[1].mean_wins, reverse=True)

    lines.append("")
    lines.append(
        f"{'Rank':<5} {'Team':<5} {'W':>6} {'L':>6} {'Playoff%':>9} "
        f"{'90+ W%':>7} {'100+ W%':>8}"
    )
    lines.append("-" * 52)

    for i, (team, result) in enumerate(all_teams, 1):
        losses = 162 - result.mean_wins
        lines.append(
            f"{i:<5} {team:<5} {result.mean_wins:>6.1f} {losses:>6.1f} "
            f"{result.playoff_pct:>8.1f}% {result.wins_90_plus_pct:>6.1f}% "
            f"{result.wins_100_plus_pct:>7.1f}%"
        )

    # Contender tiers
    lines.append("")
    lines.append("=" * 72)
    lines.append("  CONTENDER TIERS".center(72))
    lines.append("=" * 72)

    # Elite (>90% playoff odds)
    elite = [(t, r) for t, r in all_teams if r.playoff_pct >= 90]
    if elite:
        lines.append("\nELITE CONTENDERS (90%+ playoff odds):")
        for team, result in elite:
            lines.append(f"  {team}: {result.mean_wins:.0f} wins, {result.playoff_pct:.0f}% playoff")

    # Strong (60-90%)
    strong = [(t, r) for t, r in all_teams if 60 <= r.playoff_pct < 90]
    if strong:
        lines.append("\nSTRONG CONTENDERS (60-90% playoff odds):")
        for team, result in strong:
            lines.append(f"  {team}: {result.mean_wins:.0f} wins, {result.playoff_pct:.0f}% playoff")

    # Bubble (30-60%)
    bubble = [(t, r) for t, r in all_teams if 30 <= r.playoff_pct < 60]
    if bubble:
        lines.append("\nBUBBLE TEAMS (30-60% playoff odds):")
        for team, result in bubble:
            lines.append(f"  {team}: {result.mean_wins:.0f} wins, {result.playoff_pct:.0f}% playoff")

    # Longshots (10-30%)
    longshots = [(t, r) for t, r in all_teams if 10 <= r.playoff_pct < 30]
    if longshots:
        lines.append("\nLONGSHOTS (10-30% playoff odds):")
        for team, result in longshots:
            lines.append(f"  {team}: {result.mean_wins:.0f} wins, {result.playoff_pct:.0f}% playoff")

    # Rebuilding (<10%)
    rebuilding = [(t, r) for t, r in all_teams if r.playoff_pct < 10]
    if rebuilding:
        lines.append("\nREBUILDING (<10% playoff odds):")
        for team, result in rebuilding:
            lines.append(f"  {team}: {result.mean_wins:.0f} wins, {result.playoff_pct:.1f}% playoff")

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
    team_projection: "Optional[TeamProjectionSet]" = None,
) -> str:
    """Generate a detailed report for a single team.

    Args:
        team: Team abbreviation.
        simulation: Completed season simulation.
        team_projection: Optional detailed projection with player breakdowns.

    Returns:
        Team report as a string.
    """
    if team not in simulation.team_results:
        return f"Team {team} not found in simulation results"

    result = simulation.team_results[team]
    full_name = TEAM_FULL_NAMES.get(team, team)
    division = TEAM_TO_DIVISION.get(team, "Unknown")

    lines = []
    lines.append("")
    lines.append("=" * 50)
    lines.append(f"  {full_name.upper()}".center(50))
    lines.append(f"  ({team} - {division})".center(50))
    lines.append("=" * 50)

    lines.append(f"\nProjected Record: {result.mean_wins:.0f}-{162 - result.mean_wins:.0f}")
    lines.append(f"Win Probability: 50% chance of {result.median_wins:.0f}+ wins")

    lines.append(f"\nWIN DISTRIBUTION:")
    lines.append(f"  Best case (90th pct):  {result.wins_90th_pct:.0f} wins")
    lines.append(f"  Optimistic (75th pct): {result.wins_75th_pct:.0f} wins")
    lines.append(f"  Expected (median):     {result.median_wins:.0f} wins")
    lines.append(f"  Pessimistic (25th):    {result.wins_25th_pct:.0f} wins")
    lines.append(f"  Worst case (10th pct): {result.wins_10th_pct:.0f} wins")

    lines.append(f"\nPLAYOFF ODDS:")
    lines.append(f"  Win Division:    {result.division_winner_pct:>5.1f}%")
    lines.append(f"  Wild Card:       {result.wild_card_pct:>5.1f}%")
    lines.append(f"  Make Playoffs:   {result.playoff_pct:>5.1f}%")
    lines.append(f"  Miss Playoffs:   {100 - result.playoff_pct:>5.1f}%")
    lines.append(f"  Last in Division:{result.last_place_pct:>5.1f}%")

    lines.append(f"\nMILESTONE PROBABILITIES:")
    lines.append(f"  90+ wins:   {result.wins_90_plus_pct:>5.1f}%")
    lines.append(f"  100+ wins:  {result.wins_100_plus_pct:>5.1f}%")
    lines.append(f"  .500+ (81): {result.wins_over_500_pct:>5.1f}%")

    # Add roster breakdown if available
    if team_projection:
        lines.append(f"\nROSTER STRENGTH:")
        lines.append(f"  Proj RS:        {team_projection.projected_runs_scored:>6.0f}")
        lines.append(f"  Proj RA:        {team_projection.projected_runs_allowed:>6.0f}")
        lines.append(f"  Run Diff:       {team_projection.projected_runs_scored - team_projection.projected_runs_allowed:>+6.0f}")

        # Strength of Schedule
        if hasattr(team_projection, 'sos') and team_projection.sos != 0.500:
            lines.append(f"\nSTRENGTH OF SCHEDULE:")
            lines.append(f"  SOS:            {team_projection.sos:>6.3f}")
            sos_desc = "harder" if team_projection.sos > 0.500 else "easier"
            lines.append(f"  Schedule:       {sos_desc} than average")
            lines.append(f"  Win Adjustment: {team_projection.sos_win_adjustment:>+5.1f}")
            if team_projection.projected_wins_pre_sos > 0:
                lines.append(f"  Pre-SOS Wins:   {team_projection.projected_wins_pre_sos:>6.1f}")
                lines.append(f"  Post-SOS Wins:  {team_projection.projected_wins:>6.1f}")

        # Top batters
        if team_projection.batters:
            lines.append(f"\nTOP PROJECTED BATTERS:")
            top_batters = sorted(
                team_projection.batters,
                key=lambda b: b.projected_batting_runs,
                reverse=True
            )[:5]
            for b in top_batters:
                lines.append(
                    f"  {b.name[:20]:<20} "
                    f"wRC+: {b.projected_wrc_plus:>3.0f}  "
                    f"PA: {b.projected_pa:>4.0f}  "
                    f"Runs: {b.projected_batting_runs:>+5.1f}"
                )

        # Top pitchers
        if team_projection.pitchers:
            lines.append(f"\nTOP PROJECTED PITCHERS:")
            top_pitchers = sorted(
                team_projection.pitchers,
                key=lambda p: p.projected_pitching_runs,
                reverse=True
            )[:5]
            for p in top_pitchers:
                lines.append(
                    f"  {p.name[:20]:<20} "
                    f"FIP: {p.projected_fip:>4.2f}  "
                    f"IP: {p.projected_ip:>5.1f}  "
                    f"Runs: {p.projected_pitching_runs:>+5.1f}"
                )

    return "\n".join(lines)


def export_to_csv(
    simulation: SeasonSimulation,
    output_path: Union[Path, str],
    team_projections: Optional[Dict[str, "TeamProjectionSet"]] = None,
) -> None:
    """Export simulation results to CSV.

    Args:
        simulation: Completed season simulation.
        output_path: Path for output CSV.
        team_projections: Optional team projections with SOS data.
    """
    rows = []
    for team, result in simulation.team_results.items():
        row = {
            "Team": team,
            "Full Name": TEAM_FULL_NAMES.get(team, team),
            "Division": TEAM_TO_DIVISION.get(team, ""),
            "Projected Wins": round(result.mean_wins, 1),
            "Projected Losses": round(162 - result.mean_wins, 1),
            "Median Wins": round(result.median_wins, 1),
            "Std Dev": round(result.std_wins, 2),
            "10th Percentile": round(result.wins_10th_pct, 1),
            "90th Percentile": round(result.wins_90th_pct, 1),
            "Division Winner %": round(result.division_winner_pct, 1),
            "Wild Card %": round(result.wild_card_pct, 1),
            "Playoff %": round(result.playoff_pct, 1),
            "90+ Wins %": round(result.wins_90_plus_pct, 1),
            "100+ Wins %": round(result.wins_100_plus_pct, 1),
            "Over .500 %": round(result.wins_over_500_pct, 1),
            "Last Place %": round(result.last_place_pct, 1),
        }

        # Add SOS data if available
        if team_projections and team in team_projections:
            proj = team_projections[team]
            if hasattr(proj, 'sos'):
                row["SOS"] = round(proj.sos, 3)
                row["SOS Win Adj"] = round(proj.sos_win_adjustment, 1)

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("Projected Wins", ascending=False).reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Results exported to {output_path}")


def export_player_projections_csv(
    team_projections: Dict[str, "TeamProjectionSet"],
    output_dir: Union[Path, str],
) -> Tuple[Path, Path]:
    """Export player projections to CSV files.

    Args:
        team_projections: Dict of team projections.
        output_dir: Directory for output files.

    Returns:
        Tuple of (batters_path, pitchers_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Batters
    batter_rows = []
    for team, proj_set in team_projections.items():
        for b in proj_set.batters:
            batter_rows.append({
                "Team": team,
                "Name": b.name,
                "Age": b.age,
                "PA": round(b.projected_pa),
                "AVG": round(b.projected_avg, 3),
                "OBP": round(b.projected_obp, 3),
                "SLG": round(b.projected_slg, 3),
                "wRC+": round(b.projected_wrc_plus),
                "HR": round(b.projected_hr),
                "RBI": round(b.projected_rbi),
                "Runs": round(b.projected_runs),
                "Batting Runs": round(b.projected_batting_runs, 1),
            })

    batters_df = pd.DataFrame(batter_rows)
    batters_path = output_dir / "projected_batters.csv"
    batters_df.to_csv(batters_path, index=False)

    # Pitchers
    pitcher_rows = []
    for team, proj_set in team_projections.items():
        for p in proj_set.pitchers:
            pitcher_rows.append({
                "Team": team,
                "Name": p.name,
                "Age": p.age,
                "Role": p.role,
                "IP": round(p.projected_ip, 1),
                "ERA": round(p.projected_era, 2),
                "FIP": round(p.projected_fip, 2),
                "WHIP": round(p.projected_whip, 2),
                "K/9": round(p.projected_k_per_9, 1),
                "K": round(p.projected_strikeouts),
                "W": round(p.projected_wins),
                "SV": round(p.projected_saves),
                "Pitching Runs": round(p.projected_pitching_runs, 1),
            })

    pitchers_df = pd.DataFrame(pitcher_rows)
    pitchers_path = output_dir / "projected_pitchers.csv"
    pitchers_df.to_csv(pitchers_path, index=False)

    logger.info(f"Exported {len(batter_rows)} batters to {batters_path}")
    logger.info(f"Exported {len(pitcher_rows)} pitchers to {pitchers_path}")

    return batters_path, pitchers_path


def generate_sos_report(
    team_projections: Dict[str, "TeamProjectionSet"],
) -> str:
    """Generate a Strength of Schedule report.

    Args:
        team_projections: Dict of team projections with SOS data.

    Returns:
        Formatted SOS report string.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("  STRENGTH OF SCHEDULE REPORT".center(60))
    lines.append("=" * 60)
    lines.append("")
    lines.append("SOS > 0.500 = harder schedule (face better teams)")
    lines.append("SOS < 0.500 = easier schedule (face weaker teams)")
    lines.append("")

    # Sort teams by SOS (hardest to easiest)
    teams_with_sos = [
        (team, proj)
        for team, proj in team_projections.items()
        if hasattr(proj, 'sos') and proj.sos != 0.500
    ]

    if not teams_with_sos:
        return "No SOS data available"

    teams_with_sos.sort(key=lambda x: x[1].sos, reverse=True)

    lines.append(f"{'Rank':<5} {'Team':<5} {'SOS':>7} {'Adj':>6} {'Pre-SOS W':>10} {'Post-SOS W':>11}")
    lines.append("-" * 60)

    for i, (team, proj) in enumerate(teams_with_sos, 1):
        pre_sos = proj.projected_wins_pre_sos if proj.projected_wins_pre_sos > 0 else proj.projected_wins
        post_sos = proj.projected_wins
        lines.append(
            f"{i:<5} {team:<5} {proj.sos:>7.3f} {proj.sos_win_adjustment:>+5.1f} "
            f"{pre_sos:>10.1f} {post_sos:>11.1f}"
        )

    lines.append("")
    lines.append("-" * 60)

    # Division breakdown
    lines.append("\nSCHEDULE DIFFICULTY BY DIVISION:")
    lines.append("-" * 40)

    for div in sorted(DIVISIONS.keys()):
        div_teams = [
            (team, proj)
            for team, proj in team_projections.items()
            if TEAM_TO_DIVISION.get(team) == div and hasattr(proj, 'sos')
        ]
        if div_teams:
            avg_sos = sum(p.sos for _, p in div_teams) / len(div_teams)
            hardest_team = max(div_teams, key=lambda x: x[1].sos)
            easiest_team = min(div_teams, key=lambda x: x[1].sos)
            lines.append(f"\n{div}:")
            lines.append(f"  Avg SOS: {avg_sos:.3f}")
            lines.append(f"  Hardest: {hardest_team[0]} ({hardest_team[1].sos:.3f})")
            lines.append(f"  Easiest: {easiest_team[0]} ({easiest_team[1].sos:.3f})")

    lines.append("")

    # Key insights
    lines.append("\nKEY INSIGHTS:")
    lines.append("-" * 40)

    # Teams with biggest adjustments
    biggest_neg = min(teams_with_sos, key=lambda x: x[1].sos_win_adjustment)
    biggest_pos = max(teams_with_sos, key=lambda x: x[1].sos_win_adjustment)

    lines.append(
        f"Toughest schedule: {biggest_neg[0]} "
        f"(SOS: {biggest_neg[1].sos:.3f}, {biggest_neg[1].sos_win_adjustment:+.1f} wins)"
    )
    lines.append(
        f"Easiest schedule:  {biggest_pos[0]} "
        f"(SOS: {biggest_pos[1].sos:.3f}, {biggest_pos[1].sos_win_adjustment:+.1f} wins)"
    )

    return "\n".join(lines)
