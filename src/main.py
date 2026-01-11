#!/usr/bin/env python3
"""CLI entry point for MLB Season Prediction Model."""

import logging
import sys
from pathlib import Path

import click

from .data import DataCache, fetch_batting_stats, fetch_pitching_stats, fetch_team_rosters
from .data.roster_changes import create_sample_roster_changes_csv, load_roster_changes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """MLB Season Prediction Model - Monte Carlo simulation for projecting season outcomes."""
    pass


# =============================================================================
# Data Commands
# =============================================================================

@cli.command()
@click.option("--year", "-y", default=2024, help="Season year to fetch")
@click.option("--qual", "-q", default="y", help="Qualification threshold ('y' for qualified, or integer)")
def fetch_data(year: int, qual: str):
    """Fetch player statistics from FanGraphs."""
    click.echo(f"Fetching {year} season data...")

    try:
        qual_val = qual if qual == "y" else int(qual)
    except ValueError:
        click.echo(f"Invalid qual value: {qual}", err=True)
        sys.exit(1)

    click.echo("Fetching batting statistics...")
    batting = fetch_batting_stats(year, qual=qual_val)
    click.echo(f"  Retrieved {len(batting)} batters")

    click.echo("Fetching pitching statistics...")
    pitching = fetch_pitching_stats(year, qual=qual_val)
    click.echo(f"  Retrieved {len(pitching)} pitchers")

    click.echo("Building team rosters...")
    rosters = fetch_team_rosters(year)
    click.echo(f"  Retrieved {len(rosters)} roster entries")

    click.echo("\nData fetched and cached successfully!")


@cli.command()
@click.option("--start-year", "-s", default=2022, help="First year for historical data")
@click.option("--end-year", "-e", default=2024, help="Last year for historical data")
def fetch_historical(start_year: int, end_year: int):
    """Fetch multiple years of historical data."""
    click.echo(f"Fetching data for {start_year}-{end_year}...")

    for year in range(start_year, end_year + 1):
        click.echo(f"\n{year}:")

        batting = fetch_batting_stats(year, qual=1)
        click.echo(f"  Batting: {len(batting)} players")

        pitching = fetch_pitching_stats(year, qual=1)
        click.echo(f"  Pitching: {len(pitching)} players")

    click.echo("\nHistorical data cached successfully!")


@cli.command()
def cache_stats():
    """Show cache statistics."""
    cache = DataCache()
    stats = cache.stats()

    click.echo("Cache Statistics:")
    click.echo(f"  Directory: {stats['cache_dir']}")
    click.echo(f"  Entries: {stats['entries']}")
    click.echo(f"  Total size: {stats['total_size_mb']} MB")


@cli.command()
@click.option("--older-than", type=int, help="Only clear entries older than N days")
@click.confirmation_option(prompt="Are you sure you want to clear the cache?")
def clear_cache(older_than: int | None):
    """Clear cached data."""
    cache = DataCache()
    cleared = cache.clear(older_than_days=older_than)
    click.echo(f"Cleared {cleared} cache entries")


# =============================================================================
# Roster Commands
# =============================================================================

@cli.command()
@click.argument("output_path", type=click.Path())
def create_roster_template(output_path: str):
    """Create a sample roster changes CSV template."""
    create_sample_roster_changes_csv(output_path)
    click.echo(f"Created roster changes template at: {output_path}")


@cli.command()
@click.argument("roster_file", type=click.Path(exists=True))
def load_changes(roster_file: str):
    """Load and validate roster changes from a CSV file."""
    changes = load_roster_changes(roster_file)
    click.echo(f"Loaded {len(changes)} roster changes:")

    for change in changes:
        click.echo(f"  - {change.player_name}: {change.change_type.value} "
                   f"({change.from_team or 'FA'} -> {change.to_team or 'N/A'})")


# =============================================================================
# Projection Commands
# =============================================================================

@cli.command()
@click.option("--year", "-y", default=2025, help="Projection year")
@click.option("--iterations", "-n", default=10000, help="Number of simulations")
@click.option("--roster-changes", "-r", type=click.Path(exists=True), help="Roster changes CSV")
@click.option("--output-dir", "-o", type=click.Path(), default="output", help="Output directory")
@click.option("--player-variance/--no-player-variance", default=False, help="Use player-level variance")
@click.option("--seed", type=int, help="Random seed for reproducibility")
def project(
    year: int,
    iterations: int,
    roster_changes: str | None,
    output_dir: str,
    player_variance: bool,
    seed: int | None,
):
    """Run full season projections with Monte Carlo simulation."""
    from .projections import Projector, ProjectionConfig
    from .simulation.monte_carlo import run_simulation, run_simulation_with_player_variance
    from .output.reports import generate_season_report, export_to_csv, export_player_projections_csv

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    click.echo(f"\n{'='*60}")
    click.echo(f"  MLB {year} Season Projection")
    click.echo(f"{'='*60}")
    click.echo(f"Simulations: {iterations:,}")
    click.echo(f"Player variance: {'Yes' if player_variance else 'No'}")
    if seed:
        click.echo(f"Random seed: {seed}")
    click.echo()

    # Configure projections
    config = ProjectionConfig(
        projection_year=year,
        historical_years=3,
    )

    # Generate projections
    click.echo("Generating player projections...")
    projector = Projector(config)

    # Load roster changes if provided
    changes = None
    if roster_changes:
        changes = load_roster_changes(roster_changes)
        click.echo(f"  Loaded {len(changes)} roster changes")

    team_projections = projector.project_all_teams(roster_changes=changes)
    click.echo(f"  Projected {len(team_projections)} teams")

    # Run simulation
    click.echo("\nRunning Monte Carlo simulation...")
    if player_variance:
        simulation = run_simulation_with_player_variance(
            team_projections,
            iterations=iterations,
            random_seed=seed,
        )
    else:
        simulation = run_simulation(
            team_projections,
            iterations=iterations,
            random_seed=seed,
        )

    # Generate outputs
    click.echo("\nGenerating reports...")

    # Text report
    report = generate_season_report(simulation, projection_year=year)
    report_path = output_path / f"projection_{year}.txt"
    with open(report_path, "w") as f:
        f.write(report)
    click.echo(f"  Report: {report_path}")

    # CSV export
    csv_path = output_path / f"standings_{year}.csv"
    export_to_csv(simulation, csv_path)
    click.echo(f"  Standings CSV: {csv_path}")

    # Player projections CSV
    batters_path, pitchers_path = export_player_projections_csv(team_projections, output_path)
    click.echo(f"  Batters CSV: {batters_path}")
    click.echo(f"  Pitchers CSV: {pitchers_path}")

    # Print summary
    click.echo(f"\n{'='*60}")
    click.echo("  TOP 10 PROJECTED TEAMS")
    click.echo(f"{'='*60}")

    playoff_df = simulation.get_playoff_odds_df()
    for i, row in playoff_df.head(10).iterrows():
        click.echo(
            f"  {i+1:2}. {row['Team']:<5} {row['Proj W']:>5.1f} W  "
            f"({row['10th']:.0f}-{row['90th']:.0f})  "
            f"Playoff: {row['Playoff %']:>5.1f}%"
        )

    click.echo(f"\nFull report saved to: {report_path}")


@cli.command()
@click.argument("team")
@click.option("--year", "-y", default=2025, help="Projection year")
@click.option("--iterations", "-n", default=10000, help="Number of simulations")
def project_team(team: str, year: int, iterations: int):
    """Generate detailed projection for a single team."""
    from .projections import Projector, ProjectionConfig
    from .simulation.monte_carlo import run_simulation
    from .output.reports import generate_team_report

    team = team.upper()

    config = ProjectionConfig(projection_year=year)
    projector = Projector(config)

    click.echo(f"Generating {year} projection for {team}...")

    team_projections = projector.project_all_teams()

    if team not in team_projections:
        click.echo(f"Team '{team}' not found. Valid teams:", err=True)
        for t in sorted(team_projections.keys()):
            click.echo(f"  {t}", err=True)
        sys.exit(1)

    click.echo("Running simulation...")
    simulation = run_simulation(team_projections, iterations=iterations)

    report = generate_team_report(team, simulation, team_projections.get(team))
    click.echo(report)


@cli.command()
@click.option("--year", "-y", default=2025, help="Projection year")
@click.option("--output", "-o", type=click.Path(), help="Output PNG path")
def plot_standings(year: int, output: str | None):
    """Generate standings visualization."""
    from .projections import Projector, ProjectionConfig
    from .simulation.monte_carlo import run_simulation
    from .output.visualize import plot_playoff_odds

    config = ProjectionConfig(projection_year=year)
    projector = Projector(config)

    click.echo(f"Generating {year} projections...")
    team_projections = projector.project_all_teams()

    click.echo("Running simulation...")
    simulation = run_simulation(team_projections, iterations=5000)

    click.echo("Generating plot...")
    fig = plot_playoff_odds(simulation, save_path=output, show=output is None)

    if output:
        click.echo(f"Saved to: {output}")


# =============================================================================
# Status Command
# =============================================================================

@cli.command()
def status():
    """Show project implementation status."""
    click.echo("\nMLB Season Prediction Model - Implementation Status\n")
    click.echo("=" * 55)

    phases = [
        ("Phase 1: Data Pipeline", True, [
            ("fetch.py - pybaseball wrappers", True),
            ("cache.py - local caching", True),
            ("roster_changes.py - input parsing", True),
        ]),
        ("Phase 2: Player Projections", True, [
            ("player.py - individual projections", True),
            ("playing_time.py - PA/IP projections", True),
            ("adjustments.py - aging, parks, regression", True),
            ("projector.py - orchestration", True),
        ]),
        ("Phase 3: Team Projections", True, [
            ("team.py - aggregate projections", True),
            ("pitching_staff.py - innings allocation", True),
        ]),
        ("Phase 4: Monte Carlo Simulation", True, [
            ("monte_carlo.py - run simulations", True),
            ("monte_carlo.py - player-level variance", True),
            ("standings.py - playoff odds", True),
        ]),
        ("Phase 5: Output & Polish", True, [
            ("reports.py - text reports", True),
            ("visualize.py - charts", True),
            ("CLI - full workflow", True),
        ]),
    ]

    for phase_name, phase_complete, items in phases:
        status_icon = "[x]" if phase_complete else "[ ]"
        click.echo(f"\n{status_icon} {phase_name}")
        for item_name, item_complete in items:
            item_status = "[x]" if item_complete else "[ ]"
            click.echo(f"    {item_status} {item_name}")

    click.echo("\n" + "=" * 55)
    click.echo("All phases complete! Run 'project --help' to get started.")


if __name__ == "__main__":
    cli()
