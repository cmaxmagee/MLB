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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """MLB Season Prediction Model - Monte Carlo simulation for projecting season outcomes."""
    pass


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

    # Fetch batting stats
    click.echo("Fetching batting statistics...")
    batting = fetch_batting_stats(year, qual=qual_val)
    click.echo(f"  Retrieved {len(batting)} batters")

    # Fetch pitching stats
    click.echo("Fetching pitching statistics...")
    pitching = fetch_pitching_stats(year, qual=qual_val)
    click.echo(f"  Retrieved {len(pitching)} pitchers")

    # Fetch rosters
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


@cli.command()
@click.option("--year", "-y", default=2025, help="Projection year")
@click.option("--iterations", "-n", default=10000, help="Number of simulations")
@click.option("--roster-changes", "-r", type=click.Path(exists=True), help="Roster changes CSV")
@click.option("--output", "-o", type=click.Path(), help="Output report path")
def project(year: int, iterations: int, roster_changes: str | None, output: str | None):
    """Run season projections (Phase 4+ - not yet implemented)."""
    click.echo("Season projection not yet fully implemented.")
    click.echo("Current status:")
    click.echo("  [x] Phase 1: Data pipeline")
    click.echo("  [ ] Phase 2: Player projections")
    click.echo("  [ ] Phase 3: Team projections")
    click.echo("  [ ] Phase 4: Monte Carlo simulation")
    click.echo("  [ ] Phase 5: Output & polish")
    click.echo("\nRun 'python -m src.main fetch-data' to populate the data cache first.")


@cli.command()
def status():
    """Show project implementation status."""
    click.echo("\nMLB Season Prediction Model - Implementation Status\n")
    click.echo("=" * 50)

    phases = [
        ("Phase 1: Data Pipeline", True, [
            ("fetch.py - pybaseball wrappers", True),
            ("cache.py - local caching", True),
            ("roster_changes.py - input parsing", True),
        ]),
        ("Phase 2: Player Projections", False, [
            ("player.py - individual projections", False),
            ("playing_time.py - PA/IP projections", False),
            ("adjustments.py - aging, parks, regression", False),
        ]),
        ("Phase 3: Team Projections", False, [
            ("team.py - aggregate projections", False),
            ("pitching_staff.py - innings allocation", False),
        ]),
        ("Phase 4: Monte Carlo Simulation", False, [
            ("monte_carlo.py - run simulations", False),
            ("standings.py - playoff odds", False),
        ]),
        ("Phase 5: Output & Polish", False, [
            ("reports.py - text reports", False),
            ("visualize.py - charts", False),
        ]),
    ]

    for phase_name, phase_complete, items in phases:
        status = "[x]" if phase_complete else "[ ]"
        click.echo(f"\n{status} {phase_name}")
        for item_name, item_complete in items:
            item_status = "[x]" if item_complete else "[ ]"
            click.echo(f"    {item_status} {item_name}")


if __name__ == "__main__":
    cli()
