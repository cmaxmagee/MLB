"""Cache simulation results to disk for read-only dashboard mode.

This allows running simulations offline and serving pre-computed results
to users without waiting for projections to generate.
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..projections.projector import TeamProjectionSet, projections_to_dataframe
from ..projections.player import BatterProjection, PitcherProjection
from ..simulation.monte_carlo import SeasonSimulation, SimulationResult

logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"


def save_simulation_results(
    projections: Dict[str, TeamProjectionSet],
    simulation: SeasonSimulation,
    output_dir: Optional[Path] = None,
    year: int = 2026,
) -> Path:
    """Save projection and simulation results to disk.

    Args:
        projections: Dictionary of team projections.
        simulation: Completed season simulation.
        output_dir: Directory to save results. Defaults to data/cache.
        year: Projection year for filename.

    Returns:
        Path to the saved results file.
    """
    output_dir = output_dir or DEFAULT_CACHE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"simulation_{year}_{timestamp}.json"
    filepath = output_dir / filename

    # Also save as "latest" for easy loading
    latest_path = output_dir / f"simulation_{year}_latest.json"

    # Build serializable data structure
    data = {
        "metadata": {
            "year": year,
            "timestamp": datetime.now().isoformat(),
            "iterations": simulation.iterations,
            "random_seed": simulation.random_seed,
        },
        "team_results": {},
        "projections": {},
        "batter_projections": [],
        "pitcher_projections": [],
    }

    # Serialize team results
    for team, result in simulation.team_results.items():
        data["team_results"][team] = {
            "team": result.team,
            "iterations": result.iterations,
            "mean_wins": result.mean_wins,
            "median_wins": result.median_wins,
            "std_wins": result.std_wins,
            "wins_10th_pct": result.wins_10th_pct,
            "wins_25th_pct": result.wins_25th_pct,
            "wins_75th_pct": result.wins_75th_pct,
            "wins_90th_pct": result.wins_90th_pct,
            "division_winner_pct": result.division_winner_pct,
            "wild_card_pct": result.wild_card_pct,
            "playoff_pct": result.playoff_pct,
            "world_series_pct": result.world_series_pct,
            "wins_over_500_pct": result.wins_over_500_pct,
            "wins_90_plus_pct": result.wins_90_plus_pct,
            "wins_100_plus_pct": result.wins_100_plus_pct,
            "last_place_pct": result.last_place_pct,
            "win_distribution": result.win_distribution.tolist(),
        }

    # Serialize team projections
    for team, proj in projections.items():
        data["projections"][team] = {
            "team": proj.team,
            "division": proj.division,
            "total_batting_runs": proj.total_batting_runs,
            "total_pitching_runs": proj.total_pitching_runs,
            "projected_runs_scored": proj.projected_runs_scored,
            "projected_runs_allowed": proj.projected_runs_allowed,
            "projected_wins": proj.projected_wins,
            "sos": proj.sos,
            "sos_win_adjustment": proj.sos_win_adjustment,
            "projected_wins_pre_sos": proj.projected_wins_pre_sos,
        }

    # Serialize player projections
    batter_df, pitcher_df = projections_to_dataframe(projections)
    data["batter_projections"] = batter_df.to_dict(orient="records")
    data["pitcher_projections"] = pitcher_df.to_dict(orient="records")

    # Write to file
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    # Also write to latest
    with open(latest_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved simulation results to {filepath}")
    logger.info(f"Updated latest results at {latest_path}")

    return filepath


def load_simulation_results(
    filepath: Optional[Path] = None,
    year: int = 2026,
    cache_dir: Optional[Path] = None,
) -> Optional[Tuple[Dict[str, TeamProjectionSet], SeasonSimulation, pd.DataFrame, pd.DataFrame, Dict[str, Any]]]:
    """Load cached simulation results from disk.

    Args:
        filepath: Specific file to load. If None, loads latest for year.
        year: Projection year to load (used if filepath is None).
        cache_dir: Cache directory to search. Defaults to data/cache.

    Returns:
        Tuple of (projections, simulation, batter_df, pitcher_df, metadata)
        or None if no cached results found.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR

    if filepath is None:
        filepath = cache_dir / f"simulation_{year}_latest.json"

    if not filepath.exists():
        logger.warning(f"No cached results found at {filepath}")
        return None

    with open(filepath, "r") as f:
        data = json.load(f)

    metadata = data["metadata"]

    # Reconstruct team results
    team_results = {}
    for team, result_data in data["team_results"].items():
        team_results[team] = SimulationResult(
            team=result_data["team"],
            iterations=result_data["iterations"],
            mean_wins=result_data["mean_wins"],
            median_wins=result_data["median_wins"],
            std_wins=result_data["std_wins"],
            wins_10th_pct=result_data["wins_10th_pct"],
            wins_25th_pct=result_data["wins_25th_pct"],
            wins_75th_pct=result_data["wins_75th_pct"],
            wins_90th_pct=result_data["wins_90th_pct"],
            division_winner_pct=result_data["division_winner_pct"],
            wild_card_pct=result_data["wild_card_pct"],
            playoff_pct=result_data["playoff_pct"],
            world_series_pct=result_data["world_series_pct"],
            wins_over_500_pct=result_data["wins_over_500_pct"],
            wins_90_plus_pct=result_data["wins_90_plus_pct"],
            wins_100_plus_pct=result_data["wins_100_plus_pct"],
            last_place_pct=result_data["last_place_pct"],
            win_distribution=np.array(result_data["win_distribution"]),
        )

    simulation = SeasonSimulation(
        iterations=metadata["iterations"],
        random_seed=metadata.get("random_seed"),
        team_results=team_results,
    )

    # Reconstruct projections (simplified - no player objects, just team-level)
    projections = {}
    for team, proj_data in data["projections"].items():
        projections[team] = TeamProjectionSet(
            team=proj_data["team"],
            division=proj_data["division"],
            total_batting_runs=proj_data["total_batting_runs"],
            total_pitching_runs=proj_data["total_pitching_runs"],
            projected_runs_scored=proj_data["projected_runs_scored"],
            projected_runs_allowed=proj_data["projected_runs_allowed"],
            projected_wins=proj_data["projected_wins"],
            sos=proj_data["sos"],
            sos_win_adjustment=proj_data["sos_win_adjustment"],
            projected_wins_pre_sos=proj_data.get("projected_wins_pre_sos", 0),
        )

    # Load player DataFrames
    batter_df = pd.DataFrame(data["batter_projections"])
    pitcher_df = pd.DataFrame(data["pitcher_projections"])

    logger.info(f"Loaded cached results from {filepath}")
    logger.info(f"  Generated: {metadata['timestamp']}")
    logger.info(f"  Iterations: {metadata['iterations']}")

    return projections, simulation, batter_df, pitcher_df, metadata


def get_available_cache_files(
    cache_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """List available cached simulation files.

    Args:
        cache_dir: Directory to search. Defaults to data/cache.

    Returns:
        List of dicts with file info (path, year, timestamp, is_latest).
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR

    if not cache_dir.exists():
        return []

    files = []
    for filepath in cache_dir.glob("simulation_*.json"):
        name = filepath.stem
        parts = name.split("_")

        if len(parts) >= 3:
            year = int(parts[1])
            is_latest = parts[2] == "latest"
            timestamp = None if is_latest else "_".join(parts[2:])

            files.append({
                "path": filepath,
                "year": year,
                "timestamp": timestamp,
                "is_latest": is_latest,
            })

    return sorted(files, key=lambda x: (x["year"], x["is_latest"]), reverse=True)
