"""Visualization module for projections and simulation results."""

import logging
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..simulation.monte_carlo import SeasonSimulation

logger = logging.getLogger(__name__)

# Style settings
plt.style.use("seaborn-v0_8-whitegrid")
TEAM_COLORS = {
    "NYY": "#003087", "BOS": "#BD3039", "TOR": "#134A8E", "TBR": "#092C5C", "BAL": "#DF4601",
    "CLE": "#00385D", "MIN": "#002B5C", "CHW": "#27251F", "DET": "#0C2340", "KCR": "#004687",
    "HOU": "#002D62", "TEX": "#003278", "SEA": "#0C2C56", "LAA": "#BA0021", "OAK": "#003831",
    "ATL": "#CE1141", "PHI": "#E81828", "NYM": "#002D72", "MIA": "#00A3E0", "WSN": "#AB0003",
    "MIL": "#12284B", "CHC": "#0E3386", "STL": "#C41E3A", "PIT": "#27251F", "CIN": "#C6011F",
    "LAD": "#005A9C", "SDP": "#2F241D", "ARI": "#A71930", "SFG": "#FD5A1E", "COL": "#33006F",
}


def plot_win_distribution(
    team: str,
    simulation: SeasonSimulation,
    save_path: Optional[Union[Path, str]] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot win distribution histogram for a team.

    Args:
        team: Team abbreviation.
        simulation: Completed season simulation.
        save_path: Optional path to save figure.
        show: Whether to display the plot.

    Returns:
        Matplotlib figure.
    """
    if team not in simulation.team_results:
        raise ValueError(f"Team {team} not found in simulation")

    result = simulation.team_results[team]
    wins = result.win_distribution

    fig, ax = plt.subplots(figsize=(10, 6))

    color = TEAM_COLORS.get(team, "#333333")

    # Histogram
    ax.hist(wins, bins=30, color=color, alpha=0.7, edgecolor="white")

    # Add mean and percentile lines
    ax.axvline(result.mean_wins, color="black", linestyle="-", linewidth=2, label=f"Mean: {result.mean_wins:.1f}")
    ax.axvline(result.wins_10th_pct, color="gray", linestyle="--", linewidth=1.5, label=f"10th pct: {result.wins_10th_pct:.0f}")
    ax.axvline(result.wins_90th_pct, color="gray", linestyle="--", linewidth=1.5, label=f"90th pct: {result.wins_90th_pct:.0f}")

    ax.set_xlabel("Wins", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"{team} Projected Win Distribution ({simulation.iterations:,} simulations)", fontsize=14)
    ax.legend(loc="upper right")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")

    if show:
        plt.show()

    return fig


def plot_division_standings(
    division: str,
    simulation: SeasonSimulation,
    save_path: Optional[Union[Path, str]] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot projected standings for a division.

    Args:
        division: Division name (e.g., "AL East").
        simulation: Completed season simulation.
        save_path: Optional path to save figure.
        show: Whether to display the plot.

    Returns:
        Matplotlib figure.
    """
    from ..projections.team import DIVISIONS

    if division not in DIVISIONS:
        raise ValueError(f"Unknown division: {division}")

    teams = DIVISIONS[division]
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(teams))

    # Get results sorted by wins
    team_data = []
    for team in teams:
        if team in simulation.team_results:
            result = simulation.team_results[team]
            team_data.append({
                "team": team,
                "mean": result.mean_wins,
                "low": result.wins_10th_pct,
                "high": result.wins_90th_pct,
            })

    team_data.sort(key=lambda x: x["mean"], reverse=True)

    for i, data in enumerate(team_data):
        color = TEAM_COLORS.get(data["team"], "#333333")
        ax.barh(i, data["mean"], color=color, alpha=0.8, height=0.6)
        ax.plot(
            [data["low"], data["high"]],
            [i, i],
            color="black",
            linewidth=2,
            marker="|",
            markersize=10,
        )

    ax.set_yticks(range(len(team_data)))
    ax.set_yticklabels([d["team"] for d in team_data])
    ax.set_xlabel("Projected Wins", fontsize=12)
    ax.set_title(f"{division} Projected Standings", fontsize=14)
    ax.invert_yaxis()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_playoff_odds(
    simulation: SeasonSimulation,
    top_n: int = 15,
    save_path: Optional[Union[Path, str]] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot playoff odds for top contenders.

    Args:
        simulation: Completed season simulation.
        top_n: Number of teams to show.
        save_path: Optional path to save figure.
        show: Whether to display the plot.

    Returns:
        Matplotlib figure.
    """
    # Sort teams by playoff odds
    team_odds = [
        (team, result.playoff_pct, result.division_winner_pct)
        for team, result in simulation.team_results.items()
    ]
    team_odds.sort(key=lambda x: x[1], reverse=True)
    team_odds = team_odds[:top_n]

    fig, ax = plt.subplots(figsize=(12, 8))

    teams = [t[0] for t in team_odds]
    playoff = [t[1] for t in team_odds]
    division = [t[2] for t in team_odds]

    y_pos = np.arange(len(teams))
    bar_height = 0.35

    # Division winner bars
    colors = [TEAM_COLORS.get(t, "#333333") for t in teams]
    ax.barh(y_pos - bar_height/2, division, bar_height, label="Division Winner", color=colors, alpha=0.9)

    # Wild card portion (playoff - division)
    wild_card = [p - d for p, d in zip(playoff, division)]
    ax.barh(y_pos + bar_height/2, wild_card, bar_height, label="Wild Card", color=colors, alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(teams)
    ax.set_xlabel("Probability (%)", fontsize=12)
    ax.set_title("Playoff Odds", fontsize=14)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig
