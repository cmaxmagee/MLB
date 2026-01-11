"""Individual player projection module.

Projects player performance based on weighted historical averages,
regression to the mean, and aging curves.
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from .adjustments import (
    get_aging_adjustment,
    adjust_for_park_change,
    get_league_average,
    LEAGUE_AVERAGES,
)

logger = logging.getLogger(__name__)


# Regression constants for common stats (approximate PA/IP needed for reliability)
REGRESSION_CONSTANTS = {
    # Batting (PA-based)
    "AVG": 910,
    "OBP": 460,
    "SLG": 320,
    "ISO": 160,
    "BABIP": 820,
    "K%": 60,
    "BB%": 120,
    "wRC+": 400,
    "wOBA": 340,
    # Pitching (IP-based, converted to batters faced ~4.3 per IP)
    "ERA": 140,
    "FIP": 50,
    "xFIP": 50,
    "K/9": 17,
    "BB/9": 40,
    "HR/9": 70,
    "WHIP": 60,
    "BABIP_P": 470,
    "LOB%": 140,
}

# Default year weights: most recent year weighted most heavily
DEFAULT_YEAR_WEIGHTS = [5, 4, 3]  # Last 3 years with 5:4:3 ratio


@dataclass
class PlayerProjection:
    """Projected stats for a single player."""

    player_id: int
    name: str
    team: str
    position: Literal["Batter", "Pitcher"]
    age: int = 0

    # Projected playing time
    projected_pa: float = 0.0  # Plate appearances (batters)
    projected_ip: float = 0.0  # Innings pitched (pitchers)
    projected_games: float = 0.0

    # Projected rate stats (batters)
    projected_avg: float = 0.248
    projected_obp: float = 0.315
    projected_slg: float = 0.399
    projected_woba: float = 0.310
    projected_wrc_plus: float = 100.0
    projected_iso: float = 0.151
    projected_bb_pct: float = 0.085
    projected_k_pct: float = 0.227

    # Projected rate stats (pitchers)
    projected_era: float = 4.17
    projected_fip: float = 4.12
    projected_xfip: float = 4.15
    projected_whip: float = 1.28
    projected_k_per_9: float = 8.6
    projected_bb_per_9: float = 3.2
    projected_hr_per_9: float = 1.22

    # Projected counting stats (derived from rate * playing time)
    projected_hr: float = 0.0
    projected_rbi: float = 0.0
    projected_runs: float = 0.0
    projected_sb: float = 0.0
    projected_wins: float = 0.0  # Pitcher wins
    projected_saves: float = 0.0
    projected_strikeouts: float = 0.0  # Pitcher K's

    # Runs contribution (for team aggregation)
    projected_batting_runs: float = 0.0
    projected_baserunning_runs: float = 0.0
    projected_pitching_runs: float = 0.0

    # Uncertainty metrics
    projection_confidence: float = 1.0  # 0-1, based on sample size
    std_dev_wins: float = 0.0  # For simulation variance

    # Metadata
    seasons_of_data: int = 0
    total_pa_historical: int = 0
    total_ip_historical: float = 0.0


@dataclass
class BatterProjection(PlayerProjection):
    """Batter-specific projection with additional fields."""

    position: Literal["Batter", "Pitcher"] = "Batter"

    # Position breakdown (for multi-position players)
    positions_played: List[str] = field(default_factory=list)
    primary_position: str = "DH"


@dataclass
class PitcherProjection(PlayerProjection):
    """Pitcher-specific projection with additional fields."""

    position: Literal["Batter", "Pitcher"] = "Pitcher"

    # Role
    role: str = "SP"  # SP, RP, CL
    projected_starts: float = 0.0
    projected_relief_appearances: float = 0.0


def calculate_weighted_average(
    stats: pd.DataFrame,
    stat_col: str,
    weight_col: str = "PA",
    year_weights: Optional[List[float]] = None,
    year_col: str = "Season",
) -> Tuple[float, int]:
    """Calculate weighted average of a stat across seasons.

    Args:
        stats: DataFrame with yearly stats.
        stat_col: Column name of the stat to average.
        weight_col: Column to use for weighting (PA or IP).
        year_weights: Recency weights [most_recent, ..., oldest].
        year_col: Column containing the year.

    Returns:
        Tuple of (weighted average, total sample size used).
    """
    if year_weights is None:
        year_weights = DEFAULT_YEAR_WEIGHTS

    if stat_col not in stats.columns:
        return np.nan, 0

    # Sort by year descending and take most recent N years
    stats = stats.sort_values(year_col, ascending=False).head(len(year_weights))

    if len(stats) == 0:
        return np.nan, 0

    total_weight = 0.0
    weighted_sum = 0.0
    total_sample = 0

    for i, (_, row) in enumerate(stats.iterrows()):
        if i >= len(year_weights):
            break

        stat_value = row.get(stat_col)
        if pd.isna(stat_value):
            continue

        year_weight = year_weights[i]
        playing_time = row.get(weight_col, 1)
        if pd.isna(playing_time):
            playing_time = 1

        combined_weight = year_weight * playing_time
        weighted_sum += stat_value * combined_weight
        total_weight += combined_weight
        total_sample += int(playing_time)

    if total_weight == 0:
        return np.nan, 0

    return weighted_sum / total_weight, total_sample


def regress_to_mean(
    observed_value: float,
    sample_size: int,
    league_mean: float,
    regression_constant: int,
) -> float:
    """Apply regression to the mean for a stat.

    Uses the formula: regressed = (observed * n + mean * k) / (n + k)
    where k is the regression constant (sample size at which we trust
    the observed value 50%).

    Args:
        observed_value: Player's observed rate stat.
        sample_size: Number of PA/IP/etc.
        league_mean: League average for this stat.
        regression_constant: Sample size for 50% regression.

    Returns:
        Regressed value.
    """
    if pd.isna(observed_value):
        return league_mean

    if sample_size <= 0:
        return league_mean

    weight = sample_size / (sample_size + regression_constant)
    return (observed_value * weight) + (league_mean * (1 - weight))


def calculate_projection_confidence(
    sample_size: int,
    years_of_data: int,
    is_pitcher: bool = False,
) -> float:
    """Calculate confidence in a projection (0-1 scale).

    Higher confidence = more reliable projection, less regression needed.

    Args:
        sample_size: Total PA or IP.
        years_of_data: Number of seasons of data.
        is_pitcher: Whether this is a pitcher projection.

    Returns:
        Confidence score from 0 to 1.
    """
    # Base confidence from sample size
    if is_pitcher:
        # ~500 IP for full confidence
        sample_confidence = min(1.0, sample_size / 500)
    else:
        # ~1500 PA for full confidence
        sample_confidence = min(1.0, sample_size / 1500)

    # Boost for multiple years of data
    year_boost = min(0.2, years_of_data * 0.05)

    return min(1.0, sample_confidence + year_boost)


def project_batter(
    player_id: int,
    historical_stats: pd.DataFrame,
    projection_year: int,
    new_team: Optional[str] = None,
    year_weights: Optional[List[float]] = None,
) -> BatterProjection:
    """Generate projection for a batter.

    Args:
        player_id: FanGraphs player ID.
        historical_stats: DataFrame with player's yearly batting stats.
        projection_year: Year to project for.
        new_team: New team if player changed teams (for park adjustment).
        year_weights: Custom weights for recent seasons.

    Returns:
        BatterProjection with projected stats.
    """
    if len(historical_stats) == 0:
        raise ValueError(f"No historical stats for player {player_id}")

    # Get most recent info
    most_recent = historical_stats.sort_values("Season", ascending=False).iloc[0]
    name = most_recent.get("Name", f"Player {player_id}")
    old_team = most_recent.get("Team", "")
    team = new_team or old_team

    # Calculate age for projection year
    # Estimate birth year from most recent season age if available
    if "Age" in most_recent:
        current_age = int(most_recent["Age"])
        years_forward = projection_year - int(most_recent["Season"])
        age = current_age + years_forward
    else:
        age = 27  # Default to peak age if unknown

    # Calculate weighted averages and regress to mean
    stats_to_project = {
        "AVG": ("AVG", "PA", LEAGUE_AVERAGES["AVG"]),
        "OBP": ("OBP", "PA", LEAGUE_AVERAGES["OBP"]),
        "SLG": ("SLG", "PA", LEAGUE_AVERAGES["SLG"]),
        "wOBA": ("wOBA", "PA", LEAGUE_AVERAGES["wOBA"]),
        "wRC+": ("wRC+", "PA", 100.0),
        "ISO": ("ISO", "PA", LEAGUE_AVERAGES["ISO"]),
        "BB%": ("BB%", "PA", LEAGUE_AVERAGES["BB%"]),
        "K%": ("K%", "PA", LEAGUE_AVERAGES["K%"]),
    }

    projected_stats = {}
    total_pa = 0
    seasons = len(historical_stats["Season"].unique())

    for stat_name, (col, weight_col, league_avg) in stats_to_project.items():
        if col not in historical_stats.columns:
            projected_stats[stat_name] = league_avg
            continue

        weighted_avg, sample_size = calculate_weighted_average(
            historical_stats, col, weight_col, year_weights
        )
        total_pa = max(total_pa, sample_size)

        # Regress to mean
        reg_constant = REGRESSION_CONSTANTS.get(stat_name, 400)
        regressed = regress_to_mean(weighted_avg, sample_size, league_avg, reg_constant)

        projected_stats[stat_name] = regressed

    # Apply aging adjustment
    years_forward = projection_year - int(most_recent["Season"])
    if years_forward > 0 and "Age" in most_recent:
        aging_factor = get_aging_adjustment(
            current_age=int(most_recent["Age"]),
            projection_age=age,
            player_type="batter",
        )
        # Adjust rate stats by aging factor
        for stat in ["AVG", "OBP", "SLG", "wOBA", "ISO"]:
            if stat in projected_stats:
                # Regress the adjustment toward 1.0 for uncertainty
                adj = 1.0 + (aging_factor - 1.0) * 0.7
                projected_stats[stat] *= adj

        # wRC+ adjustment
        if "wRC+" in projected_stats:
            wrc_adj = 100 + (projected_stats["wRC+"] - 100) * aging_factor
            projected_stats["wRC+"] = wrc_adj

    # Apply park factor adjustment if changing teams
    if new_team and new_team != old_team:
        for stat in ["AVG", "OBP", "SLG", "ISO"]:
            if stat in projected_stats:
                projected_stats[stat] = adjust_for_park_change(
                    projected_stats[stat], old_team, new_team, "batting"
                )

    # Calculate batting runs above average
    # Using wRC+ as primary metric: (wRC+ - 100) / 100 * PA * 0.12
    wrc_plus = projected_stats.get("wRC+", 100)
    estimated_pa = 550  # Will be updated by playing_time module
    batting_runs = (wrc_plus - 100) / 100 * estimated_pa * 0.12

    # Calculate confidence
    confidence = calculate_projection_confidence(total_pa, seasons, is_pitcher=False)

    projection = BatterProjection(
        player_id=player_id,
        name=name,
        team=team,
        age=age,
        projected_avg=projected_stats.get("AVG", 0.248),
        projected_obp=projected_stats.get("OBP", 0.315),
        projected_slg=projected_stats.get("SLG", 0.399),
        projected_woba=projected_stats.get("wOBA", 0.310),
        projected_wrc_plus=projected_stats.get("wRC+", 100),
        projected_iso=projected_stats.get("ISO", 0.151),
        projected_bb_pct=projected_stats.get("BB%", 0.085),
        projected_k_pct=projected_stats.get("K%", 0.227),
        projected_batting_runs=batting_runs,
        projection_confidence=confidence,
        seasons_of_data=seasons,
        total_pa_historical=total_pa,
    )

    return projection


def project_pitcher(
    player_id: int,
    historical_stats: pd.DataFrame,
    projection_year: int,
    new_team: Optional[str] = None,
    year_weights: Optional[List[float]] = None,
) -> PitcherProjection:
    """Generate projection for a pitcher.

    Args:
        player_id: FanGraphs player ID.
        historical_stats: DataFrame with player's yearly pitching stats.
        projection_year: Year to project for.
        new_team: New team if player changed teams (for park adjustment).
        year_weights: Custom weights for recent seasons.

    Returns:
        PitcherProjection with projected stats.
    """
    if len(historical_stats) == 0:
        raise ValueError(f"No historical stats for player {player_id}")

    # Get most recent info
    most_recent = historical_stats.sort_values("Season", ascending=False).iloc[0]
    name = most_recent.get("Name", f"Player {player_id}")
    old_team = most_recent.get("Team", "")
    team = new_team or old_team

    # Determine role based on games started
    avg_gs = historical_stats.get("GS", pd.Series([0])).mean()
    avg_sv = historical_stats.get("SV", pd.Series([0])).mean()
    if avg_gs >= 15:
        role = "SP"
    elif avg_sv >= 10:
        role = "CL"
    else:
        role = "RP"

    # Calculate age
    if "Age" in most_recent:
        current_age = int(most_recent["Age"])
        years_forward = projection_year - int(most_recent["Season"])
        age = current_age + years_forward
    else:
        age = 27

    # Stats to project
    stats_to_project = {
        "ERA": ("ERA", "IP", LEAGUE_AVERAGES["ERA"]),
        "FIP": ("FIP", "IP", LEAGUE_AVERAGES["FIP"]),
        "xFIP": ("xFIP", "IP", LEAGUE_AVERAGES["xFIP"]),
        "WHIP": ("WHIP", "IP", LEAGUE_AVERAGES["WHIP"]),
        "K/9": ("K/9", "IP", LEAGUE_AVERAGES["K/9"]),
        "BB/9": ("BB/9", "IP", LEAGUE_AVERAGES["BB/9"]),
        "HR/9": ("HR/9", "IP", LEAGUE_AVERAGES["HR/9"]),
    }

    projected_stats = {}
    total_ip = 0.0
    seasons = len(historical_stats["Season"].unique())

    for stat_name, (col, weight_col, league_avg) in stats_to_project.items():
        if col not in historical_stats.columns:
            projected_stats[stat_name] = league_avg
            continue

        weighted_avg, sample_size = calculate_weighted_average(
            historical_stats, col, weight_col, year_weights
        )
        total_ip = max(total_ip, sample_size)

        # Regress to mean (use IP as sample size proxy)
        reg_constant = REGRESSION_CONSTANTS.get(stat_name, 50)
        regressed = regress_to_mean(weighted_avg, sample_size, league_avg, reg_constant)

        projected_stats[stat_name] = regressed

    # Apply aging adjustment for pitchers
    years_forward = projection_year - int(most_recent["Season"])
    if years_forward > 0 and "Age" in most_recent:
        aging_factor = get_aging_adjustment(
            current_age=int(most_recent["Age"]),
            projection_age=age,
            player_type="pitcher",
        )
        # For pitchers, aging_factor > 1 means getting worse
        # Apply to ERA/FIP (higher = worse)
        for stat in ["ERA", "FIP", "xFIP", "WHIP", "BB/9", "HR/9"]:
            if stat in projected_stats:
                adj = 1.0 + (aging_factor - 1.0) * 0.7
                projected_stats[stat] *= adj

        # K/9 decreases with age
        if "K/9" in projected_stats:
            k_adj = 1.0 + (1.0 - aging_factor) * 0.7  # Inverse
            projected_stats["K/9"] *= k_adj

    # Apply park factor adjustment
    if new_team and new_team != old_team:
        for stat in ["ERA", "FIP"]:
            if stat in projected_stats:
                projected_stats[stat] = adjust_for_park_change(
                    projected_stats[stat], old_team, new_team, "pitching"
                )

    # Calculate pitching runs (negative = good for pitcher)
    # Using FIP: runs_above_avg = (FIP - lgFIP) / 9 * IP
    fip = projected_stats.get("FIP", 4.12)
    estimated_ip = 150 if role == "SP" else 60
    pitching_runs = -((fip - LEAGUE_AVERAGES["FIP"]) / 9 * estimated_ip)

    # Confidence
    confidence = calculate_projection_confidence(int(total_ip), seasons, is_pitcher=True)

    projection = PitcherProjection(
        player_id=player_id,
        name=name,
        team=team,
        age=age,
        role=role,
        projected_era=projected_stats.get("ERA", 4.17),
        projected_fip=projected_stats.get("FIP", 4.12),
        projected_xfip=projected_stats.get("xFIP", 4.15),
        projected_whip=projected_stats.get("WHIP", 1.28),
        projected_k_per_9=projected_stats.get("K/9", 8.6),
        projected_bb_per_9=projected_stats.get("BB/9", 3.2),
        projected_hr_per_9=projected_stats.get("HR/9", 1.22),
        projected_pitching_runs=pitching_runs,
        projection_confidence=confidence,
        seasons_of_data=seasons,
        total_ip_historical=total_ip,
    )

    return projection


def project_player(
    player_id: int,
    historical_stats: pd.DataFrame,
    projection_year: int,
    player_type: Literal["batter", "pitcher"] = "batter",
    new_team: Optional[str] = None,
    year_weights: Optional[List[float]] = None,
) -> PlayerProjection:
    """Generate projection for a player (dispatcher function).

    Args:
        player_id: FanGraphs player ID.
        historical_stats: DataFrame with player's historical stats by year.
        projection_year: Year to project for.
        player_type: "batter" or "pitcher".
        new_team: New team if player changed teams.
        year_weights: Weights for recent seasons [most_recent, ..., oldest].

    Returns:
        PlayerProjection (BatterProjection or PitcherProjection) with projected stats.
    """
    if player_type == "batter":
        return project_batter(
            player_id, historical_stats, projection_year, new_team, year_weights
        )
    else:
        return project_pitcher(
            player_id, historical_stats, projection_year, new_team, year_weights
        )


def calculate_counting_stats(
    projection: PlayerProjection,
    pa: Optional[float] = None,
    ip: Optional[float] = None,
) -> PlayerProjection:
    """Calculate counting stats from rate stats and playing time.

    Args:
        projection: PlayerProjection with rate stats.
        pa: Plate appearances (overrides projected_pa).
        ip: Innings pitched (overrides projected_ip).

    Returns:
        Updated projection with counting stats.
    """
    if isinstance(projection, BatterProjection):
        pa = pa or projection.projected_pa
        if pa > 0:
            # HR: ~3-4% of PA for good power hitters, adjusted by ISO
            hr_rate = projection.projected_iso / 4  # Rough conversion
            projection.projected_hr = pa * hr_rate

            # RBI: ~15% of PA for good run producers
            rbi_rate = 0.12 + (projection.projected_wrc_plus - 100) / 1000
            projection.projected_rbi = pa * rbi_rate

            # Runs: ~12% of PA
            runs_rate = 0.10 + (projection.projected_obp - 0.300) * 0.3
            projection.projected_runs = pa * runs_rate

            # Recalculate batting runs with actual PA
            projection.projected_batting_runs = (
                (projection.projected_wrc_plus - 100) / 100 * pa * 0.12
            )

    elif isinstance(projection, PitcherProjection):
        ip = ip or projection.projected_ip
        if ip > 0:
            # Strikeouts
            projection.projected_strikeouts = ip * projection.projected_k_per_9 / 9

            # Wins (very rough - depends on team)
            if projection.role == "SP":
                # ~60% of starts turn into decisions, ~55% of decisions are wins for avg pitcher
                starts = ip / 5.5  # ~5.5 IP per start
                projection.projected_wins = starts * 0.60 * 0.55
            else:
                projection.projected_wins = 0

            # Saves
            if projection.role == "CL":
                # Closers get ~60% of save opportunities
                projection.projected_saves = 35 * (ip / 60)

            # Recalculate pitching runs with actual IP
            projection.projected_pitching_runs = -(
                (projection.projected_fip - LEAGUE_AVERAGES["FIP"]) / 9 * ip
            )

    return projection
