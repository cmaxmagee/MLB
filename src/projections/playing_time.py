"""Playing time (PA/IP) projections by role.

Projects plate appearances for batters and innings pitched for pitchers
based on historical patterns, role expectations, and age-based injury risk.
"""

import logging
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Role Definitions and Baselines
# ============================================================================

@dataclass
class RoleBaseline:
    """Baseline playing time for a role."""
    pa: float = 0  # For batters
    ip: float = 0  # For pitchers
    games: float = 0
    variance: float = 0.15  # Standard deviation as fraction of baseline


# Batter roles
BATTER_ROLES = {
    "Everyday": RoleBaseline(pa=600, games=150, variance=0.12),
    "Everyday_Plus": RoleBaseline(pa=650, games=155, variance=0.10),
    "Regular": RoleBaseline(pa=520, games=135, variance=0.15),
    "Platoon": RoleBaseline(pa=350, games=110, variance=0.18),
    "Utility": RoleBaseline(pa=280, games=100, variance=0.20),
    "Bench": RoleBaseline(pa=180, games=80, variance=0.25),
    "Part_Time": RoleBaseline(pa=120, games=60, variance=0.30),
}

# Pitcher roles
PITCHER_ROLES = {
    "SP1": RoleBaseline(ip=185, games=32, variance=0.12),
    "SP2": RoleBaseline(ip=175, games=31, variance=0.13),
    "SP3": RoleBaseline(ip=165, games=30, variance=0.14),
    "SP4": RoleBaseline(ip=150, games=28, variance=0.15),
    "SP5": RoleBaseline(ip=130, games=26, variance=0.18),
    "Swing": RoleBaseline(ip=90, games=35, variance=0.25),  # Swing starter/long relief
    "CL": RoleBaseline(ip=62, games=62, variance=0.15),
    "SU": RoleBaseline(ip=65, games=68, variance=0.15),  # Setup
    "RP_High": RoleBaseline(ip=60, games=65, variance=0.18),
    "RP_Mid": RoleBaseline(ip=50, games=55, variance=0.20),
    "RP_Low": RoleBaseline(ip=35, games=40, variance=0.25),
    "LOOGY": RoleBaseline(ip=25, games=45, variance=0.30),  # Specialist
}

# Legacy dict for backward compatibility
ROLE_BASELINES = {
    **{k: {"pa": v.pa, "games": v.games} for k, v in BATTER_ROLES.items()},
    **{k: {"ip": v.ip, "games": v.games} for k, v in PITCHER_ROLES.items()},
}


# ============================================================================
# Age Adjustments for Playing Time
# ============================================================================

def get_health_multiplier(age: int, player_type: Literal["batter", "pitcher"]) -> float:
    """Get playing time multiplier based on age/health risk.

    Older players have higher injury risk and tend to get less playing time.

    Args:
        age: Player's age.
        player_type: "batter" or "pitcher".

    Returns:
        Multiplier (0.0-1.0) to apply to baseline playing time.
    """
    if player_type == "batter":
        # Batters have a more gradual decline
        multipliers = {
            21: 0.92, 22: 0.95, 23: 0.97, 24: 0.99, 25: 1.00,
            26: 1.00, 27: 1.00, 28: 1.00, 29: 0.98, 30: 0.96,
            31: 0.94, 32: 0.92, 33: 0.89, 34: 0.86, 35: 0.82,
            36: 0.78, 37: 0.73, 38: 0.68, 39: 0.62, 40: 0.55,
        }
    else:
        # Pitchers have steeper injury risk with age
        multipliers = {
            21: 0.88, 22: 0.92, 23: 0.95, 24: 0.98, 25: 1.00,
            26: 1.00, 27: 1.00, 28: 0.98, 29: 0.95, 30: 0.92,
            31: 0.88, 32: 0.83, 33: 0.78, 34: 0.72, 35: 0.65,
            36: 0.58, 37: 0.50, 38: 0.42, 39: 0.35, 40: 0.28,
        }

    if age < 21:
        return multipliers[21]
    elif age > 40:
        return multipliers[40] * (0.85 ** (age - 40))
    else:
        return multipliers.get(age, 0.70)


def apply_age_adjustment(
    baseline_playing_time: float,
    age: int,
    player_type: Literal["batter", "pitcher"],
) -> float:
    """Adjust projected playing time based on age/injury risk.

    Args:
        baseline_playing_time: Baseline PA or IP projection.
        age: Player's age.
        player_type: "batter" or "pitcher".

    Returns:
        Adjusted playing time projection.
    """
    return baseline_playing_time * get_health_multiplier(age, player_type)


# ============================================================================
# Role Detection
# ============================================================================

def detect_batter_role(historical_stats: pd.DataFrame) -> str:
    """Detect a batter's role based on historical playing time.

    Args:
        historical_stats: DataFrame with player's batting stats by year.

    Returns:
        Role string (e.g., "Everyday", "Platoon", "Bench").
    """
    if len(historical_stats) == 0:
        return "Bench"

    # Weight recent years more heavily
    stats = historical_stats.sort_values("Season", ascending=False).head(3)
    weights = [3, 2, 1][:len(stats)]

    # Calculate weighted average PA
    total_weight = sum(weights)
    weighted_pa = sum(
        row.get("PA", 0) * weights[i]
        for i, (_, row) in enumerate(stats.iterrows())
        if i < len(weights)
    ) / total_weight

    # Also consider games played for injury-prone players
    if "G" in stats.columns:
        weighted_games = sum(
            row.get("G", 0) * weights[i]
            for i, (_, row) in enumerate(stats.iterrows())
            if i < len(weights)
        ) / total_weight
    else:
        weighted_games = weighted_pa / 4  # Rough estimate

    # Classify based on PA
    if weighted_pa >= 620:
        return "Everyday_Plus"
    elif weighted_pa >= 550:
        return "Everyday"
    elif weighted_pa >= 450:
        return "Regular"
    elif weighted_pa >= 300:
        if weighted_games >= 100:
            return "Utility"
        else:
            return "Platoon"
    elif weighted_pa >= 150:
        return "Bench"
    else:
        return "Part_Time"


def detect_pitcher_role(historical_stats: pd.DataFrame) -> str:
    """Detect a pitcher's role based on historical usage.

    Args:
        historical_stats: DataFrame with player's pitching stats by year.

    Returns:
        Role string (e.g., "SP1", "CL", "RP_Mid").
    """
    if len(historical_stats) == 0:
        return "RP_Low"

    # Weight recent years
    stats = historical_stats.sort_values("Season", ascending=False).head(3)
    weights = [3, 2, 1][:len(stats)]
    total_weight = sum(weights)

    # Calculate weighted averages
    weighted_ip = sum(
        row.get("IP", 0) * weights[i]
        for i, (_, row) in enumerate(stats.iterrows())
        if i < len(weights)
    ) / total_weight

    weighted_gs = sum(
        row.get("GS", 0) * weights[i]
        for i, (_, row) in enumerate(stats.iterrows())
        if i < len(weights)
    ) / total_weight

    weighted_sv = sum(
        row.get("SV", 0) * weights[i]
        for i, (_, row) in enumerate(stats.iterrows())
        if i < len(weights)
    ) / total_weight

    weighted_g = sum(
        row.get("G", 0) * weights[i]
        for i, (_, row) in enumerate(stats.iterrows())
        if i < len(weights)
    ) / total_weight

    # Classify starter vs reliever
    if weighted_gs >= 20:
        # Starter - classify by workload
        if weighted_ip >= 180:
            return "SP1"
        elif weighted_ip >= 165:
            return "SP2"
        elif weighted_ip >= 150:
            return "SP3"
        elif weighted_ip >= 130:
            return "SP4"
        elif weighted_ip >= 100:
            return "SP5"
        else:
            return "Swing"
    else:
        # Reliever
        if weighted_sv >= 25:
            return "CL"
        elif weighted_sv >= 10 or (weighted_ip >= 60 and weighted_g >= 60):
            return "SU"
        elif weighted_ip >= 55:
            return "RP_High"
        elif weighted_ip >= 40:
            return "RP_Mid"
        elif weighted_ip >= 20:
            return "RP_Low"
        else:
            return "LOOGY"


def estimate_role(
    historical_stats: pd.DataFrame,
    player_type: Literal["batter", "pitcher"],
) -> str:
    """Estimate a player's role based on historical usage.

    Args:
        historical_stats: DataFrame with player's historical stats.
        player_type: "batter" or "pitcher".

    Returns:
        Role string (e.g., "Everyday", "SP1", "RP_High").
    """
    if player_type == "batter":
        return detect_batter_role(historical_stats)
    else:
        return detect_pitcher_role(historical_stats)


# ============================================================================
# Playing Time Projection
# ============================================================================

@dataclass
class PlayingTimeProjection:
    """Projected playing time for a player."""

    player_id: int
    name: str
    team: str
    role: str

    # Point estimates
    projected_pa: float = 0.0
    projected_ip: float = 0.0
    projected_games: float = 0.0

    # Confidence intervals (for simulation)
    pa_10th_pct: float = 0.0
    pa_90th_pct: float = 0.0
    ip_10th_pct: float = 0.0
    ip_90th_pct: float = 0.0

    # Legacy aliases
    @property
    def low_pa(self) -> float:
        return self.pa_10th_pct

    @property
    def high_pa(self) -> float:
        return self.pa_90th_pct

    @property
    def low_ip(self) -> float:
        return self.ip_10th_pct

    @property
    def high_ip(self) -> float:
        return self.ip_90th_pct

    # Metadata
    age: int = 0
    health_multiplier: float = 1.0
    historical_avg_pa: float = 0.0
    historical_avg_ip: float = 0.0


def project_playing_time(
    player_id: int,
    historical_stats: pd.DataFrame,
    player_type: Literal["batter", "pitcher"],
    age: int,
    role_override: Optional[str] = None,
) -> PlayingTimeProjection:
    """Project playing time for a player.

    Args:
        player_id: Player ID.
        historical_stats: DataFrame with player's historical stats.
        player_type: "batter" or "pitcher".
        age: Player's age for projection year.
        role_override: Override detected role with specific role.

    Returns:
        PlayingTimeProjection with projected PA/IP.
    """
    if len(historical_stats) == 0:
        # No history - use role baseline with high variance
        if player_type == "batter":
            role = role_override or "Bench"
            baseline = BATTER_ROLES.get(role, BATTER_ROLES["Bench"])
            return PlayingTimeProjection(
                player_id=player_id,
                name=f"Player {player_id}",
                team="",
                role=role,
                projected_pa=baseline.pa * 0.8,  # Conservative for unknowns
                projected_games=baseline.games * 0.8,
                age=age,
            )
        else:
            role = role_override or "RP_Low"
            baseline = PITCHER_ROLES.get(role, PITCHER_ROLES["RP_Low"])
            return PlayingTimeProjection(
                player_id=player_id,
                name=f"Player {player_id}",
                team="",
                role=role,
                projected_ip=baseline.ip * 0.8,
                projected_games=baseline.games * 0.8,
                age=age,
            )

    # Get player info
    most_recent = historical_stats.sort_values("Season", ascending=False).iloc[0]
    name = most_recent.get("Name", f"Player {player_id}")
    team = most_recent.get("Team", "")

    # Detect role
    if player_type == "batter":
        role = role_override or detect_batter_role(historical_stats)
        baseline = BATTER_ROLES.get(role, BATTER_ROLES["Bench"])
    else:
        role = role_override or detect_pitcher_role(historical_stats)
        baseline = PITCHER_ROLES.get(role, PITCHER_ROLES["RP_Low"])

    # Calculate historical average (weighted toward recent)
    weights = [3, 2, 1]
    stats = historical_stats.sort_values("Season", ascending=False).head(3)
    total_weight = sum(weights[:len(stats)])

    if player_type == "batter":
        historical_avg = sum(
            row.get("PA", 0) * weights[i]
            for i, (_, row) in enumerate(stats.iterrows())
            if i < len(weights)
        ) / total_weight
        baseline_val = baseline.pa
    else:
        historical_avg = sum(
            row.get("IP", 0) * weights[i]
            for i, (_, row) in enumerate(stats.iterrows())
            if i < len(weights)
        ) / total_weight
        baseline_val = baseline.ip

    # Blend historical average with role baseline (60/40 split)
    blended = 0.6 * historical_avg + 0.4 * baseline_val

    # Apply age/health adjustment
    health_mult = get_health_multiplier(age, player_type)
    adjusted = blended * health_mult

    # Calculate confidence intervals
    variance = baseline.variance
    std_dev = adjusted * variance

    if player_type == "batter":
        projection = PlayingTimeProjection(
            player_id=player_id,
            name=name,
            team=team,
            role=role,
            projected_pa=adjusted,
            projected_games=baseline.games * health_mult,
            pa_10th_pct=max(0, adjusted - 1.28 * std_dev),
            pa_90th_pct=adjusted + 1.28 * std_dev,
            age=age,
            health_multiplier=health_mult,
            historical_avg_pa=historical_avg,
        )
    else:
        projection = PlayingTimeProjection(
            player_id=player_id,
            name=name,
            team=team,
            role=role,
            projected_ip=adjusted,
            projected_games=baseline.games * health_mult,
            ip_10th_pct=max(0, adjusted - 1.28 * std_dev),
            ip_90th_pct=adjusted + 1.28 * std_dev,
            age=age,
            health_multiplier=health_mult,
            historical_avg_ip=historical_avg,
        )

    return projection


def allocate_team_playing_time(
    team: str,
    batter_projections: List[PlayingTimeProjection],
    pitcher_projections: List[PlayingTimeProjection],
) -> Tuple[List[PlayingTimeProjection], List[PlayingTimeProjection]]:
    """Allocate playing time across a team's roster.

    Ensures total PA and IP are realistic for a team.

    Args:
        team: Team abbreviation.
        batter_projections: List of batter playing time projections.
        pitcher_projections: List of pitcher playing time projections.

    Returns:
        Tuple of (adjusted batter projections, adjusted pitcher projections).
    """
    # Team season totals (approximately)
    TEAM_PA = 6200  # ~38-39 PA per game * 162
    TEAM_IP = 1458  # 162 * 9

    # Adjust batters
    total_pa = sum(p.projected_pa for p in batter_projections)
    if total_pa > 0:
        pa_ratio = TEAM_PA / total_pa
        # Don't over-inflate - cap adjustment at 1.1x
        pa_ratio = min(pa_ratio, 1.1)
        for p in batter_projections:
            p.projected_pa *= pa_ratio
            p.pa_10th_pct *= pa_ratio
            p.pa_90th_pct *= pa_ratio

    # Adjust pitchers
    total_ip = sum(p.projected_ip for p in pitcher_projections)
    if total_ip > 0:
        ip_ratio = TEAM_IP / total_ip
        ip_ratio = min(ip_ratio, 1.1)
        for p in pitcher_projections:
            p.projected_ip *= ip_ratio
            p.ip_10th_pct *= ip_ratio
            p.ip_90th_pct *= ip_ratio

    return batter_projections, pitcher_projections


def sample_playing_time(projection: PlayingTimeProjection, n_samples: int = 1) -> np.ndarray:
    """Sample from the playing time distribution for Monte Carlo simulation.

    Args:
        projection: PlayingTimeProjection.
        n_samples: Number of samples to generate.

    Returns:
        Array of sampled PA or IP values.
    """
    if projection.projected_pa > 0:
        mean = projection.projected_pa
        # Estimate std from percentiles
        std = (projection.pa_90th_pct - projection.pa_10th_pct) / 2.56
        if std <= 0:
            std = mean * 0.15  # Default 15% variance
        samples = np.random.normal(mean, std, n_samples)
        return np.maximum(0, samples)  # Can't have negative PA
    elif projection.projected_ip > 0:
        mean = projection.projected_ip
        std = (projection.ip_90th_pct - projection.ip_10th_pct) / 2.56
        if std <= 0:
            std = mean * 0.15
        samples = np.random.normal(mean, std, n_samples)
        return np.maximum(0, samples)
    else:
        return np.zeros(n_samples)
