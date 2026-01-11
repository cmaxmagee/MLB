"""Rotation and bullpen innings allocation.

Distributes team innings (~1450 total) among starting rotation and bullpen,
handling innings overflow when starters underperform projections.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# League averages for innings distribution
TEAM_TOTAL_IP = 1458  # 162 games * 9 innings
STARTER_TARGET_IP = 950  # ~65% of innings
BULLPEN_TARGET_IP = 508  # ~35% of innings


@dataclass
class PitchingStaffProjection:
    """Projected innings distribution for a team's pitching staff."""

    team: str
    total_ip: float = TEAM_TOTAL_IP

    # Rotation
    rotation: list[dict] = field(default_factory=list)
    total_starter_ip: float = 0.0

    # Bullpen
    bullpen: list[dict] = field(default_factory=list)
    total_bullpen_ip: float = 0.0

    # Overflow tracking
    overflow_ip: float = 0.0  # IP needed from depth/callups


def allocate_team_innings(
    pitchers: pd.DataFrame,
    team: str,
) -> PitchingStaffProjection:
    """Allocate innings to a team's pitching staff.

    Args:
        pitchers: DataFrame with pitcher projections for this team.
                  Expected columns: playerid, Name, projected_ip, role, projected_fip
        team: Team abbreviation.

    Returns:
        PitchingStaffProjection with innings allocated.
    """
    projection = PitchingStaffProjection(team=team)

    # Separate starters and relievers
    starters = pitchers[pitchers["role"].str.startswith("SP")].copy()
    relievers = pitchers[~pitchers["role"].str.startswith("SP")].copy()

    # Sort starters by projected quality (FIP)
    starters = starters.sort_values("projected_fip")

    # Allocate starter innings
    starter_ip_remaining = STARTER_TARGET_IP
    for _, pitcher in starters.iterrows():
        ip = min(pitcher["projected_ip"], starter_ip_remaining)
        projection.rotation.append({
            "playerid": pitcher["playerid"],
            "name": pitcher["Name"],
            "ip": ip,
            "fip": pitcher["projected_fip"],
        })
        projection.total_starter_ip += ip
        starter_ip_remaining -= ip
        if starter_ip_remaining <= 0:
            break

    # Any remaining starter innings become overflow
    if starter_ip_remaining > 0:
        projection.overflow_ip += starter_ip_remaining
        logger.warning(
            f"{team}: {starter_ip_remaining:.0f} starter IP unfilled, "
            "will need depth/callups"
        )

    # Allocate bullpen innings
    bullpen_ip_remaining = BULLPEN_TARGET_IP

    # Sort relievers by leverage expectation (closers first, then high-leverage)
    role_order = {"CL": 0, "RP_High": 1, "RP_Mid": 2, "RP_Low": 3}
    relievers["role_order"] = relievers["role"].map(role_order).fillna(4)
    relievers = relievers.sort_values(["role_order", "projected_fip"])

    for _, pitcher in relievers.iterrows():
        ip = min(pitcher["projected_ip"], bullpen_ip_remaining)
        projection.bullpen.append({
            "playerid": pitcher["playerid"],
            "name": pitcher["Name"],
            "ip": ip,
            "fip": pitcher["projected_fip"],
            "role": pitcher["role"],
        })
        projection.total_bullpen_ip += ip
        bullpen_ip_remaining -= ip
        if bullpen_ip_remaining <= 0:
            break

    # Any remaining bullpen innings become overflow
    if bullpen_ip_remaining > 0:
        projection.overflow_ip += bullpen_ip_remaining
        logger.warning(
            f"{team}: {bullpen_ip_remaining:.0f} bullpen IP unfilled"
        )

    return projection


def calculate_team_runs_allowed(
    staff: PitchingStaffProjection,
    league_avg_fip: float = 4.00,
) -> float:
    """Calculate projected team runs allowed from pitching staff.

    Uses FIP-based runs calculation for each pitcher weighted by IP.

    Args:
        staff: PitchingStaffProjection with innings allocations.
        league_avg_fip: League average FIP for calculating runs.

    Returns:
        Projected total runs allowed for the season.
    """
    total_runs = 0.0

    # FIP to runs conversion (FIP is on ERA scale)
    # Runs = IP * FIP / 9
    for starter in staff.rotation:
        runs = starter["ip"] * starter["fip"] / 9
        total_runs += runs

    for reliever in staff.bullpen:
        runs = reliever["ip"] * reliever["fip"] / 9
        total_runs += runs

    # Overflow innings at league average
    if staff.overflow_ip > 0:
        overflow_runs = staff.overflow_ip * (league_avg_fip + 0.5) / 9
        total_runs += overflow_runs
        logger.debug(
            f"{staff.team}: {staff.overflow_ip:.0f} overflow IP "
            f"projected at {overflow_runs:.0f} runs"
        )

    return total_runs


def validate_innings_distribution(staff: PitchingStaffProjection) -> list[str]:
    """Validate a team's innings distribution.

    Args:
        staff: PitchingStaffProjection to validate.

    Returns:
        List of warning messages (empty if valid).
    """
    warnings = []

    total = staff.total_starter_ip + staff.total_bullpen_ip + staff.overflow_ip

    if abs(total - TEAM_TOTAL_IP) > 1:
        warnings.append(
            f"Total IP ({total:.0f}) doesn't match expected ({TEAM_TOTAL_IP})"
        )

    if len(staff.rotation) < 5:
        warnings.append(f"Only {len(staff.rotation)} starters projected")

    if staff.overflow_ip > 100:
        warnings.append(f"High overflow IP ({staff.overflow_ip:.0f}) - roster depth issue")

    return warnings
