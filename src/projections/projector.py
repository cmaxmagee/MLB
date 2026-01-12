"""High-level projection orchestrator.

Ties together player projections, playing time, and team aggregation
into a unified projection workflow.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import pandas as pd

from ..data.fetch import (
    fetch_batting_stats,
    fetch_pitching_stats,
    fetch_historical_player_stats,
)
from ..data.roster_changes import RosterChange, apply_roster_changes
from .player import (
    PlayerProjection,
    BatterProjection,
    PitcherProjection,
    project_batter,
    project_pitcher,
    calculate_counting_stats,
)
from .playing_time import (
    PlayingTimeProjection,
    project_playing_time,
    allocate_team_playing_time,
)
from .team import TEAM_TO_DIVISION
from .schedule import apply_sos_adjustments, calculate_all_sos

logger = logging.getLogger(__name__)


@dataclass
class ProjectionConfig:
    """Configuration for projection generation."""

    projection_year: int = 2025
    historical_years: int = 3  # How many years of history to use
    year_weights: List[float] = field(default_factory=lambda: [5, 4, 3])
    min_pa: int = 50  # Minimum PA to include batter
    min_ip: float = 10.0  # Minimum IP to include pitcher


@dataclass
class TeamProjectionSet:
    """Complete projections for a team."""

    team: str
    division: str
    batters: List[BatterProjection] = field(default_factory=list)
    pitchers: List[PitcherProjection] = field(default_factory=list)

    # Aggregated stats
    total_batting_runs: float = 0.0
    total_pitching_runs: float = 0.0
    projected_runs_scored: float = 0.0
    projected_runs_allowed: float = 0.0
    projected_wins: float = 0.0

    # Strength of Schedule adjustments
    sos: float = 0.500  # Overall SOS (0.500 = neutral)
    sos_win_adjustment: float = 0.0  # Adjustment applied to wins
    projected_wins_pre_sos: float = 0.0  # Wins before SOS adjustment


class Projector:
    """Main class for generating MLB projections.

    Orchestrates data fetching, player projections, and team aggregation.
    """

    def __init__(self, config: Optional[ProjectionConfig] = None):
        """Initialize the projector.

        Args:
            config: Projection configuration. Uses defaults if not provided.
        """
        self.config = config or ProjectionConfig()
        self._batting_cache: Dict[int, pd.DataFrame] = {}
        self._pitching_cache: Dict[int, pd.DataFrame] = {}

    def project_all_teams(
        self,
        roster_changes: Optional[List[RosterChange]] = None,
    ) -> Dict[str, TeamProjectionSet]:
        """Generate projections for all 30 MLB teams.

        Args:
            roster_changes: Optional roster changes to apply.

        Returns:
            Dictionary mapping team abbreviation to TeamProjectionSet.
        """
        logger.info(f"Generating {self.config.projection_year} projections...")

        # Fetch all player stats for the historical period
        start_year = self.config.projection_year - self.config.historical_years
        end_year = self.config.projection_year - 1

        logger.info(f"Fetching stats from {start_year}-{end_year}")
        batting_stats = fetch_batting_stats(start_year, end_year, qual=self.config.min_pa)
        pitching_stats = fetch_pitching_stats(start_year, end_year, qual=self.config.min_ip)

        # Normalize team abbreviations (ATH -> OAK, etc.)
        batting_stats = self._normalize_team_names(batting_stats)
        pitching_stats = self._normalize_team_names(pitching_stats)

        # Apply roster changes to update player teams
        # Build team change mapping for park factor adjustments
        team_changes = {}  # {player_name_lower: (from_team, to_team)}
        if roster_changes:
            logger.info(f"Applying {len(roster_changes)} roster changes...")
            batting_stats = self._apply_roster_changes_to_stats(batting_stats, roster_changes, end_year)
            pitching_stats = self._apply_roster_changes_to_stats(pitching_stats, roster_changes, end_year)

            # Build mapping for park factor adjustments
            from ..data.mlb_api import normalize_name
            for change in roster_changes:
                if change.from_team and change.to_team and change.from_team != change.to_team:
                    name_key = normalize_name(change.player_name)
                    team_changes[name_key] = (change.from_team, change.to_team)

        # Get most recent year's rosters as baseline
        most_recent_batting = batting_stats[batting_stats["Season"] == end_year]
        most_recent_pitching = pitching_stats[pitching_stats["Season"] == end_year]

        # Group by team (filter out invalid team names)
        all_teams = set(most_recent_batting["Team"].unique()) | set(most_recent_pitching["Team"].unique())
        # Valid MLB team abbreviations are 2-3 uppercase letters
        teams = {t for t in all_teams if t and isinstance(t, str) and len(t) <= 3 and t.isalpha()}

        projections = {}
        for team in sorted(teams):
            logger.info(f"Projecting {team}...")
            team_proj = self._project_team(
                team,
                batting_stats,
                pitching_stats,
                team_changes,
            )
            projections[team] = team_proj

        # Apply Strength of Schedule adjustments
        logger.info("Calculating Strength of Schedule adjustments...")
        sos_results = calculate_all_sos(projections)
        projections = apply_sos_adjustments(projections, sos_results)

        return projections

    def _project_team(
        self,
        team: str,
        all_batting: pd.DataFrame,
        all_pitching: pd.DataFrame,
        team_changes: Optional[dict] = None,
    ) -> TeamProjectionSet:
        """Generate projections for a single team.

        Args:
            team: Team abbreviation.
            all_batting: All batting stats for the historical period.
            all_pitching: All pitching stats for the historical period.
            team_changes: Dict mapping normalized player names to (from_team, to_team).

        Returns:
            TeamProjectionSet with all projections.
        """
        team_changes = team_changes or {}
        end_year = self.config.projection_year - 1

        # Get current roster (most recent year)
        team_batters = all_batting[
            (all_batting["Team"] == team) & (all_batting["Season"] == end_year)
        ]
        team_pitchers = all_pitching[
            (all_pitching["Team"] == team) & (all_pitching["Season"] == end_year)
        ]

        # Project each batter
        batter_projections = []
        batter_pt_projections = []

        from ..data.mlb_api import normalize_name

        for _, row in team_batters.iterrows():
            player_id = row.get("IDfg", row.get("playerid", 0))
            if player_id == 0:
                continue

            # Get full history for this player
            player_history = all_batting[all_batting["IDfg"] == player_id]
            if len(player_history) == 0:
                player_history = all_batting[all_batting["playerid"] == player_id]

            # Check if player changed teams (for park factor adjustment)
            player_name = row.get("Name", "")
            name_key = normalize_name(player_name)
            new_team_for_park = None
            if name_key in team_changes:
                # Player changed teams - pass new_team for park factor adjustment
                new_team_for_park = team

            try:
                # Generate projection
                proj = project_batter(
                    player_id=int(player_id),
                    historical_stats=player_history,
                    projection_year=self.config.projection_year,
                    year_weights=self.config.year_weights,
                    new_team=new_team_for_park,
                )

                # Project playing time
                pt_proj = project_playing_time(
                    player_id=int(player_id),
                    historical_stats=player_history,
                    player_type="batter",
                    age=proj.age,
                )

                # Update projection with playing time
                proj.projected_pa = pt_proj.projected_pa
                proj.projected_games = pt_proj.projected_games

                # Calculate counting stats
                proj = calculate_counting_stats(proj, pa=pt_proj.projected_pa)

                batter_projections.append(proj)
                batter_pt_projections.append(pt_proj)

            except Exception as e:
                logger.warning(f"Failed to project batter {player_id}: {e}")

        # Project each pitcher
        pitcher_projections = []
        pitcher_pt_projections = []

        for _, row in team_pitchers.iterrows():
            player_id = row.get("IDfg", row.get("playerid", 0))
            if player_id == 0:
                continue

            player_history = all_pitching[all_pitching["IDfg"] == player_id]
            if len(player_history) == 0:
                player_history = all_pitching[all_pitching["playerid"] == player_id]

            # Check if player changed teams (for park factor adjustment)
            player_name = row.get("Name", "")
            name_key = normalize_name(player_name)
            new_team_for_park = None
            if name_key in team_changes:
                new_team_for_park = team

            try:
                proj = project_pitcher(
                    player_id=int(player_id),
                    historical_stats=player_history,
                    projection_year=self.config.projection_year,
                    year_weights=self.config.year_weights,
                    new_team=new_team_for_park,
                )

                pt_proj = project_playing_time(
                    player_id=int(player_id),
                    historical_stats=player_history,
                    player_type="pitcher",
                    age=proj.age,
                )

                proj.projected_ip = pt_proj.projected_ip
                proj.projected_games = pt_proj.projected_games
                proj = calculate_counting_stats(proj, ip=pt_proj.projected_ip)

                pitcher_projections.append(proj)
                pitcher_pt_projections.append(pt_proj)

            except Exception as e:
                logger.warning(f"Failed to project pitcher {player_id}: {e}")

        # Allocate team playing time
        if batter_pt_projections and pitcher_pt_projections:
            allocate_team_playing_time(team, batter_pt_projections, pitcher_pt_projections)

            # Update projections with adjusted playing time
            for proj, pt in zip(batter_projections, batter_pt_projections):
                proj.projected_pa = pt.projected_pa
                proj = calculate_counting_stats(proj, pa=pt.projected_pa)

            for proj, pt in zip(pitcher_projections, pitcher_pt_projections):
                proj.projected_ip = pt.projected_ip
                proj = calculate_counting_stats(proj, ip=pt.projected_ip)

        # Aggregate team stats
        total_batting_runs = sum(p.projected_batting_runs for p in batter_projections)
        total_pitching_runs = sum(p.projected_pitching_runs for p in pitcher_projections)

        # Convert to team runs (league average ~700 runs/season)
        league_avg_runs = 700
        projected_runs_scored = league_avg_runs + total_batting_runs
        projected_runs_allowed = league_avg_runs - total_pitching_runs

        # Pythagorean wins
        rs = projected_runs_scored
        ra = projected_runs_allowed
        if ra > 0 and rs > 0:
            exponent = 1.83
            win_pct = (rs ** exponent) / (rs ** exponent + ra ** exponent)
            projected_wins = win_pct * 162
        else:
            projected_wins = 81

        division = TEAM_TO_DIVISION.get(team, "Unknown")

        return TeamProjectionSet(
            team=team,
            division=division,
            batters=batter_projections,
            pitchers=pitcher_projections,
            total_batting_runs=total_batting_runs,
            total_pitching_runs=total_pitching_runs,
            projected_runs_scored=projected_runs_scored,
            projected_runs_allowed=projected_runs_allowed,
            projected_wins=projected_wins,
        )

    def _normalize_team_names(self, stats: pd.DataFrame) -> pd.DataFrame:
        """Normalize team abbreviations in stats DataFrame.

        FanGraphs uses different abbreviations (ATH, FLA, etc.) than standard MLB.
        This ensures consistent team names throughout the projection.
        """
        from ..data.mlb_api import normalize_team_abbrev

        stats = stats.copy()
        if "Team" in stats.columns:
            stats["Team"] = stats["Team"].apply(
                lambda x: normalize_team_abbrev(x) if pd.notna(x) else x
            )
        return stats

    def _apply_roster_changes_to_stats(
        self,
        stats: pd.DataFrame,
        roster_changes: List[RosterChange],
        most_recent_year: int,
    ) -> pd.DataFrame:
        """Apply roster changes to update player team assignments.

        Args:
            stats: DataFrame with player stats (must have Name, Team, Season, IDfg columns).
            roster_changes: List of roster changes to apply.
            most_recent_year: The most recent season year in the data.

        Returns:
            Updated DataFrame with players moved to their new teams.
        """
        from ..data.roster_changes import ChangeType

        stats = stats.copy()

        for change in roster_changes:
            if change.change_type not in (ChangeType.TRADE, ChangeType.SIGNING):
                continue

            if not change.to_team:
                continue

            # Find player by name (case-insensitive)
            name_mask = stats["Name"].str.lower() == change.player_name.lower()

            # Also try matching by player ID if available
            if change.player_id:
                id_col = "IDfg" if "IDfg" in stats.columns else "playerid"
                id_mask = stats[id_col] == change.player_id
                mask = name_mask | id_mask
            else:
                mask = name_mask

            if mask.any():
                # Update team for most recent year's stats
                year_mask = stats["Season"] == most_recent_year
                stats.loc[mask & year_mask, "Team"] = change.to_team
                logger.debug(f"Moved {change.player_name} to {change.to_team}")

        return stats

    def project_single_player(
        self,
        player_id: int,
        player_type: Literal["batter", "pitcher"],
        new_team: Optional[str] = None,
    ) -> PlayerProjection:
        """Generate projection for a single player.

        Args:
            player_id: FanGraphs player ID.
            player_type: "batter" or "pitcher".
            new_team: New team if player changed teams.

        Returns:
            PlayerProjection (BatterProjection or PitcherProjection).
        """
        start_year = self.config.projection_year - self.config.historical_years
        end_year = self.config.projection_year - 1

        # Fetch player history
        history = fetch_historical_player_stats(
            player_id=player_id,
            start_year=start_year,
            end_year=end_year,
            player_type=player_type,
        )

        if player_type == "batter":
            proj = project_batter(
                player_id=player_id,
                historical_stats=history,
                projection_year=self.config.projection_year,
                new_team=new_team,
                year_weights=self.config.year_weights,
            )
        else:
            proj = project_pitcher(
                player_id=player_id,
                historical_stats=history,
                projection_year=self.config.projection_year,
                new_team=new_team,
                year_weights=self.config.year_weights,
            )

        # Add playing time
        pt = project_playing_time(
            player_id=player_id,
            historical_stats=history,
            player_type=player_type,
            age=proj.age,
        )

        if player_type == "batter":
            proj.projected_pa = pt.projected_pa
            proj = calculate_counting_stats(proj, pa=pt.projected_pa)
        else:
            proj.projected_ip = pt.projected_ip
            proj = calculate_counting_stats(proj, ip=pt.projected_ip)

        return proj


def projections_to_dataframe(
    team_projections: Dict[str, TeamProjectionSet],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert team projections to DataFrames.

    Args:
        team_projections: Dictionary of team projections.

    Returns:
        Tuple of (batter DataFrame, pitcher DataFrame).
    """
    batter_rows = []
    pitcher_rows = []

    for team, proj_set in team_projections.items():
        for batter in proj_set.batters:
            batter_rows.append({
                "Team": team,
                "Name": batter.name,
                "Age": batter.age,
                "PA": round(batter.projected_pa),
                "AVG": round(batter.projected_avg, 3),
                "OBP": round(batter.projected_obp, 3),
                "SLG": round(batter.projected_slg, 3),
                "wRC+": round(batter.projected_wrc_plus),
                "HR": round(batter.projected_hr),
                "RBI": round(batter.projected_rbi),
                "Runs": round(batter.projected_runs),
                "Batting Runs": round(batter.projected_batting_runs, 1),
            })

        for pitcher in proj_set.pitchers:
            pitcher_rows.append({
                "Team": team,
                "Name": pitcher.name,
                "Age": pitcher.age,
                "Role": pitcher.role,
                "IP": round(pitcher.projected_ip, 1),
                "ERA": round(pitcher.projected_era, 2),
                "FIP": round(pitcher.projected_fip, 2),
                "WHIP": round(pitcher.projected_whip, 2),
                "K/9": round(pitcher.projected_k_per_9, 1),
                "K": round(pitcher.projected_strikeouts),
                "W": round(pitcher.projected_wins),
                "SV": round(pitcher.projected_saves),
                "Pitching Runs": round(pitcher.projected_pitching_runs, 1),
            })

    return pd.DataFrame(batter_rows), pd.DataFrame(pitcher_rows)
