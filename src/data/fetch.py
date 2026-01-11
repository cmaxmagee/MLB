"""Pybaseball wrappers for fetching MLB data."""

import logging
from typing import Literal

import pandas as pd
from pybaseball import (
    batting_stats,
    pitching_stats,
    team_batting,
    team_pitching,
    playerid_lookup,
    playerid_reverse_lookup,
)
from pybaseball import cache as pybaseball_cache

from .cache import DataCache, DEFAULT_CACHE_DIR

logger = logging.getLogger(__name__)

# Enable pybaseball's built-in caching
pybaseball_cache.enable()

# Default cache instance
_cache = DataCache(DEFAULT_CACHE_DIR)


def fetch_batting_stats(
    start_year: int,
    end_year: int | None = None,
    qual: int | str = "y",
    cache: DataCache | None = None,
) -> pd.DataFrame:
    """Fetch batting statistics from FanGraphs.

    Args:
        start_year: First season to fetch.
        end_year: Last season to fetch. Defaults to start_year.
        qual: Minimum plate appearances. "y" for qualified, or an integer.
        cache: Cache instance to use. Uses default if not provided.

    Returns:
        DataFrame with batting statistics including:
        - Player identifiers (Name, playerid, Team)
        - Counting stats (PA, AB, H, HR, RBI, SB, etc.)
        - Rate stats (AVG, OBP, SLG, wOBA, wRC+)
        - Advanced metrics (WAR, Batting, Baserunning)
    """
    end_year = end_year or start_year
    cache = cache or _cache

    def _fetch() -> pd.DataFrame:
        logger.info(f"Fetching batting stats for {start_year}-{end_year}")
        df = batting_stats(start_year, end_year, qual=qual)
        # Standardize column names
        df.columns = df.columns.str.strip()
        return df

    return cache.get_or_fetch(
        name="batting_stats",
        fetch_fn=_fetch,
        start_year=start_year,
        end_year=end_year,
        qual=qual,
    )


def fetch_pitching_stats(
    start_year: int,
    end_year: int | None = None,
    qual: int | str = "y",
    cache: DataCache | None = None,
) -> pd.DataFrame:
    """Fetch pitching statistics from FanGraphs.

    Args:
        start_year: First season to fetch.
        end_year: Last season to fetch. Defaults to start_year.
        qual: Minimum innings pitched. "y" for qualified, or an integer.
        cache: Cache instance to use. Uses default if not provided.

    Returns:
        DataFrame with pitching statistics including:
        - Player identifiers (Name, playerid, Team)
        - Counting stats (W, L, SV, IP, SO, BB, HR)
        - Rate stats (ERA, FIP, WHIP, K/9, BB/9)
        - Advanced metrics (WAR, xFIP, SIERA)
    """
    end_year = end_year or start_year
    cache = cache or _cache

    def _fetch() -> pd.DataFrame:
        logger.info(f"Fetching pitching stats for {start_year}-{end_year}")
        df = pitching_stats(start_year, end_year, qual=qual)
        df.columns = df.columns.str.strip()
        return df

    return cache.get_or_fetch(
        name="pitching_stats",
        fetch_fn=_fetch,
        start_year=start_year,
        end_year=end_year,
        qual=qual,
    )


def fetch_team_batting(
    start_year: int,
    end_year: int | None = None,
    cache: DataCache | None = None,
) -> pd.DataFrame:
    """Fetch team-level batting statistics.

    Args:
        start_year: First season to fetch.
        end_year: Last season to fetch. Defaults to start_year.
        cache: Cache instance to use. Uses default if not provided.

    Returns:
        DataFrame with team batting totals.
    """
    end_year = end_year or start_year
    cache = cache or _cache

    def _fetch() -> pd.DataFrame:
        logger.info(f"Fetching team batting for {start_year}-{end_year}")
        df = team_batting(start_year, end_year)
        df.columns = df.columns.str.strip()
        return df

    return cache.get_or_fetch(
        name="team_batting",
        fetch_fn=_fetch,
        start_year=start_year,
        end_year=end_year,
    )


def fetch_team_pitching(
    start_year: int,
    end_year: int | None = None,
    cache: DataCache | None = None,
) -> pd.DataFrame:
    """Fetch team-level pitching statistics.

    Args:
        start_year: First season to fetch.
        end_year: Last season to fetch. Defaults to start_year.
        cache: Cache instance to use. Uses default if not provided.

    Returns:
        DataFrame with team pitching totals.
    """
    end_year = end_year or start_year
    cache = cache or _cache

    def _fetch() -> pd.DataFrame:
        logger.info(f"Fetching team pitching for {start_year}-{end_year}")
        df = team_pitching(start_year, end_year)
        df.columns = df.columns.str.strip()
        return df

    return cache.get_or_fetch(
        name="team_pitching",
        fetch_fn=_fetch,
        start_year=start_year,
        end_year=end_year,
    )


def fetch_team_rosters(
    year: int,
    cache: DataCache | None = None,
) -> pd.DataFrame:
    """Fetch team rosters for a given year.

    This fetches all players who appeared for each team in the given year
    by combining batting and pitching stats.

    Args:
        year: Season year.
        cache: Cache instance to use. Uses default if not provided.

    Returns:
        DataFrame with columns: playerid, Name, Team, Position, PA, IP
    """
    cache = cache or _cache

    def _fetch() -> pd.DataFrame:
        logger.info(f"Building team rosters for {year}")

        # Get all batters (lower qual to capture bench players)
        batters = batting_stats(year, qual=1)
        batters = batters[["Name", "IDfg", "Team", "PA"]].copy()
        batters.columns = ["Name", "playerid", "Team", "PA"]
        batters["Position"] = "Batter"
        batters["IP"] = 0.0

        # Get all pitchers
        pitchers = pitching_stats(year, qual=1)
        pitchers = pitchers[["Name", "IDfg", "Team", "IP"]].copy()
        pitchers.columns = ["Name", "playerid", "Team", "IP"]
        pitchers["Position"] = "Pitcher"
        pitchers["PA"] = 0

        # Combine, handling two-way players by keeping both entries
        roster = pd.concat([batters, pitchers], ignore_index=True)
        roster = roster.sort_values(["Team", "Position", "Name"]).reset_index(drop=True)

        return roster

    return cache.get_or_fetch(
        name="team_rosters",
        fetch_fn=_fetch,
        year=year,
    )


def lookup_player(
    last_name: str,
    first_name: str | None = None,
) -> pd.DataFrame:
    """Look up player ID by name.

    Args:
        last_name: Player's last name.
        first_name: Player's first name (optional, for disambiguation).

    Returns:
        DataFrame with matching players and their IDs.
    """
    if first_name:
        return playerid_lookup(last_name, first_name)
    return playerid_lookup(last_name)


def lookup_player_by_id(
    player_ids: list[int],
    id_type: Literal["fangraphs", "mlbam", "bbref"] = "fangraphs",
) -> pd.DataFrame:
    """Look up player info by ID.

    Args:
        player_ids: List of player IDs.
        id_type: Type of ID ("fangraphs", "mlbam", or "bbref").

    Returns:
        DataFrame with player information.
    """
    key_type_map = {
        "fangraphs": "key_fangraphs",
        "mlbam": "key_mlbam",
        "bbref": "key_bbref",
    }
    return playerid_reverse_lookup(player_ids, key_type=key_type_map[id_type])


def fetch_historical_player_stats(
    player_id: int,
    start_year: int,
    end_year: int,
    player_type: Literal["batter", "pitcher"] = "batter",
    cache: DataCache | None = None,
) -> pd.DataFrame:
    """Fetch multi-year stats for a single player.

    Args:
        player_id: FanGraphs player ID.
        start_year: First season.
        end_year: Last season.
        player_type: "batter" or "pitcher".
        cache: Cache instance to use.

    Returns:
        DataFrame with year-by-year stats for the player.
    """
    cache = cache or _cache

    def _fetch() -> pd.DataFrame:
        if player_type == "batter":
            all_stats = batting_stats(start_year, end_year, qual=0)
        else:
            all_stats = pitching_stats(start_year, end_year, qual=0)

        # Filter to this player
        # FanGraphs uses 'IDfg' for player ID
        if "IDfg" in all_stats.columns:
            player_stats = all_stats[all_stats["IDfg"] == player_id].copy()
        else:
            player_stats = all_stats[all_stats["playerid"] == player_id].copy()

        return player_stats

    return cache.get_or_fetch(
        name=f"player_stats_{player_type}",
        fetch_fn=_fetch,
        player_id=player_id,
        start_year=start_year,
        end_year=end_year,
    )
