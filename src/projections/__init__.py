"""Player and team projection modules."""

from .player import (
    PlayerProjection,
    BatterProjection,
    PitcherProjection,
    project_player,
    project_batter,
    project_pitcher,
    calculate_counting_stats,
    regress_to_mean,
    calculate_weighted_average,
)

from .playing_time import (
    PlayingTimeProjection,
    project_playing_time,
    estimate_role,
    allocate_team_playing_time,
    BATTER_ROLES,
    PITCHER_ROLES,
)

from .adjustments import (
    get_aging_adjustment,
    get_park_factor,
    adjust_for_park_change,
    get_league_average,
    PARK_FACTORS,
    LEAGUE_AVERAGES,
)

from .team import (
    TeamProjection,
    pythagorean_wins,
    DIVISIONS,
    TEAM_TO_DIVISION,
)

from .projector import (
    Projector,
    ProjectionConfig,
    TeamProjectionSet,
    projections_to_dataframe,
)

__all__ = [
    # Player projections
    "PlayerProjection",
    "BatterProjection",
    "PitcherProjection",
    "project_player",
    "project_batter",
    "project_pitcher",
    "calculate_counting_stats",
    "regress_to_mean",
    "calculate_weighted_average",
    # Playing time
    "PlayingTimeProjection",
    "project_playing_time",
    "estimate_role",
    "allocate_team_playing_time",
    "BATTER_ROLES",
    "PITCHER_ROLES",
    # Adjustments
    "get_aging_adjustment",
    "get_park_factor",
    "adjust_for_park_change",
    "get_league_average",
    "PARK_FACTORS",
    "LEAGUE_AVERAGES",
    # Team
    "TeamProjection",
    "pythagorean_wins",
    "DIVISIONS",
    "TEAM_TO_DIVISION",
    # Projector
    "Projector",
    "ProjectionConfig",
    "TeamProjectionSet",
    "projections_to_dataframe",
]
