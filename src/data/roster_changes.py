"""Parse roster change inputs (trades, signings, retirements)."""

import logging
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Iterator, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Type of roster change."""

    TRADE = "trade"
    SIGNING = "signing"
    RETIREMENT = "retirement"
    RELEASE = "release"
    CALLUP = "callup"
    DEMOTION = "demotion"


@dataclass
class RosterChange:
    """Represents a single roster change."""

    player_name: str
    player_id: Optional[int]  # FanGraphs ID if known
    change_type: ChangeType
    from_team: Optional[str]  # None for signings from free agency
    to_team: Optional[str]  # None for retirements/releases
    effective_date: Optional[date]
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate the roster change."""
        if self.change_type == ChangeType.SIGNING and self.from_team is not None:
            logger.warning(
                f"Signing for {self.player_name} has from_team set; "
                "assuming free agent signing"
            )
        if self.change_type == ChangeType.RETIREMENT and self.to_team is not None:
            logger.warning(
                f"Retirement for {self.player_name} has to_team set; ignoring"
            )
            self.to_team = None


def load_roster_changes(
    filepath: Union[Path, str],
    validate_players: bool = False,
) -> List[RosterChange]:
    """Load roster changes from a CSV file.

    Expected CSV columns:
        - player_name: Full name of the player
        - player_id: FanGraphs player ID (optional)
        - change_type: One of trade, signing, retirement, release, callup, demotion
        - from_team: Team abbreviation player is leaving (optional)
        - to_team: Team abbreviation player is joining (optional)
        - effective_date: Date of the change (YYYY-MM-DD format, optional)
        - notes: Additional notes (optional)

    Args:
        filepath: Path to the CSV file.
        validate_players: If True, validate player IDs exist in database.

    Returns:
        List of RosterChange objects.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Roster changes file not found: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.strip()

    required_cols = {"player_name", "change_type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    changes = []
    for _, row in df.iterrows():
        # Skip rows with missing player_name
        if pd.isna(row.get("player_name")):
            continue

        player_name = str(row["player_name"]).strip()
        if not player_name:
            continue

        try:
            change_type = ChangeType(str(row["change_type"]).lower().strip())
        except ValueError:
            logger.warning(
                f"Unknown change type '{row['change_type']}' for {player_name}"
            )
            continue

        # Parse optional fields
        player_id = (
            int(row["player_id"])
            if pd.notna(row.get("player_id"))
            else None
        )
        from_team = (
            str(row["from_team"]).strip().upper()
            if pd.notna(row.get("from_team"))
            else None
        )
        to_team = (
            str(row["to_team"]).strip().upper()
            if pd.notna(row.get("to_team"))
            else None
        )
        effective_date = (
            pd.to_datetime(row["effective_date"]).date()
            if pd.notna(row.get("effective_date"))
            else None
        )
        notes = str(row.get("notes")) if pd.notna(row.get("notes")) else None

        change = RosterChange(
            player_name=player_name,
            player_id=player_id,
            change_type=change_type,
            from_team=from_team,
            to_team=to_team,
            effective_date=effective_date,
            notes=notes,
        )
        changes.append(change)

    logger.info(f"Loaded {len(changes)} roster changes from {filepath}")
    return changes


def apply_roster_changes(
    base_roster: pd.DataFrame,
    changes: List[RosterChange],
) -> pd.DataFrame:
    """Apply roster changes to a base roster.

    Args:
        base_roster: DataFrame with columns: playerid, Name, Team, Position
        changes: List of roster changes to apply.

    Returns:
        Updated roster DataFrame.
    """
    roster = base_roster.copy()

    for change in changes:
        # Find player in roster (by ID if available, else by name)
        if change.player_id:
            mask = roster["playerid"] == change.player_id
        else:
            mask = roster["Name"].str.lower() == change.player_name.lower()

        if change.change_type == ChangeType.RETIREMENT:
            # Remove player from roster
            roster = roster[~mask]
            logger.debug(f"Removed {change.player_name} (retirement)")

        elif change.change_type == ChangeType.RELEASE:
            # Remove player from roster
            roster = roster[~mask]
            logger.debug(f"Removed {change.player_name} (release)")

        elif change.change_type in (ChangeType.TRADE, ChangeType.SIGNING):
            if mask.any():
                # Update existing player's team
                roster.loc[mask, "Team"] = change.to_team
                logger.debug(f"Moved {change.player_name} to {change.to_team}")
            else:
                # Player not in roster (new signing) - add placeholder
                logger.warning(
                    f"Player {change.player_name} not found in base roster; "
                    "will need manual stats entry"
                )

        elif change.change_type == ChangeType.CALLUP:
            if not mask.any():
                # Add new player (minor leaguer called up)
                logger.info(
                    f"Callup: {change.player_name} to {change.to_team}; "
                    "will need stats projection"
                )

        elif change.change_type == ChangeType.DEMOTION:
            # Remove from MLB roster
            roster = roster[~mask]
            logger.debug(f"Removed {change.player_name} (demotion)")

    return roster.reset_index(drop=True)


def create_sample_roster_changes_csv(output_path: Union[Path, str]) -> None:
    """Create a sample roster changes CSV template.

    Args:
        output_path: Where to save the sample file.
    """
    sample_data = {
        "player_name": [
            "Juan Soto",
            "Shohei Ohtani",
            "Max Scherzer",
            "Sample Player",
        ],
        "player_id": [20123, 19755, 3137, None],
        "change_type": ["signing", "signing", "trade", "retirement"],
        "from_team": [None, None, "TEX", "NYY"],
        "to_team": ["NYM", "LAD", "BOS", None],
        "effective_date": ["2024-12-15", "2023-12-09", "2024-07-30", "2024-10-01"],
        "notes": [
            "15-year, $765M contract",
            "10-year, $700M contract",
            "Deadline trade",
            "Retired after 15 seasons",
        ],
    }

    df = pd.DataFrame(sample_data)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Created sample roster changes CSV at {output_path}")


# Team abbreviation mappings for standardization
TEAM_ABBREVIATIONS = {
    # Standard abbreviations
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CHW": "CHW", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KCR": "KCR",
    "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SDP": "SDP", "SEA": "SEA",
    "SFG": "SFG", "STL": "STL", "TBR": "TBR", "TEX": "TEX",
    "TOR": "TOR", "WSN": "WSN",
    # Common alternates
    "AZ": "ARI", "KC": "KCR", "SD": "SDP", "SF": "SFG",
    "TB": "TBR", "WAS": "WSN", "WSH": "WSN",
    "CWS": "CHW", "CHI": "CHC",  # Ambiguous, default to Cubs
}


def standardize_team(team: Optional[str]) -> Optional[str]:
    """Standardize team abbreviation.

    Args:
        team: Team abbreviation to standardize.

    Returns:
        Standardized 3-letter abbreviation or None.
    """
    if team is None:
        return None
    return TEAM_ABBREVIATIONS.get(team.upper().strip(), team.upper().strip())
