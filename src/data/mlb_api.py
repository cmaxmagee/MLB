"""MLB Stats API integration for current roster data.

Fetches current 40-man rosters and transactions from the official MLB API
to keep projections up-to-date with offseason moves.
"""

import logging
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    import statsapi
except ImportError:
    statsapi = None

logger = logging.getLogger(__name__)

# MLB team ID to abbreviation mapping
MLB_TEAM_IDS = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KCR", 119: "LAD", 120: "WSN", 121: "NYM", 133: "OAK",
    134: "PIT", 135: "SDP", 136: "SEA", 137: "SFG", 138: "STL",
    139: "TBR", 140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CHW", 146: "MIA", 147: "NYY", 158: "MIL",
}

# Reverse mapping
ABBREV_TO_MLB_ID = {v: k for k, v in MLB_TEAM_IDS.items()}

# FanGraphs uses different abbreviations for some teams
# Map FanGraphs abbreviations to standard MLB abbreviations
FANGRAPHS_TO_MLB_ABBREV = {
    "ATH": "OAK",   # Athletics
    "FLA": "MIA",   # Florida Marlins -> Miami Marlins
    "ANA": "LAA",   # Anaheim Angels -> LA Angels
    "TBD": "TBR",   # Tampa Bay Devil Rays -> Rays
    "MON": "WSN",   # Montreal Expos -> Washington Nationals
    "CAL": "LAA",   # California Angels
    "KCA": "KCR",   # Kansas City
    "SDN": "SDP",   # San Diego
    "SFN": "SFG",   # San Francisco
    "NYA": "NYY",   # NY American League
    "NYN": "NYM",   # NY National League
    "CHA": "CHW",   # Chicago American League
    "CHN": "CHC",   # Chicago National League
    "LAN": "LAD",   # LA National League
    "SLN": "STL",   # St. Louis National League
    "WAS": "WSN",   # Washington
}


def normalize_team_abbrev(team: str) -> str:
    """Normalize team abbreviation to standard MLB format.

    FanGraphs and other sources use different abbreviations.
    This ensures consistent team matching.
    """
    if not team:
        return team
    team_upper = team.upper().strip()
    return FANGRAPHS_TO_MLB_ABBREV.get(team_upper, team_upper)


def normalize_name(name: str) -> str:
    """Normalize player name for comparison.

    Removes accents and special characters to ensure matching between
    different data sources (e.g., "Félix" -> "Felix", "Pagán" -> "Pagan").
    """
    if not name:
        return ""
    # Normalize unicode to decomposed form (separates base char from accent)
    # Then encode to ASCII, ignoring non-ASCII characters
    normalized = unicodedata.normalize("NFD", name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_name.lower().strip()


@dataclass
class MLBPlayer:
    """Player data from MLB API."""

    mlb_id: int
    name: str
    team: str
    position: str
    jersey_number: Optional[str] = None
    status: str = "Active"  # Active, 60-Day IL, etc.


def check_api_available() -> bool:
    """Check if MLB Stats API is available."""
    if statsapi is None:
        logger.error(
            "MLB-StatsAPI not installed. Run: pip install MLB-StatsAPI"
        )
        return False
    return True


def fetch_current_roster(team: str) -> List[MLBPlayer]:
    """Fetch current 40-man roster for a team.

    Args:
        team: Team abbreviation (e.g., "NYY").

    Returns:
        List of MLBPlayer objects on the 40-man roster.
    """
    if not check_api_available():
        return []

    team_id = ABBREV_TO_MLB_ID.get(team.upper())
    if team_id is None:
        logger.error(f"Unknown team abbreviation: {team}")
        return []

    try:
        # Use the JSON API for structured data
        roster_data = statsapi.get(
            "team_roster",
            {"teamId": team_id, "rosterType": "40Man"}
        )

        players = []
        for entry in roster_data.get("roster", []):
            person = entry.get("person", {})
            position = entry.get("position", {})
            status = entry.get("status", {})

            player_name = person.get("fullName", "")
            if not player_name:
                continue

            players.append(MLBPlayer(
                mlb_id=person.get("id", 0),
                name=player_name,
                team=team,
                position=position.get("abbreviation", ""),
                jersey_number=entry.get("jerseyNumber", ""),
                status=status.get("description", "Active"),
            ))

        logger.info(f"Fetched {len(players)} players for {team}")
        return players

    except Exception as e:
        logger.error(f"Error fetching roster for {team}: {e}")
        return []


def fetch_all_rosters() -> Dict[str, List[MLBPlayer]]:
    """Fetch current 40-man rosters for all 30 teams.

    Returns:
        Dict mapping team abbreviation to list of players.
    """
    if not check_api_available():
        return {}

    all_rosters = {}

    for team in sorted(MLB_TEAM_IDS.values()):
        roster = fetch_current_roster(team)
        if roster:
            all_rosters[team] = roster

    logger.info(f"Fetched rosters for {len(all_rosters)} teams")
    return all_rosters


def fetch_transactions(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """Fetch recent MLB transactions.

    Args:
        start_date: Start date for transactions (default: 30 days ago).
        end_date: End date for transactions (default: today).

    Returns:
        DataFrame with transaction data.
    """
    if not check_api_available():
        return pd.DataFrame()

    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = date(end_date.year - 1, 11, 1)  # Start of offseason

    try:
        # Fetch transactions per team (more reliable than date-range query)
        all_transactions = []

        for team_id, team_abbrev in MLB_TEAM_IDS.items():
            try:
                transactions = statsapi.get(
                    "transactions",
                    {"teamId": team_id}
                )

                for txn in transactions.get("transactions", []):
                    txn_date_str = txn.get("date", "")
                    if txn_date_str:
                        try:
                            txn_date = datetime.strptime(txn_date_str, "%Y-%m-%d").date()
                            if not (start_date <= txn_date <= end_date):
                                continue
                        except ValueError:
                            continue

                    player = txn.get("player", {})
                    from_team = txn.get("fromTeam", {})
                    to_team = txn.get("toTeam", {})

                    # Try multiple ways to get player name
                    description = txn.get("description", "")
                    player_name = (
                        player.get("fullName") or
                        player.get("name") or
                        txn.get("person", {}).get("fullName") or
                        txn.get("name") or
                        # Parse from description (usually starts with player name)
                        (description.split(" signed")[0] if " signed" in description else None) or
                        (description.split(" traded")[0] if " traded" in description else None) or
                        (description.split(" claimed")[0] if " claimed" in description else None) or
                        (description.split(" designated")[0] if " designated" in description else None) or
                        (description.split(" outrighted")[0] if " outrighted" in description else None) or
                        (description.split(" released")[0] if " released" in description else None) or
                        (description.split(" assigned")[0] if " assigned" in description else None)
                    )

                    all_transactions.append({
                        "date": txn_date_str,
                        "type": txn.get("typeDesc"),
                        "player_name": player_name,
                        "player_id": player.get("id"),
                        "from_team": MLB_TEAM_IDS.get(from_team.get("id"), ""),
                        "to_team": MLB_TEAM_IDS.get(to_team.get("id"), ""),
                        "description": description,
                    })
            except Exception as team_err:
                logger.debug(f"Error fetching transactions for {team_abbrev}: {team_err}")
                continue

        rows = all_transactions

        df = pd.DataFrame(rows)
        logger.info(f"Fetched {len(df)} transactions from {start_date} to {end_date}")
        return df

    except Exception as e:
        logger.error(f"Error fetching transactions: {e}")
        return pd.DataFrame()


def compare_rosters(
    current_rosters: Dict[str, List[MLBPlayer]],
    historical_players: pd.DataFrame,
) -> pd.DataFrame:
    """Compare current MLB rosters with historical FanGraphs data.

    Identifies players who have changed teams or are new.

    Args:
        current_rosters: Dict of current rosters from MLB API.
        historical_players: DataFrame with columns: Name, Team, playerid

    Returns:
        DataFrame of roster changes in the format needed for projections.
    """
    changes = []

    # Build lookup of historical players by normalized name (removes accents)
    historical_by_name = {}
    for _, row in historical_players.iterrows():
        name = row.get("Name", "").strip()
        if name:
            name_key = normalize_name(name)
            historical_by_name[name_key] = {
                "team": row.get("Team", ""),
                "playerid": row.get("playerid", row.get("IDfg", 0)),
                "original_name": name,
            }

    # Check each player in current rosters
    for team, players in current_rosters.items():
        current_team_normalized = normalize_team_abbrev(team)

        for player in players:
            name_key = normalize_name(player.name)

            if name_key in historical_by_name:
                hist = historical_by_name[name_key]
                old_team = hist["team"]
                old_team_normalized = normalize_team_abbrev(old_team)

                # Player changed teams (compare normalized abbreviations)
                if old_team and old_team_normalized != current_team_normalized:
                    changes.append({
                        "player_name": player.name,
                        "player_id": hist["playerid"],
                        "change_type": "trade",
                        "from_team": old_team_normalized,
                        "to_team": current_team_normalized,
                        "effective_date": "",
                        "notes": f"Moved from {old_team_normalized} to {current_team_normalized}",
                    })
            else:
                # New player not in historical data (free agent signing or callup)
                changes.append({
                    "player_name": player.name,
                    "player_id": "",
                    "change_type": "signing",
                    "from_team": "",
                    "to_team": current_team_normalized,
                    "effective_date": "",
                    "notes": "New to team (no historical data)",
                })

    # Check for players who left (in historical but not current)
    current_names = set()
    for players in current_rosters.values():
        for p in players:
            current_names.add(normalize_name(p.name))

    for name_key, hist in historical_by_name.items():
        if name_key not in current_names and hist["team"]:
            # Player no longer on any 40-man roster
            # Could be release, retirement, or minor leagues
            # Skip for now - too many false positives
            pass

    df = pd.DataFrame(changes)
    logger.info(f"Identified {len(df)} roster changes")
    return df


def sync_rosters_to_csv(
    historical_batting: pd.DataFrame,
    historical_pitching: pd.DataFrame,
    output_path: Path,
) -> Tuple[int, Path]:
    """Sync current rosters and generate roster changes CSV.

    Args:
        historical_batting: Historical batting stats DataFrame.
        historical_pitching: Historical pitching stats DataFrame.
        output_path: Path to save the roster changes CSV.

    Returns:
        Tuple of (number of changes, output path).
    """
    # Fetch current rosters
    logger.info("Fetching current 40-man rosters from MLB API...")
    current_rosters = fetch_all_rosters()

    if not current_rosters:
        logger.error("Failed to fetch rosters")
        return 0, output_path

    # Combine historical data (pybaseball uses IDfg for player ID)
    historical_players = pd.concat([
        historical_batting[["Name", "Team", "IDfg"]].drop_duplicates(),
        historical_pitching[["Name", "Team", "IDfg"]].drop_duplicates(),
    ]).drop_duplicates(subset=["Name"])

    # Get most recent team for each player
    most_recent_batting = historical_batting.sort_values("Season", ascending=False)
    most_recent_pitching = historical_pitching.sort_values("Season", ascending=False)

    recent_teams = pd.concat([
        most_recent_batting.groupby("Name").first()[["Team", "IDfg"]].reset_index(),
        most_recent_pitching.groupby("Name").first()[["Team", "IDfg"]].reset_index(),
    ]).drop_duplicates(subset=["Name"])

    # Compare and generate changes
    changes_df = compare_rosters(current_rosters, recent_teams)

    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    changes_df.to_csv(output_path, index=False)

    logger.info(f"Saved {len(changes_df)} roster changes to {output_path}")
    return len(changes_df), output_path


def get_player_info(player_name: str) -> Optional[Dict]:
    """Look up player info from MLB API.

    Args:
        player_name: Player's full name.

    Returns:
        Dict with player info or None if not found.
    """
    if not check_api_available():
        return None

    try:
        results = statsapi.lookup_player(player_name)
        if results:
            player = results[0]
            return {
                "mlb_id": player.get("id"),
                "name": player.get("fullName"),
                "position": player.get("primaryPosition", {}).get("abbreviation"),
                "team": MLB_TEAM_IDS.get(
                    player.get("currentTeam", {}).get("id"), ""
                ),
                "birth_date": player.get("birthDate"),
            }
    except Exception as e:
        logger.error(f"Error looking up player {player_name}: {e}")

    return None
