"""Tests for automatic roster synchronization."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.data.mlb_api import (
    detect_roster_changes,
    normalize_name,
    normalize_team_abbrev,
    MLBPlayer,
)
from src.data.roster_changes import RosterChange, ChangeType
from src.projections.projector import ProjectionConfig


class TestNormalizeName:
    """Tests for player name normalization."""

    def test_removes_accents(self):
        """Should remove accented characters."""
        assert normalize_name("José Ramírez") == "jose ramirez"
        assert normalize_name("Félix Hernández") == "felix hernandez"
        assert normalize_name("Yoán Moncada") == "yoan moncada"

    def test_lowercase_and_strip(self):
        """Should lowercase and strip whitespace."""
        assert normalize_name("  Mike Trout  ") == "mike trout"
        assert normalize_name("SHOHEI OHTANI") == "shohei ohtani"

    def test_empty_string(self):
        """Should handle empty strings."""
        assert normalize_name("") == ""
        assert normalize_name(None) == ""


class TestNormalizeTeamAbbrev:
    """Tests for team abbreviation normalization."""

    def test_fangraphs_to_mlb(self):
        """Should convert FanGraphs abbreviations to MLB standard."""
        assert normalize_team_abbrev("ATH") == "OAK"
        assert normalize_team_abbrev("FLA") == "MIA"
        assert normalize_team_abbrev("ANA") == "LAA"

    def test_standard_unchanged(self):
        """Standard abbreviations should remain unchanged."""
        assert normalize_team_abbrev("NYY") == "NYY"
        assert normalize_team_abbrev("LAD") == "LAD"
        assert normalize_team_abbrev("BOS") == "BOS"

    def test_empty_unchanged(self):
        """Empty string should return empty."""
        assert normalize_team_abbrev("") == ""


class TestDetectRosterChanges:
    """Tests for roster change detection."""

    def create_mock_batting_df(self, players):
        """Create a mock batting DataFrame.

        players: list of (name, team, player_id, season)
        """
        return pd.DataFrame([
            {"Name": name, "Team": team, "IDfg": pid, "Season": season}
            for name, team, pid, season in players
        ])

    def create_mock_pitching_df(self, players):
        """Create a mock pitching DataFrame.

        players: list of (name, team, player_id, season)
        """
        return pd.DataFrame([
            {"Name": name, "Team": team, "IDfg": pid, "Season": season}
            for name, team, pid, season in players
        ])

    def test_detects_team_change(self):
        """Should detect when a player changed teams."""
        batting = self.create_mock_batting_df([
            ("Juan Soto", "SDP", 20123, 2024),
        ])
        pitching = self.create_mock_pitching_df([])

        # Mock MLB API returning Soto on NYM
        mock_rosters = {
            "NYM": [MLBPlayer(mlb_id=665742, name="Juan Soto", team="NYM", position="RF")]
        }

        with patch("src.data.mlb_api.fetch_all_rosters", return_value=mock_rosters):
            with patch("src.data.mlb_api.check_api_available", return_value=True):
                changes = detect_roster_changes(batting, pitching)

        assert len(changes) == 1
        change = changes[0]
        assert change.player_name == "Juan Soto"
        assert change.from_team == "SDP"
        assert change.to_team == "NYM"
        assert change.change_type == ChangeType.TRADE

    def test_no_change_same_team(self):
        """Should not detect change if player is on same team."""
        batting = self.create_mock_batting_df([
            ("Aaron Judge", "NYY", 12345, 2024),
        ])
        pitching = self.create_mock_pitching_df([])

        mock_rosters = {
            "NYY": [MLBPlayer(mlb_id=592450, name="Aaron Judge", team="NYY", position="RF")]
        }

        with patch("src.data.mlb_api.fetch_all_rosters", return_value=mock_rosters):
            with patch("src.data.mlb_api.check_api_available", return_value=True):
                changes = detect_roster_changes(batting, pitching)

        assert len(changes) == 0

    def test_handles_name_normalization(self):
        """Should match players despite accent differences."""
        batting = self.create_mock_batting_df([
            ("Jose Ramirez", "CLE", 12345, 2024),  # No accent
        ])
        pitching = self.create_mock_pitching_df([])

        # API returns with accent
        mock_rosters = {
            "NYY": [MLBPlayer(mlb_id=608070, name="José Ramírez", team="NYY", position="3B")]
        }

        with patch("src.data.mlb_api.fetch_all_rosters", return_value=mock_rosters):
            with patch("src.data.mlb_api.check_api_available", return_value=True):
                changes = detect_roster_changes(batting, pitching)

        assert len(changes) == 1
        assert changes[0].from_team == "CLE"
        assert changes[0].to_team == "NYY"

    def test_empty_when_api_unavailable(self):
        """Should return empty list when API is unavailable."""
        batting = self.create_mock_batting_df([
            ("Juan Soto", "SDP", 20123, 2024),
        ])
        pitching = self.create_mock_pitching_df([])

        with patch("src.data.mlb_api.check_api_available", return_value=False):
            changes = detect_roster_changes(batting, pitching)

        assert changes == []

    def test_empty_when_rosters_fail(self):
        """Should return empty list when roster fetch fails."""
        batting = self.create_mock_batting_df([
            ("Juan Soto", "SDP", 20123, 2024),
        ])
        pitching = self.create_mock_pitching_df([])

        with patch("src.data.mlb_api.check_api_available", return_value=True):
            with patch("src.data.mlb_api.fetch_all_rosters", return_value={}):
                changes = detect_roster_changes(batting, pitching)

        assert changes == []

    def test_multiple_changes(self):
        """Should detect multiple roster changes."""
        batting = self.create_mock_batting_df([
            ("Player A", "NYY", 1, 2024),
            ("Player B", "BOS", 2, 2024),
        ])
        pitching = self.create_mock_pitching_df([])

        mock_rosters = {
            "LAD": [
                MLBPlayer(mlb_id=101, name="Player A", team="LAD", position="1B"),
                MLBPlayer(mlb_id=102, name="Player B", team="LAD", position="2B"),
            ]
        }

        with patch("src.data.mlb_api.fetch_all_rosters", return_value=mock_rosters):
            with patch("src.data.mlb_api.check_api_available", return_value=True):
                changes = detect_roster_changes(batting, pitching)

        assert len(changes) == 2
        teams_changed_to = {c.to_team for c in changes}
        assert teams_changed_to == {"LAD"}


class TestProjectionConfigAutoSync:
    """Tests for ProjectionConfig auto_sync_rosters flag."""

    def test_default_is_false(self):
        """Auto sync should be disabled by default."""
        config = ProjectionConfig()
        assert config.auto_sync_rosters is False

    def test_can_enable(self):
        """Should be able to enable auto sync."""
        config = ProjectionConfig(auto_sync_rosters=True)
        assert config.auto_sync_rosters is True
