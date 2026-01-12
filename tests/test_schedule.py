"""Tests for Strength of Schedule calculations."""

import pytest

from src.projections.schedule import (
    get_division_opponents,
    get_league_opponents,
    get_interleague_opponents,
    calculate_opponent_strength,
    calculate_sos,
    calculate_all_sos,
    apply_sos_adjustments,
    ScheduleBreakdown,
)
from src.projections.projector import TeamProjectionSet
from src.projections.team import DIVISIONS, TEAM_TO_DIVISION


class TestDivisionOpponents:
    """Tests for getting division opponents."""

    def test_al_east_has_four_opponents(self):
        """AL East team should have 4 division opponents."""
        opponents = get_division_opponents("NYY")
        assert len(opponents) == 4
        assert "NYY" not in opponents  # Should not include self
        assert "BOS" in opponents
        assert "TOR" in opponents
        assert "TBR" in opponents
        assert "BAL" in opponents

    def test_nl_west_has_four_opponents(self):
        """NL West team should have 4 division opponents."""
        opponents = get_division_opponents("LAD")
        assert len(opponents) == 4
        assert "LAD" not in opponents
        assert "SDP" in opponents
        assert "SFG" in opponents

    def test_unknown_team_returns_empty(self):
        """Unknown team should return empty list."""
        opponents = get_division_opponents("XXX")
        assert opponents == []


class TestLeagueOpponents:
    """Tests for getting same-league opponents."""

    def test_league_opponents_excludes_division(self):
        """League opponents should exclude division rivals by default."""
        opponents = get_league_opponents("NYY", exclude_division=True)
        # Should have 10 AL teams (15 total - 5 in division)
        assert len(opponents) == 10
        assert "BOS" not in opponents  # Division rival
        assert "CLE" in opponents  # AL Central
        assert "HOU" in opponents  # AL West

    def test_league_opponents_includes_division(self):
        """League opponents can include division if specified."""
        opponents = get_league_opponents("NYY", exclude_division=False)
        assert len(opponents) == 14  # All AL teams except self
        assert "BOS" in opponents


class TestInterleagueOpponents:
    """Tests for getting interleague opponents."""

    def test_al_team_gets_nl_opponents(self):
        """AL team should get all NL teams as interleague."""
        opponents = get_interleague_opponents("NYY")
        assert len(opponents) == 15
        # Should all be NL teams
        assert "LAD" in opponents
        assert "ATL" in opponents
        assert "CHC" in opponents
        # Should not include AL teams
        assert "NYY" not in opponents
        assert "BOS" not in opponents

    def test_nl_team_gets_al_opponents(self):
        """NL team should get all AL teams as interleague."""
        opponents = get_interleague_opponents("LAD")
        assert len(opponents) == 15
        assert "NYY" in opponents
        assert "HOU" in opponents


class TestOpponentStrength:
    """Tests for opponent strength calculation."""

    def test_neutral_opponents(self):
        """Teams with 0.500 win% should give 0.500 SOS."""
        team_win_pcts = {"BOS": 0.500, "TOR": 0.500, "TBR": 0.500, "BAL": 0.500}
        strength = calculate_opponent_strength(
            ["BOS", "TOR", "TBR", "BAL"],
            team_win_pcts
        )
        assert strength == 0.500

    def test_strong_opponents(self):
        """Strong opponents should give high SOS."""
        team_win_pcts = {"BOS": 0.600, "TOR": 0.580, "TBR": 0.560, "BAL": 0.620}
        strength = calculate_opponent_strength(
            ["BOS", "TOR", "TBR", "BAL"],
            team_win_pcts
        )
        assert strength > 0.550
        assert strength == pytest.approx(0.590, abs=0.001)

    def test_weak_opponents(self):
        """Weak opponents should give low SOS."""
        team_win_pcts = {"BOS": 0.400, "TOR": 0.380}
        strength = calculate_opponent_strength(["BOS", "TOR"], team_win_pcts)
        assert strength < 0.500
        assert strength == pytest.approx(0.390, abs=0.001)

    def test_missing_team_defaults_to_500(self):
        """Missing team in dict should default to 0.500."""
        team_win_pcts = {"BOS": 0.600}
        strength = calculate_opponent_strength(["BOS", "MISSING"], team_win_pcts)
        assert strength == pytest.approx(0.550, abs=0.001)

    def test_empty_opponents(self):
        """Empty opponent list should return 0.500."""
        strength = calculate_opponent_strength([], {})
        assert strength == 0.500


class TestCalculateSOS:
    """Tests for full SOS calculation."""

    def create_mock_projections(self, win_dict):
        """Create mock TeamProjectionSet objects from win dictionary."""
        projections = {}
        for team, wins in win_dict.items():
            proj = TeamProjectionSet(
                team=team,
                division=TEAM_TO_DIVISION.get(team, "Unknown"),
                projected_wins=wins,
                projected_runs_scored=700,
                projected_runs_allowed=700,
            )
            projections[team] = proj
        return projections

    def test_sos_calculation_structure(self):
        """SOS calculation should return proper structure."""
        # Create equal projections for all teams
        wins = {team: 81 for div in DIVISIONS.values() for team in div}
        projections = self.create_mock_projections(wins)

        sos = calculate_sos("NYY", projections)

        assert isinstance(sos, ScheduleBreakdown)
        assert sos.team == "NYY"
        assert sos.division == "AL East"
        assert sos.divisional_games == 52
        assert sos.non_divisional_league_games == 66
        assert sos.interleague_games == 44

    def test_equal_teams_give_neutral_sos(self):
        """Equal teams should give neutral SOS (0.500)."""
        wins = {team: 81 for div in DIVISIONS.values() for team in div}
        projections = self.create_mock_projections(wins)

        sos = calculate_sos("NYY", projections)

        assert sos.overall_sos == pytest.approx(0.500, abs=0.001)
        assert abs(sos.sos_win_adjustment) < 0.1

    def test_tough_division_gives_high_sos(self):
        """Team in tough division should have higher SOS."""
        # Make AL East very strong except NYY
        wins = {team: 81 for div in DIVISIONS.values() for team in div}
        wins["BOS"] = 100
        wins["TOR"] = 95
        wins["TBR"] = 95
        wins["BAL"] = 90

        projections = self.create_mock_projections(wins)
        sos = calculate_sos("NYY", projections)

        assert sos.overall_sos > 0.500
        assert sos.divisional_opponent_strength > 0.550
        assert sos.sos_win_adjustment < 0  # Negative adjustment for hard schedule

    def test_weak_division_gives_low_sos(self):
        """Team in weak division should have lower SOS."""
        wins = {team: 81 for div in DIVISIONS.values() for team in div}
        # Make AL East very weak except NYY
        wins["BOS"] = 60
        wins["TOR"] = 65
        wins["TBR"] = 65
        wins["BAL"] = 60

        projections = self.create_mock_projections(wins)
        sos = calculate_sos("NYY", projections)

        assert sos.overall_sos < 0.500
        assert sos.divisional_opponent_strength < 0.450
        assert sos.sos_win_adjustment > 0  # Positive adjustment for easy schedule


class TestApplySOSAdjustments:
    """Tests for applying SOS adjustments to projections."""

    def create_mock_projections(self, win_dict):
        """Create mock TeamProjectionSet objects from win dictionary."""
        projections = {}
        for team, wins in win_dict.items():
            proj = TeamProjectionSet(
                team=team,
                division=TEAM_TO_DIVISION.get(team, "Unknown"),
                projected_wins=wins,
                projected_runs_scored=700,
                projected_runs_allowed=700,
            )
            projections[team] = proj
        return projections

    def test_adjustments_preserve_total_wins(self):
        """SOS adjustments should approximately preserve total wins across league."""
        wins = {team: 81 for div in DIVISIONS.values() for team in div}
        projections = self.create_mock_projections(wins)

        total_before = sum(p.projected_wins for p in projections.values())

        apply_sos_adjustments(projections)

        total_after = sum(p.projected_wins for p in projections.values())

        # Total wins should be roughly preserved (small rounding differences OK)
        assert abs(total_after - total_before) < 5

    def test_adjustments_set_fields(self):
        """SOS adjustments should set SOS fields on projections."""
        wins = {team: 81 for div in DIVISIONS.values() for team in div}
        projections = self.create_mock_projections(wins)

        apply_sos_adjustments(projections)

        for proj in projections.values():
            assert hasattr(proj, 'sos')
            assert hasattr(proj, 'sos_win_adjustment')
            assert hasattr(proj, 'projected_wins_pre_sos')
            assert proj.sos != 0  # Should be set
            assert proj.projected_wins_pre_sos > 0

    def test_hard_schedule_reduces_wins(self):
        """Team with hard schedule should have wins reduced."""
        wins = {team: 81 for div in DIVISIONS.values() for team in div}
        # Make AL East strong except NYY
        wins["BOS"] = 100
        wins["TOR"] = 95
        wins["TBR"] = 95
        wins["BAL"] = 90

        projections = self.create_mock_projections(wins)
        original_nyy_wins = projections["NYY"].projected_wins

        apply_sos_adjustments(projections)

        # NYY should have wins reduced due to hard schedule
        assert projections["NYY"].projected_wins < original_nyy_wins
        assert projections["NYY"].sos > 0.500

    def test_easy_schedule_increases_wins(self):
        """Team with easy schedule should have wins increased."""
        wins = {team: 81 for div in DIVISIONS.values() for team in div}
        # Make AL East weak except NYY
        wins["BOS"] = 60
        wins["TOR"] = 65
        wins["TBR"] = 65
        wins["BAL"] = 60

        projections = self.create_mock_projections(wins)
        original_nyy_wins = projections["NYY"].projected_wins

        apply_sos_adjustments(projections)

        # NYY should have wins increased due to easy schedule
        assert projections["NYY"].projected_wins > original_nyy_wins
        assert projections["NYY"].sos < 0.500


class TestScheduleBreakdown:
    """Tests for ScheduleBreakdown dataclass."""

    def test_game_totals(self):
        """Game totals should add up to 162."""
        breakdown = ScheduleBreakdown(
            team="NYY",
            division="AL East",
            divisional_games=52,
            non_divisional_league_games=66,
            interleague_games=44,
        )
        total = (
            breakdown.divisional_games +
            breakdown.non_divisional_league_games +
            breakdown.interleague_games
        )
        assert total == 162
