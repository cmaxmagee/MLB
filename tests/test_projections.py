"""Tests for player projection module."""

import numpy as np
import pandas as pd
import pytest

from src.projections.player import (
    regress_to_mean,
    calculate_weighted_average,
    project_batter,
    project_pitcher,
    BatterProjection,
    PitcherProjection,
    REGRESSION_CONSTANTS,
)
from src.projections.playing_time import (
    detect_batter_role,
    detect_pitcher_role,
    project_playing_time,
    get_health_multiplier,
    BATTER_ROLES,
    PITCHER_ROLES,
)
from src.projections.adjustments import (
    get_aging_adjustment,
    get_park_factor,
    adjust_for_park_change,
)
from src.projections.team import pythagorean_wins


class TestRegressionToMean:
    """Tests for regression to mean calculations."""

    def test_full_regression_with_zero_sample(self):
        """With zero sample size, should return league mean."""
        result = regress_to_mean(
            observed_value=0.350,
            sample_size=0,
            league_mean=0.250,
            regression_constant=400,
        )
        assert result == 0.250

    def test_no_regression_with_large_sample(self):
        """With very large sample, should be close to observed."""
        result = regress_to_mean(
            observed_value=0.350,
            sample_size=10000,
            league_mean=0.250,
            regression_constant=400,
        )
        assert abs(result - 0.350) < 0.01

    def test_half_regression_at_constant(self):
        """At regression constant sample size, should be halfway."""
        result = regress_to_mean(
            observed_value=0.350,
            sample_size=400,
            league_mean=0.250,
            regression_constant=400,
        )
        expected = (0.350 + 0.250) / 2  # 0.300
        assert abs(result - expected) < 0.001

    def test_regression_with_nan_returns_mean(self):
        """NaN observed value should return league mean."""
        result = regress_to_mean(
            observed_value=np.nan,
            sample_size=500,
            league_mean=0.250,
            regression_constant=400,
        )
        assert result == 0.250


class TestWeightedAverage:
    """Tests for weighted average calculations."""

    def test_single_year(self):
        """Single year should return that year's value."""
        stats = pd.DataFrame({
            "Season": [2024],
            "AVG": [0.300],
            "PA": [500],
        })
        avg, sample = calculate_weighted_average(stats, "AVG", "PA")
        assert abs(avg - 0.300) < 0.001
        assert sample == 500

    def test_multiple_years_weighted(self):
        """More recent years should have higher weight."""
        stats = pd.DataFrame({
            "Season": [2024, 2023, 2022],
            "AVG": [0.300, 0.280, 0.260],
            "PA": [500, 500, 500],
        })
        # With weights [5, 4, 3], weighted avg should be closer to 0.300
        avg, sample = calculate_weighted_average(stats, "AVG", "PA", year_weights=[5, 4, 3])
        assert avg > 0.280  # Should be above unweighted mean
        assert avg < 0.300  # But below most recent

    def test_missing_column_returns_nan(self):
        """Missing column should return NaN."""
        stats = pd.DataFrame({
            "Season": [2024],
            "AVG": [0.300],
            "PA": [500],
        })
        avg, sample = calculate_weighted_average(stats, "NONEXISTENT", "PA")
        assert np.isnan(avg)
        assert sample == 0


class TestAgingCurves:
    """Tests for aging curve adjustments."""

    def test_peak_age_no_adjustment(self):
        """At peak age, adjustment should be close to 1.0."""
        adj = get_aging_adjustment(27, 27, "batter")
        assert abs(adj - 1.0) < 0.01

    def test_batter_declines_after_peak(self):
        """Batters should decline after peak age."""
        adj = get_aging_adjustment(27, 35, "batter")
        assert adj < 1.0

    def test_pitcher_declines_faster(self):
        """Pitchers should decline faster than batters."""
        batter_adj = get_aging_adjustment(27, 35, "batter")
        pitcher_adj = get_aging_adjustment(27, 35, "pitcher")
        # For pitchers, higher means worse (ERA goes up)
        assert pitcher_adj > batter_adj


class TestParkFactors:
    """Tests for park factor adjustments."""

    def test_coors_is_hitter_friendly(self):
        """Coors Field should have high park factor."""
        pf = get_park_factor("COL")
        assert pf > 110

    def test_oracle_is_pitcher_friendly(self):
        """Oracle Park should have low park factor."""
        pf = get_park_factor("SFG")
        assert pf < 100

    def test_unknown_team_returns_neutral(self):
        """Unknown team should return 100 (neutral)."""
        pf = get_park_factor("UNKNOWN")
        assert pf == 100

    def test_park_change_adjustment(self):
        """Moving to Coors should increase batting stats."""
        original = 0.280
        adjusted = adjust_for_park_change(original, "SFG", "COL", "batting")
        assert adjusted > original


class TestRoleDetection:
    """Tests for role detection."""

    def test_everyday_batter_detection(self):
        """High PA player should be classified as Everyday."""
        stats = pd.DataFrame({
            "Season": [2024, 2023, 2022],
            "PA": [650, 620, 600],
            "G": [155, 150, 148],
        })
        role = detect_batter_role(stats)
        assert role in ["Everyday", "Everyday_Plus"]

    def test_bench_batter_detection(self):
        """Low PA player should be classified as Bench."""
        stats = pd.DataFrame({
            "Season": [2024, 2023],
            "PA": [150, 180],
            "G": [70, 80],
        })
        role = detect_batter_role(stats)
        assert role in ["Bench", "Part_Time"]

    def test_starter_detection(self):
        """High GS pitcher should be classified as starter."""
        stats = pd.DataFrame({
            "Season": [2024, 2023],
            "IP": [180, 175],
            "GS": [32, 31],
            "G": [33, 32],
            "SV": [0, 0],
        })
        role = detect_pitcher_role(stats)
        assert role.startswith("SP")

    def test_closer_detection(self):
        """High SV pitcher should be classified as closer."""
        stats = pd.DataFrame({
            "Season": [2024, 2023],
            "IP": [60, 62],
            "GS": [0, 0],
            "G": [60, 58],
            "SV": [35, 30],
        })
        role = detect_pitcher_role(stats)
        assert role == "CL"


class TestHealthMultiplier:
    """Tests for age-based health multipliers."""

    def test_prime_age_full_health(self):
        """Prime age should have 1.0 multiplier."""
        mult = get_health_multiplier(27, "batter")
        assert mult == 1.0

    def test_older_players_reduced(self):
        """Older players should have reduced multiplier."""
        mult = get_health_multiplier(38, "batter")
        assert mult < 0.75

    def test_pitchers_decline_faster(self):
        """Older pitchers should decline faster."""
        batter_mult = get_health_multiplier(35, "batter")
        pitcher_mult = get_health_multiplier(35, "pitcher")
        assert pitcher_mult < batter_mult


class TestPythagoreanWins:
    """Tests for Pythagorean win expectation."""

    def test_equal_runs_gives_81(self):
        """Equal runs scored and allowed should give 81 wins."""
        wins = pythagorean_wins(700, 700)
        assert abs(wins - 81) < 0.1

    def test_more_runs_scored_gives_more_wins(self):
        """More runs scored should give more wins."""
        wins = pythagorean_wins(800, 700)
        assert wins > 81

    def test_extreme_values(self):
        """Extreme run differential should give reasonable results."""
        # Very good team
        good_wins = pythagorean_wins(900, 600)
        assert 100 < good_wins < 130

        # Very bad team
        bad_wins = pythagorean_wins(550, 850)
        assert 45 < bad_wins < 70


class TestBatterProjection:
    """Tests for batter projection generation."""

    @pytest.fixture
    def sample_batter_stats(self):
        """Sample historical batting stats."""
        return pd.DataFrame({
            "Season": [2024, 2023, 2022],
            "Name": ["Test Player", "Test Player", "Test Player"],
            "Team": ["NYY", "NYY", "NYY"],
            "Age": [28, 27, 26],
            "PA": [600, 580, 550],
            "AVG": [0.285, 0.275, 0.265],
            "OBP": [0.350, 0.340, 0.330],
            "SLG": [0.480, 0.460, 0.440],
            "wOBA": [0.360, 0.350, 0.340],
            "wRC+": [130, 125, 118],
            "ISO": [0.195, 0.185, 0.175],
            "BB%": [0.10, 0.095, 0.09],
            "K%": [0.20, 0.21, 0.22],
        })

    def test_projection_returns_batter_projection(self, sample_batter_stats):
        """Should return BatterProjection instance."""
        proj = project_batter(
            player_id=12345,
            historical_stats=sample_batter_stats,
            projection_year=2025,
        )
        assert isinstance(proj, BatterProjection)

    def test_projection_reasonable_values(self, sample_batter_stats):
        """Projected values should be reasonable."""
        proj = project_batter(
            player_id=12345,
            historical_stats=sample_batter_stats,
            projection_year=2025,
        )
        # Should be between historical range and league average
        assert 0.200 < proj.projected_avg < 0.350
        assert 0.280 < proj.projected_obp < 0.420
        assert 0.350 < proj.projected_slg < 0.550
        assert 80 < proj.projected_wrc_plus < 160


class TestPitcherProjection:
    """Tests for pitcher projection generation."""

    @pytest.fixture
    def sample_pitcher_stats(self):
        """Sample historical pitching stats."""
        return pd.DataFrame({
            "Season": [2024, 2023, 2022],
            "Name": ["Test Pitcher", "Test Pitcher", "Test Pitcher"],
            "Team": ["LAD", "LAD", "LAD"],
            "Age": [29, 28, 27],
            "IP": [180, 175, 165],
            "GS": [32, 31, 30],
            "G": [33, 32, 31],
            "SV": [0, 0, 0],
            "ERA": [3.20, 3.40, 3.60],
            "FIP": [3.30, 3.50, 3.70],
            "xFIP": [3.35, 3.55, 3.75],
            "WHIP": [1.10, 1.15, 1.20],
            "K/9": [10.5, 10.0, 9.5],
            "BB/9": [2.5, 2.7, 2.9],
            "HR/9": [1.0, 1.1, 1.2],
        })

    def test_projection_returns_pitcher_projection(self, sample_pitcher_stats):
        """Should return PitcherProjection instance."""
        proj = project_pitcher(
            player_id=54321,
            historical_stats=sample_pitcher_stats,
            projection_year=2025,
        )
        assert isinstance(proj, PitcherProjection)

    def test_starter_role_detected(self, sample_pitcher_stats):
        """Should detect starter role."""
        proj = project_pitcher(
            player_id=54321,
            historical_stats=sample_pitcher_stats,
            projection_year=2025,
        )
        assert proj.role.startswith("SP")

    def test_projection_reasonable_values(self, sample_pitcher_stats):
        """Projected values should be reasonable."""
        proj = project_pitcher(
            player_id=54321,
            historical_stats=sample_pitcher_stats,
            projection_year=2025,
        )
        assert 2.50 < proj.projected_era < 5.00
        assert 2.50 < proj.projected_fip < 5.00
        assert 0.90 < proj.projected_whip < 1.50
        assert 6.0 < proj.projected_k_per_9 < 13.0
