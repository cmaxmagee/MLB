"""Tests for player variance features in Monte Carlo simulation.

Tests injury multiplier variance (playing time sampling) and
young player upside flag (wider distributions for breakout candidates).
"""

import numpy as np
import pytest

from src.simulation.monte_carlo import (
    get_playing_time_variance,
    is_young_player_with_upside,
    sample_playing_time,
    PLAYING_TIME_CV_BY_AGE,
    YOUNG_PLAYER_AGE_THRESHOLD,
    YOUNG_PLAYER_EXPERIENCE_THRESHOLD,
    YOUNG_PLAYER_VARIANCE_MULTIPLIER,
)


class TestPlayingTimeVariance:
    """Tests for age-based playing time variance (injury modeling)."""

    def test_young_players_have_lowest_variance(self):
        """Players under 26 should have the lowest injury variance."""
        young_cv = get_playing_time_variance(23)
        prime_cv = get_playing_time_variance(28)
        veteran_cv = get_playing_time_variance(32)
        old_cv = get_playing_time_variance(37)

        assert young_cv < prime_cv < veteran_cv < old_cv

    def test_variance_by_age_bracket(self):
        """Verify correct variance for each age bracket."""
        assert get_playing_time_variance(22) == PLAYING_TIME_CV_BY_AGE["young"]
        assert get_playing_time_variance(25) == PLAYING_TIME_CV_BY_AGE["young"]
        assert get_playing_time_variance(26) == PLAYING_TIME_CV_BY_AGE["prime"]
        assert get_playing_time_variance(30) == PLAYING_TIME_CV_BY_AGE["prime"]
        assert get_playing_time_variance(31) == PLAYING_TIME_CV_BY_AGE["veteran"]
        assert get_playing_time_variance(34) == PLAYING_TIME_CV_BY_AGE["veteran"]
        assert get_playing_time_variance(35) == PLAYING_TIME_CV_BY_AGE["old"]
        assert get_playing_time_variance(40) == PLAYING_TIME_CV_BY_AGE["old"]

    def test_sample_playing_time_returns_positive(self):
        """Sampled playing time should never be negative."""
        np.random.seed(42)
        for _ in range(100):
            sampled = sample_playing_time(500, 35, is_pitcher=False)
            assert sampled >= 0

    def test_sample_playing_time_batter_bounded(self):
        """Batter PA should be bounded at 720."""
        np.random.seed(42)
        for _ in range(100):
            sampled = sample_playing_time(650, 25, is_pitcher=False)
            assert sampled <= 720

    def test_sample_playing_time_pitcher_bounded(self):
        """Pitcher IP should be bounded at 1.3x projected."""
        np.random.seed(42)
        projected_ip = 180
        for _ in range(100):
            sampled = sample_playing_time(projected_ip, 28, is_pitcher=True)
            assert sampled <= projected_ip * 1.3

    def test_sample_playing_time_older_players_more_variance(self):
        """Older players should have more variance in sampled playing time."""
        np.random.seed(42)
        n_samples = 1000
        projected_pa = 550

        young_samples = [sample_playing_time(projected_pa, 24, False) for _ in range(n_samples)]
        old_samples = [sample_playing_time(projected_pa, 36, False) for _ in range(n_samples)]

        young_std = np.std(young_samples)
        old_std = np.std(old_samples)

        # Old players should have significantly more variance
        assert old_std > young_std * 1.5

    def test_sample_playing_time_mean_near_projection(self):
        """Mean of samples should be close to projected value."""
        np.random.seed(42)
        n_samples = 5000
        projected_pa = 500

        samples = [sample_playing_time(projected_pa, 28, False) for _ in range(n_samples)]
        mean = np.mean(samples)

        # Mean should be within 5% of projection
        assert abs(mean - projected_pa) / projected_pa < 0.05


class TestYoungPlayerUpside:
    """Tests for young player upside flag."""

    def test_young_inexperienced_qualifies(self):
        """Young player with limited experience should qualify."""
        assert is_young_player_with_upside(age=23, seasons_of_data=1)
        assert is_young_player_with_upside(age=24, seasons_of_data=0)
        assert is_young_player_with_upside(age=25, seasons_of_data=1)

    def test_young_experienced_does_not_qualify(self):
        """Young player with more experience should not qualify."""
        assert not is_young_player_with_upside(age=23, seasons_of_data=2)
        assert not is_young_player_with_upside(age=24, seasons_of_data=3)

    def test_old_inexperienced_does_not_qualify(self):
        """Older player even with limited experience should not qualify."""
        assert not is_young_player_with_upside(age=26, seasons_of_data=1)
        assert not is_young_player_with_upside(age=28, seasons_of_data=0)
        assert not is_young_player_with_upside(age=30, seasons_of_data=1)

    def test_boundary_age(self):
        """Test the age boundary exactly."""
        # 25 is young, 26 is not
        assert is_young_player_with_upside(age=25, seasons_of_data=1)
        assert not is_young_player_with_upside(age=26, seasons_of_data=1)

    def test_boundary_experience(self):
        """Test the experience boundary exactly."""
        # 1 season qualifies, 2 does not
        assert is_young_player_with_upside(age=23, seasons_of_data=1)
        assert not is_young_player_with_upside(age=23, seasons_of_data=2)

    def test_variance_multiplier_constant(self):
        """Verify the variance multiplier is as expected."""
        assert YOUNG_PLAYER_VARIANCE_MULTIPLIER == 1.5

    def test_threshold_constants(self):
        """Verify threshold constants are as expected."""
        assert YOUNG_PLAYER_AGE_THRESHOLD == 26
        assert YOUNG_PLAYER_EXPERIENCE_THRESHOLD == 2


class TestVarianceIntegration:
    """Integration tests for variance features in simulation."""

    def test_variance_features_dont_crash_with_mock_data(self):
        """Basic smoke test that variance features work."""
        np.random.seed(42)

        # Simulate what happens in the simulation loop
        ages = [23, 28, 35]
        seasons = [1, 5, 10]

        for age, seas in zip(ages, seasons):
            # Sample playing time
            sampled_pa = sample_playing_time(550, age, False)
            assert sampled_pa >= 0

            # Check upside flag
            has_upside = is_young_player_with_upside(age, seas)
            variance_mult = YOUNG_PLAYER_VARIANCE_MULTIPLIER if has_upside else 1.0
            assert variance_mult >= 1.0

    def test_young_prospect_has_wider_outcomes(self):
        """Young prospects should have wider stat distributions than veterans."""
        np.random.seed(42)

        base_std = 15.0  # BATTER_WRC_PLUS_STDEV
        confidence = 0.8

        # Prospect: 23 years old, 1 season
        prospect_std = base_std * (2.0 - confidence)
        if is_young_player_with_upside(23, 1):
            prospect_std *= YOUNG_PLAYER_VARIANCE_MULTIPLIER

        # Veteran: 32 years old, 8 seasons
        veteran_std = base_std * (2.0 - confidence)
        if is_young_player_with_upside(32, 8):
            veteran_std *= YOUNG_PLAYER_VARIANCE_MULTIPLIER

        # Prospect should have wider distribution
        assert prospect_std > veteran_std
        assert prospect_std == veteran_std * YOUNG_PLAYER_VARIANCE_MULTIPLIER

    def test_injury_variance_compounds_with_age(self):
        """Test that old players can have very different PA in sims."""
        np.random.seed(123)

        projected_pa = 500
        n_sims = 1000

        # Old injury-prone player
        old_samples = [sample_playing_time(projected_pa, 37, False) for _ in range(n_sims)]

        # Should see significant variance - some sims with low PA (injury)
        min_pa = min(old_samples)
        max_pa = max(old_samples)

        # Range should be at least 200 PA
        assert max_pa - min_pa > 200

        # Should have some "injury" scenarios (< 400 PA)
        low_pa_count = sum(1 for pa in old_samples if pa < 400)
        assert low_pa_count > 0, "Should have some injury scenarios"
