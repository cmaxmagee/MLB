"""Integration tests for the MLB Season Prediction Model.

These tests verify that all components work together correctly.
"""

import numpy as np
import pandas as pd
import pytest

from src.projections.player import (
    BatterProjection,
    PitcherProjection,
    project_batter,
    project_pitcher,
    calculate_counting_stats,
)
from src.projections.playing_time import (
    project_playing_time,
    allocate_team_playing_time,
)
from src.projections.team import (
    pythagorean_wins,
    project_team,
    DIVISIONS,
    TEAM_TO_DIVISION,
)
from src.projections.projector import (
    ProjectionConfig,
    TeamProjectionSet,
)
from src.simulation.monte_carlo import (
    run_simulation,
    run_simulation_with_player_variance,
    SeasonSimulation,
    SimulationResult,
    calculate_matchup_probability,
    probability_of_n_wins,
)
from src.output.reports import (
    generate_season_report,
    generate_team_report,
    export_to_csv,
)


class TestTeamProjectionIntegration:
    """Integration tests for team projection workflow."""

    @pytest.fixture
    def sample_batter_projections(self):
        """Create sample batter projections for a team."""
        batters = []
        for i in range(9):
            proj = BatterProjection(
                player_id=1000 + i,
                name=f"Batter {i+1}",
                team="NYY",
                age=27 + (i % 5),
                projection_year=2025,
                projected_pa=550 - i * 30,
                projected_games=145 - i * 5,
                projected_avg=0.265 + i * 0.005,
                projected_obp=0.335 + i * 0.005,
                projected_slg=0.440 + i * 0.010,
                projected_wrc_plus=110 + i * 3,
                projected_batting_runs=(110 + i * 3 - 100) / 100 * (550 - i * 30) * 0.12,
                projected_baserunning_runs=1.0 - i * 0.2,
                projection_confidence=0.7 + i * 0.02,
            )
            batters.append(proj)
        return batters

    @pytest.fixture
    def sample_pitcher_projections(self):
        """Create sample pitcher projections for a team."""
        pitchers = []
        # 5 starters
        for i in range(5):
            proj = PitcherProjection(
                player_id=2000 + i,
                name=f"Starter {i+1}",
                team="NYY",
                age=28 + i,
                projection_year=2025,
                role=f"SP{i+1}",
                projected_ip=180 - i * 15,
                projected_games=32 - i,
                projected_era=3.50 + i * 0.15,
                projected_fip=3.40 + i * 0.15,
                projected_whip=1.15 + i * 0.03,
                projected_k_per_9=9.5 - i * 0.3,
                projected_pitching_runs=-((3.40 + i * 0.15 - 4.00) / 9 * (180 - i * 15)),
                projection_confidence=0.75 - i * 0.03,
            )
            pitchers.append(proj)

        # 5 relievers
        for i in range(5):
            role = "CL" if i == 0 else f"RP{i}"
            proj = PitcherProjection(
                player_id=2100 + i,
                name=f"Reliever {i+1}",
                team="NYY",
                age=29 + i,
                projection_year=2025,
                role=role,
                projected_ip=65 - i * 5,
                projected_games=65 - i * 5,
                projected_era=3.20 + i * 0.25,
                projected_fip=3.30 + i * 0.20,
                projected_whip=1.10 + i * 0.04,
                projected_k_per_9=10.0 - i * 0.4,
                projected_pitching_runs=-((3.30 + i * 0.20 - 4.00) / 9 * (65 - i * 5)),
                projection_confidence=0.65 - i * 0.02,
            )
            pitchers.append(proj)

        return pitchers

    def test_team_projection_workflow(
        self, sample_batter_projections, sample_pitcher_projections
    ):
        """Test complete team projection from player projections."""
        team_proj = project_team(
            "NYY",
            sample_batter_projections,
            sample_pitcher_projections,
        )

        assert team_proj.team == "NYY"
        assert team_proj.division == "AL East"

        # Check reasonable values
        assert 550 < team_proj.projected_runs_scored < 900
        assert 550 < team_proj.projected_runs_allowed < 850
        assert 65 < team_proj.projected_wins < 105
        assert 90 < team_proj.lineup_wrc_plus < 135
        assert 3.0 < team_proj.rotation_fip < 4.5
        assert 3.0 < team_proj.bullpen_fip < 4.5

    def test_team_projection_set_creation(
        self, sample_batter_projections, sample_pitcher_projections
    ):
        """Test TeamProjectionSet creation and aggregation."""
        proj_set = TeamProjectionSet(
            team="NYY",
            division="AL East",
            batters=sample_batter_projections,
            pitchers=sample_pitcher_projections,
        )

        # Calculate aggregates
        proj_set.total_batting_runs = sum(b.projected_batting_runs for b in sample_batter_projections)
        proj_set.total_pitching_runs = sum(p.projected_pitching_runs for p in sample_pitcher_projections)
        proj_set.projected_runs_scored = 700 + proj_set.total_batting_runs
        proj_set.projected_runs_allowed = 700 - proj_set.total_pitching_runs
        proj_set.projected_wins = pythagorean_wins(
            proj_set.projected_runs_scored,
            proj_set.projected_runs_allowed,
        )

        assert len(proj_set.batters) == 9
        assert len(proj_set.pitchers) == 10
        assert proj_set.projected_wins > 0


class TestMonteCarloSimulation:
    """Integration tests for Monte Carlo simulation."""

    @pytest.fixture
    def team_projection_sets(self):
        """Create minimal team projection sets for all 30 teams."""
        projections = {}

        for div, teams in DIVISIONS.items():
            for i, team in enumerate(teams):
                # Create simple projections with varying strength
                base_wrc = 100 + (i - 2) * 5  # Vary by position in division
                base_fip = 4.00 - (i - 2) * 0.10

                batters = []
                for j in range(5):  # Just 5 batters for speed
                    proj = BatterProjection(
                        player_id=hash(f"{team}-b{j}") % 100000,
                        name=f"{team} Batter {j+1}",
                        team=team,
                        age=27,
                        projection_year=2025,
                        projected_pa=600,
                        projected_games=150,
                        projected_avg=0.260,
                        projected_obp=0.330,
                        projected_slg=0.420,
                        projected_wrc_plus=base_wrc,
                        projected_batting_runs=(base_wrc - 100) / 100 * 600 * 0.12,
                        projection_confidence=0.7,
                    )
                    batters.append(proj)

                pitchers = []
                for j in range(5):  # Just 5 pitchers for speed
                    proj = PitcherProjection(
                        player_id=hash(f"{team}-p{j}") % 100000,
                        name=f"{team} Pitcher {j+1}",
                        team=team,
                        age=28,
                        projection_year=2025,
                        role=f"SP{j+1}" if j < 3 else f"RP{j-2}",
                        projected_ip=150 if j < 3 else 60,
                        projected_games=30 if j < 3 else 55,
                        projected_era=base_fip + 0.20,
                        projected_fip=base_fip,
                        projected_whip=1.20,
                        projected_k_per_9=9.0,
                        projected_pitching_runs=-((base_fip - 4.00) / 9 * (150 if j < 3 else 60)),
                        projection_confidence=0.7,
                    )
                    pitchers.append(proj)

                # Calculate totals
                total_batting = sum(b.projected_batting_runs for b in batters)
                total_pitching = sum(p.projected_pitching_runs for p in pitchers)
                rs = 700 + total_batting
                ra = 700 - total_pitching

                projections[team] = TeamProjectionSet(
                    team=team,
                    division=div,
                    batters=batters,
                    pitchers=pitchers,
                    total_batting_runs=total_batting,
                    total_pitching_runs=total_pitching,
                    projected_runs_scored=rs,
                    projected_runs_allowed=ra,
                    projected_wins=pythagorean_wins(rs, ra),
                )

        return projections

    def test_basic_simulation(self, team_projection_sets):
        """Test basic Monte Carlo simulation."""
        simulation = run_simulation(
            team_projection_sets,
            iterations=100,  # Small for speed
            random_seed=42,
        )

        assert isinstance(simulation, SeasonSimulation)
        assert simulation.iterations == 100
        assert len(simulation.team_results) == 30

        # Check each team has results
        for team in DIVISIONS["AL East"]:
            assert team in simulation.team_results
            result = simulation.team_results[team]
            assert isinstance(result, SimulationResult)
            assert 50 < result.mean_wins < 110
            assert result.playoff_pct >= 0
            assert result.playoff_pct <= 100

    def test_simulation_with_player_variance(self, team_projection_sets):
        """Test simulation with player-level variance."""
        simulation = run_simulation_with_player_variance(
            team_projection_sets,
            iterations=50,  # Small for speed
            random_seed=42,
            show_progress=False,
        )

        assert isinstance(simulation, SeasonSimulation)
        assert len(simulation.team_results) == 30

        # Player variance should produce slightly different results
        # but still reasonable
        for team, result in simulation.team_results.items():
            assert 45 < result.mean_wins < 115
            assert result.std_wins > 0

    def test_playoff_odds_sum(self, team_projection_sets):
        """Test that playoff odds are consistent."""
        simulation = run_simulation(
            team_projection_sets,
            iterations=500,
            random_seed=42,
        )

        # Division winners should sum to ~100% per division
        for div, teams in DIVISIONS.items():
            div_winner_sum = sum(
                simulation.team_results[t].division_winner_pct
                for t in teams
            )
            # Allow for rounding error
            assert 99 < div_winner_sum < 101

    def test_matchup_probability(self, team_projection_sets):
        """Test head-to-head matchup probability calculation."""
        simulation = run_simulation(
            team_projection_sets,
            iterations=500,
            random_seed=42,
        )

        matchup = calculate_matchup_probability("NYY", "BOS", simulation)

        assert "NYY" in matchup["team1"]
        assert "BOS" in matchup["team2"]
        assert matchup["NYY_wins_more"] + matchup["BOS_wins_more"] + matchup["tie_pct"] == pytest.approx(100, abs=0.5)

    def test_probability_of_n_wins(self, team_projection_sets):
        """Test win probability calculation."""
        simulation = run_simulation(
            team_projection_sets,
            iterations=500,
            random_seed=42,
        )

        # Probability of 0+ wins should be 100%
        prob_0 = probability_of_n_wins("NYY", 0, simulation, at_least=True)
        assert prob_0 == 100.0

        # Probability of 200+ wins should be 0%
        prob_200 = probability_of_n_wins("NYY", 200, simulation, at_least=True)
        assert prob_200 == 0.0

        # Probability of 81+ should be between 0 and 100
        prob_81 = probability_of_n_wins("NYY", 81, simulation, at_least=True)
        assert 0 <= prob_81 <= 100


class TestReportGeneration:
    """Integration tests for report generation."""

    @pytest.fixture
    def simulation_results(self):
        """Create simulation results for report testing."""
        # Create mock projection sets
        projections = {}
        for team in ["NYY", "BOS", "TOR", "TBR", "BAL"]:
            batters = [
                BatterProjection(
                    player_id=i,
                    name=f"{team} Batter {i}",
                    team=team,
                    age=27,
                    projection_year=2025,
                    projected_pa=500,
                    projected_games=140,
                    projected_avg=0.270,
                    projected_obp=0.340,
                    projected_slg=0.450,
                    projected_wrc_plus=110,
                    projected_batting_runs=10.0,
                    projection_confidence=0.7,
                )
                for i in range(5)
            ]

            pitchers = [
                PitcherProjection(
                    player_id=100 + i,
                    name=f"{team} Pitcher {i}",
                    team=team,
                    age=28,
                    projection_year=2025,
                    role="SP1" if i < 3 else "RP",
                    projected_ip=150 if i < 3 else 60,
                    projected_games=30 if i < 3 else 55,
                    projected_era=3.50,
                    projected_fip=3.40,
                    projected_whip=1.15,
                    projected_k_per_9=9.5,
                    projected_pitching_runs=8.0,
                    projection_confidence=0.7,
                )
                for i in range(5)
            ]

            projections[team] = TeamProjectionSet(
                team=team,
                division="AL East",
                batters=batters,
                pitchers=pitchers,
                total_batting_runs=50.0,
                total_pitching_runs=40.0,
                projected_runs_scored=750,
                projected_runs_allowed=660,
                projected_wins=90,
            )

        return run_simulation(projections, iterations=100, random_seed=42)

    def test_season_report_generation(self, simulation_results):
        """Test season report generation."""
        report = generate_season_report(
            simulation_results,
            projection_year=2025,
        )

        assert isinstance(report, str)
        assert "2025" in report
        assert "AL" in report or "STANDINGS" in report
        assert len(report) > 100

    def test_team_report_generation(self, simulation_results):
        """Test individual team report generation."""
        report = generate_team_report(
            "NYY",
            simulation_results,
        )

        assert isinstance(report, str)
        assert "NYY" in report or "Yankees" in report
        assert "WIN" in report.upper() or "PLAYOFF" in report.upper()

    def test_csv_export(self, simulation_results, tmp_path):
        """Test CSV export functionality."""
        csv_path = tmp_path / "test_standings.csv"
        export_to_csv(simulation_results, csv_path)

        assert csv_path.exists()

        # Read and verify
        df = pd.read_csv(csv_path)
        assert "Team" in df.columns
        assert "Projected Wins" in df.columns
        assert len(df) == 5  # We only created 5 teams


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    def test_projection_to_simulation_to_report(self):
        """Test complete workflow from projections to reports."""
        # Step 1: Create player projections
        batters = [
            BatterProjection(
                player_id=i,
                name=f"Player {i}",
                team="NYY",
                age=27,
                projection_year=2025,
                projected_pa=600,
                projected_games=150,
                projected_avg=0.270,
                projected_obp=0.340,
                projected_slg=0.450,
                projected_wrc_plus=115,
                projected_batting_runs=15.0,
                projection_confidence=0.8,
            )
            for i in range(9)
        ]

        pitchers = [
            PitcherProjection(
                player_id=100 + i,
                name=f"Pitcher {i}",
                team="NYY",
                age=28,
                projection_year=2025,
                role="SP1" if i < 5 else "RP",
                projected_ip=180 if i < 5 else 70,
                projected_games=32 if i < 5 else 60,
                projected_era=3.40,
                projected_fip=3.30,
                projected_whip=1.10,
                projected_k_per_9=10.0,
                projected_pitching_runs=12.0,
                projection_confidence=0.75,
            )
            for i in range(10)
        ]

        # Step 2: Create team projection
        team_proj = project_team("NYY", batters, pitchers)

        assert team_proj.projected_wins > 81  # Should be winning team

        # Step 3: Create projection set for simulation
        proj_set = TeamProjectionSet(
            team="NYY",
            division="AL East",
            batters=batters,
            pitchers=pitchers,
            total_batting_runs=sum(b.projected_batting_runs for b in batters),
            total_pitching_runs=sum(p.projected_pitching_runs for p in pitchers),
            projected_runs_scored=team_proj.projected_runs_scored,
            projected_runs_allowed=team_proj.projected_runs_allowed,
            projected_wins=team_proj.projected_wins,
        )

        # Create minimal opposing teams
        projections = {"NYY": proj_set}
        for team in ["BOS", "TOR", "TBR", "BAL"]:
            projections[team] = TeamProjectionSet(
                team=team,
                division="AL East",
                batters=[],
                pitchers=[],
                projected_runs_scored=700,
                projected_runs_allowed=700,
                projected_wins=81,
            )

        # Step 4: Run simulation
        simulation = run_simulation(projections, iterations=100, random_seed=42)

        assert "NYY" in simulation.team_results
        assert simulation.team_results["NYY"].mean_wins > 80

        # Step 5: Generate report
        report = generate_season_report(simulation, projection_year=2025)

        assert "NYY" in report
        assert len(report) > 0

        # Verify NYY is top team (should be with our constructed advantage)
        df = simulation.get_playoff_odds_df()
        top_team = df.iloc[0]["Team"]
        assert top_team == "NYY"
