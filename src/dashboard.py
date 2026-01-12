"""Streamlit dashboard for MLB Season Projections.

Run with: streamlit run src/dashboard.py

Supports two modes:
1. Read-only mode: Loads pre-computed results from cache (instant)
2. Live mode: Runs projections on-demand (requires API access)
"""

import logging
import os
import sys
from pathlib import Path

from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.projections.projector import (
    Projector,
    ProjectionConfig,
    TeamProjectionSet,
    projections_to_dataframe,
)
from src.projections.team import DIVISIONS, TEAM_TO_DIVISION
from src.simulation.monte_carlo import (
    run_simulation,
    run_simulation_with_player_variance,
    SeasonSimulation,
)
from src.data.cache_results import load_simulation_results, get_available_cache_files

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variable to control mode
# Set DASHBOARD_MODE=live to enable running new projections
DASHBOARD_MODE = os.environ.get("DASHBOARD_MODE", "auto")  # "auto", "readonly", "live"

# Page config
st.set_page_config(
    page_title="MLB Season Projections",
    page_icon="⚾",
    layout="wide",
)


def load_cached_results(year: int = 2026) -> bool:
    """Try to load cached simulation results.

    Returns True if cache was loaded successfully.
    """
    result = load_simulation_results(year=year)

    if result is None:
        return False

    projections, simulation, batter_df, pitcher_df, metadata = result

    st.session_state.projections = projections
    st.session_state.simulation = simulation
    st.session_state.batter_df = batter_df
    st.session_state.pitcher_df = pitcher_df
    st.session_state.metadata = metadata
    st.session_state.from_cache = True

    return True


def main():
    """Main dashboard application."""
    st.title("⚾ MLB Season Projections")
    st.markdown("Monte Carlo simulation system for MLB season outcomes")

    # Initialize session state
    if "projections" not in st.session_state:
        st.session_state.projections = None
    if "simulation" not in st.session_state:
        st.session_state.simulation = None
    if "batter_df" not in st.session_state:
        st.session_state.batter_df = None
    if "pitcher_df" not in st.session_state:
        st.session_state.pitcher_df = None
    if "metadata" not in st.session_state:
        st.session_state.metadata = None
    if "from_cache" not in st.session_state:
        st.session_state.from_cache = False
    if "cache_checked" not in st.session_state:
        st.session_state.cache_checked = False

    # Check for cached results on first load
    if not st.session_state.cache_checked:
        st.session_state.cache_checked = True
        # Try to load 2026, then 2025
        for year in [2026, 2025]:
            if load_cached_results(year):
                break

    # Determine if we're in read-only mode
    is_readonly = DASHBOARD_MODE == "readonly" or (
        DASHBOARD_MODE == "auto" and st.session_state.from_cache
    )

    # Sidebar
    with st.sidebar:
        if st.session_state.from_cache and st.session_state.metadata:
            # Show cache info
            meta = st.session_state.metadata
            st.success("📊 Viewing cached projections")
            st.caption(f"Generated: {meta['timestamp'][:10]}")
            st.caption(f"Year: {meta['year']}")
            st.caption(f"Iterations: {meta['iterations']:,}")

            # Show available cache years
            cache_files = get_available_cache_files()
            available_years = sorted(set(f["year"] for f in cache_files if f["is_latest"]), reverse=True)

            if len(available_years) > 1:
                st.divider()
                selected_year = st.selectbox(
                    "Switch Year",
                    options=available_years,
                    index=available_years.index(meta["year"]) if meta["year"] in available_years else 0,
                )
                if selected_year != meta["year"]:
                    if load_cached_results(selected_year):
                        st.rerun()

        # Show live controls only if not readonly
        if DASHBOARD_MODE == "live" or (DASHBOARD_MODE == "auto" and not st.session_state.from_cache):
            st.header("Configuration")

            projection_year = st.selectbox(
                "Projection Year",
                options=[2025, 2026, 2027],
                index=1,
            )

            iterations = st.select_slider(
                "Simulation Iterations",
                options=[1000, 5000, 10000, 25000, 50000],
                value=10000,
            )

            player_variance = st.checkbox(
                "Player Variance",
                value=True,
                help="Use player-level variance with injury modeling and young player upside",
            )

            auto_sync = st.checkbox(
                "Auto-Sync Rosters",
                value=False,
                help="Automatically fetch current rosters from MLB API",
            )

            random_seed = st.number_input(
                "Random Seed (optional)",
                min_value=0,
                max_value=99999,
                value=0,
                help="Set to 0 for random, or enter a seed for reproducibility",
            )

            st.divider()

            run_button = st.button(
                "🚀 Run Projections",
                type="primary",
                use_container_width=True,
            )

            if run_button:
                run_projections(
                    projection_year=projection_year,
                    iterations=iterations,
                    player_variance=player_variance,
                    auto_sync=auto_sync,
                    random_seed=random_seed if random_seed > 0 else None,
                )

        # Footer with mode info
        st.divider()
        if is_readonly:
            st.caption("📖 Read-only mode")
        else:
            st.caption("⚡ Live mode")

    # Main content
    if st.session_state.simulation is None:
        st.info("No projection data available.")
        if DASHBOARD_MODE != "readonly":
            st.markdown("Configure settings in the sidebar and click 'Run Projections' to begin.")
        else:
            st.markdown("Cached projections not found. Please generate cache using:")
            st.code("python -m src.main generate-cache --year 2026")
        show_demo_placeholder()
    else:
        show_results()


def run_projections(
    projection_year: int,
    iterations: int,
    player_variance: bool,
    auto_sync: bool,
    random_seed: Optional[int],
):
    """Run the projection and simulation pipeline."""
    progress_bar = st.progress(0, text="Initializing...")

    try:
        # Step 1: Generate projections
        progress_bar.progress(10, text="Generating player projections...")

        config = ProjectionConfig(
            projection_year=projection_year,
            auto_sync_rosters=auto_sync,
        )
        projector = Projector(config)

        with st.spinner("Fetching player data and generating projections..."):
            projections = projector.project_all_teams()

        st.session_state.projections = projections

        # Convert to DataFrames
        batter_df, pitcher_df = projections_to_dataframe(projections)
        st.session_state.batter_df = batter_df
        st.session_state.pitcher_df = pitcher_df

        # Step 2: Run simulation
        progress_bar.progress(50, text=f"Running {iterations:,} simulations...")

        with st.spinner(f"Simulating {iterations:,} seasons..."):
            if player_variance:
                simulation = run_simulation_with_player_variance(
                    team_projection_sets=projections,
                    iterations=iterations,
                    random_seed=random_seed,
                    show_progress=False,
                )
            else:
                simulation = run_simulation(
                    team_projections=projections,
                    iterations=iterations,
                    random_seed=random_seed,
                    show_progress=False,
                )

        st.session_state.simulation = simulation

        progress_bar.progress(100, text="Complete!")
        st.success(f"Generated projections for {len(projections)} teams with {iterations:,} simulations!")

    except Exception as e:
        st.error(f"Error running projections: {e}")
        logger.exception("Projection error")


def show_demo_placeholder():
    """Show placeholder content before projections are run."""
    st.markdown("### What this tool does:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **📊 Team Projections**
        - Win-loss records
        - Runs scored/allowed
        - Pythagorean expectation
        - Strength of schedule
        """)

    with col2:
        st.markdown("""
        **🎯 Playoff Odds**
        - Division winner %
        - Wild card %
        - Overall playoff %
        - Confidence intervals
        """)

    with col3:
        st.markdown("""
        **👤 Player Projections**
        - Batting: AVG, OBP, SLG, wRC+
        - Pitching: ERA, FIP, WHIP, K/9
        - Playing time (PA/IP)
        - Counting stats
        """)


def show_results():
    """Display projection and simulation results."""
    simulation: SeasonSimulation = st.session_state.simulation
    projections: Dict[str, TeamProjectionSet] = st.session_state.projections

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Standings & Playoff Odds",
        "🏟️ Team Details",
        "👥 Player Projections",
        "📊 Analytics",
    ])

    with tab1:
        show_standings_tab(simulation, projections)

    with tab2:
        show_team_details_tab(simulation, projections)

    with tab3:
        show_player_projections_tab()

    with tab4:
        show_analytics_tab(simulation, projections)


def show_standings_tab(simulation: SeasonSimulation, projections: Dict[str, TeamProjectionSet]):
    """Display standings and playoff odds."""
    st.header("Projected Standings & Playoff Odds")

    # Get playoff odds DataFrame
    odds_df = simulation.get_playoff_odds_df()

    # Add division info
    odds_df["Division"] = odds_df["Team"].map(TEAM_TO_DIVISION)

    # Division standings
    st.subheader("Division Standings")

    # Create columns for AL and NL
    col_al, col_nl = st.columns(2)

    with col_al:
        st.markdown("### American League")
        for div in ["AL East", "AL Central", "AL West"]:
            div_teams = odds_df[odds_df["Division"] == div].sort_values("Proj W", ascending=False)
            st.markdown(f"**{div}**")
            st.dataframe(
                div_teams[["Team", "Proj W", "10th", "90th", "Div %", "Playoff %"]],
                hide_index=True,
                use_container_width=True,
            )

    with col_nl:
        st.markdown("### National League")
        for div in ["NL East", "NL Central", "NL West"]:
            div_teams = odds_df[odds_df["Division"] == div].sort_values("Proj W", ascending=False)
            st.markdown(f"**{div}**")
            st.dataframe(
                div_teams[["Team", "Proj W", "10th", "90th", "Div %", "Playoff %"]],
                hide_index=True,
                use_container_width=True,
            )

    # League-wide playoff odds
    st.subheader("Playoff Odds Leaderboard")

    # Top 12 playoff contenders
    top_contenders = odds_df.nlargest(12, "Playoff %")

    fig = px.bar(
        top_contenders,
        x="Team",
        y="Playoff %",
        color="Division",
        title="Top 12 Playoff Contenders",
        text="Playoff %",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(yaxis_title="Playoff Probability (%)", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)


def show_team_details_tab(simulation: SeasonSimulation, projections: Dict[str, TeamProjectionSet]):
    """Display detailed view for a selected team."""
    st.header("Team Details")

    # Team selector
    teams = sorted(projections.keys())
    selected_team = st.selectbox("Select Team", teams)

    if selected_team:
        team_proj = projections[selected_team]
        team_result = simulation.team_results[selected_team]

        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Projected Wins", f"{team_result.mean_wins:.1f}")
        with col2:
            st.metric("Playoff Odds", f"{team_result.playoff_pct:.1f}%")
        with col3:
            st.metric("Division Winner", f"{team_result.division_winner_pct:.1f}%")
        with col4:
            st.metric("Wild Card", f"{team_result.wild_card_pct:.1f}%")

        # Win distribution histogram
        st.subheader("Win Distribution")

        if len(team_result.win_distribution) > 0:
            fig = px.histogram(
                x=team_result.win_distribution,
                nbins=40,
                title=f"{selected_team} Win Distribution ({simulation.iterations:,} simulations)",
                labels={"x": "Wins", "y": "Frequency"},
            )

            # Add vertical lines for percentiles
            fig.add_vline(x=team_result.mean_wins, line_dash="solid", line_color="red",
                          annotation_text=f"Mean: {team_result.mean_wins:.1f}")
            fig.add_vline(x=team_result.wins_10th_pct, line_dash="dash", line_color="gray",
                          annotation_text="10th")
            fig.add_vline(x=team_result.wins_90th_pct, line_dash="dash", line_color="gray",
                          annotation_text="90th")

            st.plotly_chart(fig, use_container_width=True)

        # Team breakdown
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Projected Runs")
            st.metric("Runs Scored", f"{team_proj.projected_runs_scored:.0f}")
            st.metric("Runs Allowed", f"{team_proj.projected_runs_allowed:.0f}")
            st.metric("Run Differential", f"{team_proj.projected_runs_scored - team_proj.projected_runs_allowed:+.0f}")

            if team_proj.sos != 0.500:
                st.subheader("Strength of Schedule")
                st.metric("SOS", f"{team_proj.sos:.3f}")
                st.metric("SOS Win Adjustment", f"{team_proj.sos_win_adjustment:+.1f}")

        with col_right:
            st.subheader("Outcome Probabilities")
            st.metric("90+ Wins", f"{team_result.wins_90_plus_pct:.1f}%")
            st.metric("100+ Wins", f"{team_result.wins_100_plus_pct:.1f}%")
            st.metric("Over .500", f"{team_result.wins_over_500_pct:.1f}%")
            st.metric("Last Place", f"{team_result.last_place_pct:.1f}%")

        # Roster
        st.subheader("Projected Roster")

        roster_tab1, roster_tab2 = st.tabs(["Batters", "Pitchers"])

        with roster_tab1:
            batter_data = []
            for b in sorted(team_proj.batters, key=lambda x: -x.projected_pa):
                batter_data.append({
                    "Name": b.name,
                    "Age": b.age,
                    "PA": round(b.projected_pa),
                    "AVG": f"{b.projected_avg:.3f}",
                    "OBP": f"{b.projected_obp:.3f}",
                    "SLG": f"{b.projected_slg:.3f}",
                    "wRC+": round(b.projected_wrc_plus),
                    "HR": round(b.projected_hr),
                    "Runs": round(b.projected_batting_runs, 1),
                })
            st.dataframe(pd.DataFrame(batter_data), hide_index=True, use_container_width=True)

        with roster_tab2:
            pitcher_data = []
            for p in sorted(team_proj.pitchers, key=lambda x: -x.projected_ip):
                pitcher_data.append({
                    "Name": p.name,
                    "Age": p.age,
                    "Role": p.role,
                    "IP": round(p.projected_ip, 1),
                    "ERA": f"{p.projected_era:.2f}",
                    "FIP": f"{p.projected_fip:.2f}",
                    "WHIP": f"{p.projected_whip:.2f}",
                    "K/9": f"{p.projected_k_per_9:.1f}",
                    "Runs": round(p.projected_pitching_runs, 1),
                })
            st.dataframe(pd.DataFrame(pitcher_data), hide_index=True, use_container_width=True)


def show_player_projections_tab():
    """Display player projection tables."""
    st.header("Player Projections")

    batter_df = st.session_state.batter_df
    pitcher_df = st.session_state.pitcher_df

    player_tab1, player_tab2 = st.tabs(["Batters", "Pitchers"])

    with player_tab1:
        st.subheader("Batter Projections")

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            min_pa = st.slider("Minimum PA", 0, 600, 200)
        with col2:
            teams_filter = st.multiselect(
                "Filter by Team",
                options=sorted(batter_df["Team"].unique()),
                default=[],
            )

        # Apply filters
        filtered = batter_df[batter_df["PA"] >= min_pa]
        if teams_filter:
            filtered = filtered[filtered["Team"].isin(teams_filter)]

        # Sort options
        sort_col = st.selectbox(
            "Sort by",
            options=["wRC+", "HR", "PA", "Batting Runs", "AVG", "OBP", "SLG"],
            index=0,
        )
        filtered = filtered.sort_values(sort_col, ascending=False)

        st.dataframe(filtered, hide_index=True, use_container_width=True)

        # Leaderboards
        st.subheader("Projected Leaders")
        lead_col1, lead_col2, lead_col3 = st.columns(3)

        qualified = batter_df[batter_df["PA"] >= 400]

        with lead_col1:
            st.markdown("**HR Leaders**")
            hr_leaders = qualified.nlargest(5, "HR")[["Name", "Team", "HR"]]
            st.dataframe(hr_leaders, hide_index=True)

        with lead_col2:
            st.markdown("**AVG Leaders**")
            avg_leaders = qualified.nlargest(5, "AVG")[["Name", "Team", "AVG"]]
            st.dataframe(avg_leaders, hide_index=True)

        with lead_col3:
            st.markdown("**wRC+ Leaders**")
            wrc_leaders = qualified.nlargest(5, "wRC+")[["Name", "Team", "wRC+"]]
            st.dataframe(wrc_leaders, hide_index=True)

    with player_tab2:
        st.subheader("Pitcher Projections")

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            min_ip = st.slider("Minimum IP", 0, 150, 50)
        with col2:
            role_filter = st.multiselect(
                "Role",
                options=["SP", "RP", "CL"],
                default=["SP", "RP", "CL"],
            )
        with col3:
            pitch_teams = st.multiselect(
                "Filter by Team",
                options=sorted(pitcher_df["Team"].unique()),
                default=[],
                key="pitcher_teams",
            )

        # Apply filters
        filtered_p = pitcher_df[pitcher_df["IP"] >= min_ip]
        if role_filter:
            filtered_p = filtered_p[filtered_p["Role"].isin(role_filter)]
        if pitch_teams:
            filtered_p = filtered_p[filtered_p["Team"].isin(pitch_teams)]

        # Sort
        sort_col_p = st.selectbox(
            "Sort by",
            options=["IP", "ERA", "FIP", "K", "WHIP", "Pitching Runs"],
            index=0,
            key="pitcher_sort",
        )
        ascending = sort_col_p in ["ERA", "FIP", "WHIP"]
        filtered_p = filtered_p.sort_values(sort_col_p, ascending=ascending)

        st.dataframe(filtered_p, hide_index=True, use_container_width=True)

        # Leaderboards
        st.subheader("Projected Leaders")
        lead_col1, lead_col2, lead_col3 = st.columns(3)

        qualified_sp = pitcher_df[(pitcher_df["IP"] >= 140) & (pitcher_df["Role"] == "SP")]

        with lead_col1:
            st.markdown("**ERA Leaders (SP)**")
            era_leaders = qualified_sp.nsmallest(5, "ERA")[["Name", "Team", "ERA"]]
            st.dataframe(era_leaders, hide_index=True)

        with lead_col2:
            st.markdown("**K Leaders**")
            k_leaders = qualified_sp.nlargest(5, "K")[["Name", "Team", "K"]]
            st.dataframe(k_leaders, hide_index=True)

        with lead_col3:
            st.markdown("**Saves Leaders**")
            sv_leaders = pitcher_df.nlargest(5, "SV")[["Name", "Team", "SV"]]
            st.dataframe(sv_leaders, hide_index=True)


def show_analytics_tab(simulation: SeasonSimulation, projections: Dict[str, TeamProjectionSet]):
    """Display advanced analytics and visualizations."""
    st.header("Analytics")

    # Win distribution comparison
    st.subheader("Win Distribution Comparison")

    teams = sorted(projections.keys())
    selected_teams = st.multiselect(
        "Select teams to compare",
        options=teams,
        default=teams[:3] if len(teams) >= 3 else teams,
        max_selections=6,
    )

    if selected_teams:
        fig = go.Figure()

        for team in selected_teams:
            result = simulation.team_results[team]
            if len(result.win_distribution) > 0:
                fig.add_trace(go.Histogram(
                    x=result.win_distribution,
                    name=team,
                    opacity=0.6,
                    nbinsx=30,
                ))

        fig.update_layout(
            barmode="overlay",
            title="Win Distribution Comparison",
            xaxis_title="Wins",
            yaxis_title="Frequency",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Projected wins scatter plot
    st.subheader("Runs vs Wins")

    scatter_data = []
    for team, proj in projections.items():
        result = simulation.team_results[team]
        scatter_data.append({
            "Team": team,
            "Division": proj.division,
            "Runs Scored": proj.projected_runs_scored,
            "Runs Allowed": proj.projected_runs_allowed,
            "Run Diff": proj.projected_runs_scored - proj.projected_runs_allowed,
            "Projected Wins": result.mean_wins,
            "Playoff %": result.playoff_pct,
        })

    scatter_df = pd.DataFrame(scatter_data)

    fig = px.scatter(
        scatter_df,
        x="Run Diff",
        y="Projected Wins",
        color="Division",
        size="Playoff %",
        hover_name="Team",
        title="Run Differential vs Projected Wins",
        hover_data=["Runs Scored", "Runs Allowed", "Playoff %"],
    )
    fig.update_layout(xaxis_title="Run Differential", yaxis_title="Projected Wins")
    st.plotly_chart(fig, use_container_width=True)

    # Division strength
    st.subheader("Division Strength")

    div_data = []
    for div in DIVISIONS:
        div_teams = [t for t, d in TEAM_TO_DIVISION.items() if d == div and t in projections]
        if div_teams:
            avg_wins = np.mean([simulation.team_results[t].mean_wins for t in div_teams])
            div_data.append({"Division": div, "Avg Projected Wins": avg_wins})

    div_df = pd.DataFrame(div_data)
    div_df = div_df.sort_values("Avg Projected Wins", ascending=False)

    fig = px.bar(
        div_df,
        x="Division",
        y="Avg Projected Wins",
        color="Avg Projected Wins",
        title="Average Projected Wins by Division",
        color_continuous_scale="RdYlGn",
    )
    fig.update_layout(yaxis_title="Average Team Wins", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
