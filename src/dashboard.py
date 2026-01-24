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
from src.data.cache_results import (
    load_simulation_results,
    get_available_cache_files,
    compare_simulations,
    load_raw_cache_file,
    load_historical_time_series,
    get_trend_summary,
)

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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Standings & Playoff Odds",
        "🏟️ Team Details",
        "👥 Player Projections",
        "📊 Analytics",
        "📉 Trends",
        "🔄 Compare Runs",
    ])

    with tab1:
        show_standings_tab(simulation, projections)

    with tab2:
        show_team_details_tab(simulation, projections)

    with tab3:
        show_player_projections_tab()

    with tab4:
        show_analytics_tab(simulation, projections)

    with tab5:
        show_trends_tab()

    with tab6:
        show_comparison_tab()


def show_standings_tab(simulation: SeasonSimulation, projections: Dict[str, TeamProjectionSet]):
    """Display standings and playoff odds."""
    st.header("Projected Standings & Playoff Odds")

    # Get playoff odds DataFrame
    odds_df = simulation.get_playoff_odds_df()

    # Add division info
    odds_df["Division"] = odds_df["Team"].map(TEAM_TO_DIVISION)

    # World Series Championship Odds - Featured section
    st.subheader("World Series Championship Odds")

    # Top 10 championship contenders
    top_ws = odds_df.nlargest(10, "WS Champ %")

    fig_ws = px.bar(
        top_ws,
        x="Team",
        y="WS Champ %",
        color="Division",
        title="Top 10 World Series Contenders",
        text="WS Champ %",
    )
    fig_ws.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_ws.update_layout(yaxis_title="World Series Win Probability (%)", xaxis_title="")
    st.plotly_chart(fig_ws, use_container_width=True)

    # Championship odds table
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**American League Pennant Odds**")
        al_teams = odds_df[odds_df["Division"].str.startswith("AL")].sort_values("Pennant %", ascending=False)
        st.dataframe(
            al_teams[["Team", "Proj W", "Playoff %", "Pennant %", "WS Champ %"]].head(8),
            hide_index=True,
            use_container_width=True,
        )

    with col2:
        st.markdown("**National League Pennant Odds**")
        nl_teams = odds_df[odds_df["Division"].str.startswith("NL")].sort_values("Pennant %", ascending=False)
        st.dataframe(
            nl_teams[["Team", "Proj W", "Playoff %", "Pennant %", "WS Champ %"]].head(8),
            hide_index=True,
            use_container_width=True,
        )

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
                div_teams[["Team", "Proj W", "10th", "90th", "Div %", "WC %", "Playoff %", "WS Champ %"]],
                hide_index=True,
                use_container_width=True,
            )

    with col_nl:
        st.markdown("### National League")
        for div in ["NL East", "NL Central", "NL West"]:
            div_teams = odds_df[odds_df["Division"] == div].sort_values("Proj W", ascending=False)
            st.markdown(f"**{div}**")
            st.dataframe(
                div_teams[["Team", "Proj W", "10th", "90th", "Div %", "WC %", "Playoff %", "WS Champ %"]],
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

        # Overview metrics - main row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Projected Wins", f"{team_result.mean_wins:.1f}")
        with col2:
            st.metric("Playoff Odds", f"{team_result.playoff_pct:.1f}%")
        with col3:
            st.metric("Division Winner", f"{team_result.division_winner_pct:.1f}%")
        with col4:
            st.metric("Wild Card", f"{team_result.wild_card_pct:.1f}%")

        # Championship odds row
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            st.metric("Pennant (LCS)", f"{team_result.pennant_pct:.1f}%")
        with col6:
            st.metric("WS Champion", f"{team_result.champion_pct:.1f}%")
        with col7:
            pass  # Empty for layout
        with col8:
            pass  # Empty for layout

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

        # Get player data from session state DataFrames (works with cached data)
        batter_df = st.session_state.batter_df
        pitcher_df = st.session_state.pitcher_df

        with roster_tab1:
            if batter_df is not None and len(batter_df) > 0:
                team_batters = batter_df[batter_df["Team"] == selected_team].copy()
                team_batters = team_batters.sort_values("PA", ascending=False)
                display_cols = ["Name", "Age", "PA", "AVG", "OBP", "SLG", "wRC+", "HR", "Batting Runs"]
                available_cols = [c for c in display_cols if c in team_batters.columns]
                st.dataframe(team_batters[available_cols], hide_index=True, use_container_width=True)
            else:
                st.info("No batter data available")

        with roster_tab2:
            if pitcher_df is not None and len(pitcher_df) > 0:
                team_pitchers = pitcher_df[pitcher_df["Team"] == selected_team].copy()
                team_pitchers = team_pitchers.sort_values("IP", ascending=False)
                display_cols = ["Name", "Age", "Role", "IP", "ERA", "FIP", "WHIP", "K/9", "Pitching Runs"]
                available_cols = [c for c in display_cols if c in team_pitchers.columns]
                st.dataframe(team_pitchers[available_cols], hide_index=True, use_container_width=True)
            else:
                st.info("No pitcher data available")


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


def show_trends_tab():
    """Display historical trends in projections over time."""
    st.header("Projection Trends Over Time")

    # Load time series data
    time_series = load_historical_time_series()

    if time_series.empty:
        st.info("Not enough historical data to show trends.")
        st.markdown("""
        Trends will appear after running simulations on multiple days.
        Each daily run with `--auto-sync` captures roster changes and projection updates.
        """)
        return

    # Get date range
    min_date = time_series["date"].min()
    max_date = time_series["date"].max()
    num_days = (max_date - min_date).days + 1

    st.markdown(f"**Tracking {num_days} days** from {min_date.strftime('%b %d')} to {max_date.strftime('%b %d, %Y')}")

    # Summary of biggest movers
    st.subheader("Biggest Movers (Since Tracking Began)")

    summary = get_trend_summary(time_series)
    if not summary.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Rising**")
            rising = summary[summary["wins_change"] > 0.1].head(5)
            if not rising.empty:
                for _, row in rising.iterrows():
                    st.markdown(
                        f"**{row['team']}**: {row['last_wins']:.1f} wins "
                        f"(+{row['wins_change']:.1f})"
                    )
            else:
                st.caption("No significant increases")

        with col2:
            st.markdown("**Falling**")
            falling = summary[summary["wins_change"] < -0.1].sort_values("wins_change").head(5)
            if not falling.empty:
                for _, row in falling.iterrows():
                    st.markdown(
                        f"**{row['team']}**: {row['last_wins']:.1f} wins "
                        f"({row['wins_change']:.1f})"
                    )
            else:
                st.caption("No significant decreases")

    # Team selector for detailed charts
    st.subheader("Team Projection History")

    teams = sorted(time_series["team"].unique())

    # Default to showing a few interesting teams
    default_teams = ["NYY", "LAD", "ATL", "HOU", "PHI"]
    default_selection = [t for t in default_teams if t in teams][:3]

    selected_teams = st.multiselect(
        "Select teams to compare:",
        options=teams,
        default=default_selection,
        max_selections=8,
    )

    if selected_teams:
        team_data = time_series[time_series["team"].isin(selected_teams)]

        # Win projection trends
        st.markdown("#### Projected Wins Over Time")
        fig_wins = px.line(
            team_data,
            x="date",
            y="mean_wins",
            color="team",
            markers=True,
            labels={"mean_wins": "Projected Wins", "date": "Date", "team": "Team"},
        )
        fig_wins.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_wins, use_container_width=True)

        # Playoff odds trends
        st.markdown("#### Playoff Odds Over Time")
        fig_playoff = px.line(
            team_data,
            x="date",
            y="playoff_pct",
            color="team",
            markers=True,
            labels={"playoff_pct": "Playoff %", "date": "Date", "team": "Team"},
        )
        fig_playoff.update_layout(
            hovermode="x unified",
            yaxis_ticksuffix="%",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_playoff, use_container_width=True)

        # Championship odds trends
        st.markdown("#### World Series Champion Odds Over Time")
        fig_champ = px.line(
            team_data,
            x="date",
            y="champion_pct",
            color="team",
            markers=True,
            labels={"champion_pct": "Champion %", "date": "Date", "team": "Team"},
        )
        fig_champ.update_layout(
            hovermode="x unified",
            yaxis_ticksuffix="%",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_champ, use_container_width=True)

    # Full data table
    with st.expander("View Raw Data"):
        st.dataframe(
            summary[[
                "team", "first_wins", "last_wins", "wins_change",
                "first_playoff", "last_playoff", "playoff_change",
            ]].rename(columns={
                "team": "Team",
                "first_wins": f"Wins ({min_date.strftime('%m/%d')})",
                "last_wins": f"Wins ({max_date.strftime('%m/%d')})",
                "wins_change": "Wins Chg",
                "first_playoff": f"Playoff ({min_date.strftime('%m/%d')})",
                "last_playoff": f"Playoff ({max_date.strftime('%m/%d')})",
                "playoff_change": "Playoff Chg",
            }),
            hide_index=True,
            use_container_width=True,
        )


def show_comparison_tab():
    """Display comparison between simulation runs."""
    st.header("Compare Simulation Runs")

    # Get available cache files (non-latest only, for comparison)
    cache_files = get_available_cache_files()
    historical_files = [f for f in cache_files if not f["is_latest"] and f.get("date")]

    if not historical_files:
        st.info("No historical simulation runs found to compare against.")
        st.markdown("""
        To enable comparison:
        1. Run `python -m src.main generate-cache --year 2026` on different days
        2. Each day's run is saved separately for comparison

        The comparison will show how projections and rosters have changed between runs.
        """)
        return

    if not st.session_state.from_cache or not st.session_state.metadata:
        st.warning("Comparison requires cached projections to be loaded.")
        return

    current_year = st.session_state.metadata.get("year", 2026)

    # Filter to same year
    same_year_files = [f for f in historical_files if f["year"] == current_year]

    if not same_year_files:
        st.info(f"No historical runs found for {current_year} to compare against.")
        return

    # Let user select a historical run to compare
    st.subheader("Select Previous Run")

    options = {f["date"]: f for f in same_year_files}
    selected_date = st.selectbox(
        "Compare current projections to:",
        options=list(options.keys()),
        format_func=lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]}" if x and len(x) >= 8 else x,
    )

    if selected_date:
        previous_file = options[selected_date]
        previous_data = load_raw_cache_file(previous_file["path"])

        if previous_data is None:
            st.error("Could not load previous simulation data.")
            return

        # Load current data for comparison
        from pathlib import Path
        current_path = Path(__file__).parent.parent / "data" / "cache" / f"simulation_{current_year}_latest.json"
        current_data = load_raw_cache_file(current_path)

        if current_data is None:
            st.error("Could not load current simulation data.")
            return

        # Compare
        comparison = compare_simulations(current_data, previous_data)

        # Display comparison
        st.subheader("Changes Since Previous Run")

        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"**Current:** {current_data['metadata']['timestamp'][:10]}")
        with col2:
            st.caption(f"**Previous:** {previous_data['metadata']['timestamp'][:10]}")

        # Create comparison DataFrame
        comp_df = pd.DataFrame(comparison["team_changes"])

        if len(comp_df) == 0:
            st.info("No changes detected between runs.")
            return

        # Format for display
        comp_df["Wins Change"] = comp_df["delta_wins"].apply(
            lambda x: f"+{x:.1f}" if x > 0 else f"{x:.1f}"
        )
        comp_df["Playoff Change"] = comp_df["delta_playoff"].apply(
            lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%"
        )

        # Biggest movers
        st.subheader("Biggest Movers (by Wins)")

        # Rising teams
        rising = comp_df[comp_df["delta_wins"] > 0.5].head(5)
        falling = comp_df[comp_df["delta_wins"] < -0.5].head(5)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📈 Rising**")
            if len(rising) > 0:
                for _, row in rising.iterrows():
                    st.markdown(
                        f"**{row['team']}**: {row['current_wins']:.1f} wins "
                        f"({row['Wins Change']})"
                    )
            else:
                st.caption("No significant increases")

        with col2:
            st.markdown("**📉 Falling**")
            if len(falling) > 0:
                for _, row in falling.iterrows():
                    st.markdown(
                        f"**{row['team']}**: {row['current_wins']:.1f} wins "
                        f"({row['Wins Change']})"
                    )
            else:
                st.caption("No significant decreases")

        # Roster Changes Section
        player_diffs = comparison.get("player_diffs", [])
        if player_diffs:
            st.subheader("Roster Changes")
            st.markdown("Players added, removed, or moved between runs:")

            # Group by change type
            added = [p for p in player_diffs if p.get("change") == "added"]
            removed = [p for p in player_diffs if p.get("change") == "removed"]
            moved = [p for p in player_diffs if p.get("change") == "moved"]

            # Show summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Players Added", len(added))
            with col2:
                st.metric("Players Removed", len(removed))
            with col3:
                st.metric("Players Moved", len(moved))

            # Tabs for different change types
            roster_tab1, roster_tab2, roster_tab3 = st.tabs(["Removed", "Moved", "Added"])

            with roster_tab1:
                if removed:
                    st.markdown("**Players no longer on rosters** (impact = runs lost by team)")
                    removed_df = pd.DataFrame(removed)
                    removed_df = removed_df.sort_values("runs_impact", ascending=True)
                    removed_df["Runs Impact"] = removed_df["runs_impact"].apply(
                        lambda x: f"{x:+.1f}"
                    )
                    display_cols = ["name", "type", "team", "Runs Impact", "details"]
                    display_cols = [c for c in display_cols if c in removed_df.columns]
                    removed_df.columns = [c.title() if c != "Runs Impact" else c for c in removed_df.columns]
                    st.dataframe(
                        removed_df[["Name", "Type", "Team", "Runs Impact"]].head(20),
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.caption("No players removed")

            with roster_tab2:
                if moved:
                    st.markdown("**Players who changed teams**")
                    moved_df = pd.DataFrame(moved)
                    moved_df = moved_df.sort_values("runs_impact", key=abs, ascending=False)
                    moved_df["Runs"] = moved_df["runs_impact"].apply(lambda x: f"{x:+.1f}")
                    moved_df["Move"] = moved_df.apply(
                        lambda r: f"{r.get('from_team', '?')} → {r.get('to_team', '?')}", axis=1
                    )
                    st.dataframe(
                        moved_df[["name", "type", "Move", "Runs"]].head(20).rename(
                            columns={"name": "Name", "type": "Type"}
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.caption("No players moved")

            with roster_tab3:
                if added:
                    st.markdown("**New players on rosters** (impact = runs gained by team)")
                    added_df = pd.DataFrame(added)
                    added_df = added_df.sort_values("runs_impact", ascending=False)
                    added_df["Runs Impact"] = added_df["runs_impact"].apply(
                        lambda x: f"+{x:.1f}" if x > 0 else f"{x:.1f}"
                    )
                    st.dataframe(
                        added_df[["name", "type", "team", "Runs Impact"]].head(20).rename(
                            columns={"name": "Name", "type": "Type", "team": "Team"}
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.caption("No players added")

            # Team impact summary
            st.subheader("Roster Change Impact by Team")

            # Calculate net runs impact per team
            team_impact = {}
            for p in player_diffs:
                runs = p.get("runs_impact", 0)
                if p.get("change") == "removed":
                    team = p.get("team")
                    if team:
                        team_impact[team] = team_impact.get(team, 0) + runs  # runs is negative
                elif p.get("change") == "added":
                    team = p.get("team")
                    if team:
                        team_impact[team] = team_impact.get(team, 0) + runs
                elif p.get("change") == "moved":
                    from_team = p.get("from_team")
                    to_team = p.get("to_team")
                    if from_team:
                        team_impact[from_team] = team_impact.get(from_team, 0) - runs
                    if to_team:
                        team_impact[to_team] = team_impact.get(to_team, 0) + runs

            if team_impact:
                impact_df = pd.DataFrame([
                    {"Team": t, "Net Runs Impact": r}
                    for t, r in sorted(team_impact.items(), key=lambda x: x[1], reverse=True)
                ])
                impact_df["Net Runs Impact"] = impact_df["Net Runs Impact"].apply(
                    lambda x: f"+{x:.1f}" if x > 0 else f"{x:.1f}"
                )
                st.dataframe(impact_df, hide_index=True, use_container_width=True)

        # Full comparison table
        st.subheader("Full Comparison")

        display_df = comp_df[[
            "team", "current_wins", "previous_wins", "Wins Change",
            "current_playoff", "previous_playoff", "Playoff Change"
        ]].copy()
        display_df.columns = [
            "Team", "Current Wins", "Prev Wins", "Δ Wins",
            "Current Playoff %", "Prev Playoff %", "Δ Playoff"
        ]
        display_df["Current Wins"] = display_df["Current Wins"].round(1)
        display_df["Prev Wins"] = display_df["Prev Wins"].round(1)
        display_df["Current Playoff %"] = display_df["Current Playoff %"].round(1)
        display_df["Prev Playoff %"] = display_df["Prev Playoff %"].round(1)

        st.dataframe(display_df, hide_index=True, use_container_width=True)

        # Visualization
        st.subheader("Win Changes Visualization")

        fig = px.bar(
            comp_df.head(15),
            x="team",
            y="delta_wins",
            color="delta_wins",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            title="Projected Win Changes by Team",
        )
        fig.update_layout(
            xaxis_title="Team",
            yaxis_title="Change in Projected Wins",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
