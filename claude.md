# Baseball Season Prediction Model

## Quick Start

```bash
# Pull latest changes
git pull

# Install dependencies
pip install -r requirements.txt

# Option 1: Run the web dashboard (recommended)
streamlit run src/dashboard.py

# Option 2: Run CLI projections with roster changes and player variance
python -m src.main project --year 2026 -r data/roster_changes/synced_changes.csv --player-variance

# Option 3: CLI with auto-sync from MLB API (fetches current rosters automatically)
python -m src.main project --year 2026 --auto-sync --player-variance

# Generate synced roster changes CSV first (optional)
python -m src.main sync-rosters --year 2025 -o data/roster_changes/synced_changes.csv
```

### CLI Options for `project` Command

| Option | Description |
|--------|-------------|
| `--year`, `-y` | Projection year (default: 2025) |
| `--iterations`, `-n` | Number of Monte Carlo simulations (default: 10,000) |
| `--roster-changes`, `-r` | Path to roster changes CSV |
| `--auto-sync` | Auto-fetch current rosters from MLB API |
| `--player-variance` | Use player-level variance (more realistic) |
| `--output-dir`, `-o` | Output directory (default: output/) |
| `--seed` | Random seed for reproducibility |

### Output Files

After running projections, you'll find in `output/`:
- `projection_2026.txt` - Full text report with standings and playoff odds
- `standings_2026.csv` - Team projections CSV (includes SOS data)
- `projected_batters.csv` - Individual batter projections
- `projected_pitchers.csv` - Individual pitcher projections

---

## Project Overview

Build a Monte Carlo simulation system that projects MLB season outcomes based on roster changes and historical player performance data.

## Core Architecture Decisions

### Simulation Approach
- Use **aggregate team-level projections**, not game-by-game simulation
- Project team runs scored and runs allowed, then convert to expected wins using Pythagorean expectation (runs^1.83 / (runs^1.83 + runs_allowed^1.83))
- Game-by-game simulation is out of scope for v1—too much complexity for marginal benefit

### Monte Carlo Simulation
- Run thousands of season simulations (default: 10,000) to generate probability distributions
- Output should include confidence intervals, not just point estimates
- Key outputs: win distribution percentiles, division/playoff odds, individual stat leader probabilities

### Data Sources
- Primary: **FanGraphs** and **Baseball Reference** for player stats
- Use **pybaseball** library for data retrieval
- Statcast/Baseball Savant for advanced metrics if needed (batted ball data, pitch characteristics)

## Key Features

### Inputs
1. **Baseline player data**: Prior season stats, multi-year performance history
2. **Roster changes**: Trades, free agent signings, retirements, call-ups (CSV or structured input)
3. **Team rosters**: Map players to teams for the projection year

### Outputs
1. **Team projections**:
   - Win-loss record distributions (mean, median, 10th/90th percentile)
   - Division standings probabilities
   - Playoff odds

2. **Individual leaders**:
   - Projected league leaders in key categories
   - Probability distributions for counting stats (HR, RBI, SB, W, K, SV)
   - Projected rate stats (AVG, OBP, SLG, ERA, WHIP, K/9)

## Projection Methodology

### Player Projections
- Use weighted average of recent seasons (weight recent years more heavily)
- Apply **regression to the mean** for small sample sizes
- Implement basic **aging curves** (peak age ~27, gradual decline after)
- Account for **park factors** when players change teams (use FanGraphs park factors as primary source)
- **Playing time projections**: Project PA for hitters and IP for pitchers based on:
  - Historical playing time patterns
  - Role expectations (starter vs bench, rotation vs bullpen)
  - Age-based injury risk adjustments

### Team Projections
- Use **runs-based approach** (not WAR) to avoid replacement-level assumptions:
  - Hitters: Project wRC (weighted runs created) based on wRC+ and projected PA
  - Pitchers: Project runs allowed using FIP-based runs and projected IP
- Convert to team runs scored/allowed by summing individual contributions
- Apply Pythagorean expectation for win projection
- Add variance for Monte Carlo simulation based on historical team-level randomness

### Pitching Staff Modeling
- **Starting rotation**: Distribute ~1000 team IP among projected starters based on role and durability
- **Bullpen**: Allocate remaining ~450 IP across relievers, weighted by leverage expectations
- Account for innings overflow when starters underperform projections

## Technical Stack

- **Language**: Python 3.11+
- **Data retrieval**: pybaseball
- **Data manipulation**: pandas, numpy
- **Simulation**: numpy for Monte Carlo
- **Visualization**: matplotlib or plotly for distributions
- **CLI or simple UI**: Start with CLI, consider Streamlit for v2

### Core Dependencies

```
# requirements.txt
pybaseball>=2.2.7
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0        # for statistical distributions
tqdm>=4.65.0         # progress bars for simulations
click>=8.1.0         # CLI framework
```

## Project Structure

```
MLB/
├── src/
│   ├── data/
│   │   ├── fetch.py          # pybaseball wrappers
│   │   ├── cache.py          # local caching of fetched data
│   │   ├── mlb_api.py        # MLB Stats API for current rosters
│   │   └── roster_changes.py # parse roster change inputs
│   ├── projections/
│   │   ├── player.py         # individual player projections
│   │   ├── playing_time.py   # PA/IP projections by role
│   │   ├── team.py           # aggregate to team level
│   │   ├── schedule.py       # strength of schedule calculations
│   │   ├── adjustments.py    # aging curves, park factors, regression
│   │   └── projector.py      # main projection orchestrator
│   ├── simulation/
│   │   ├── monte_carlo.py    # run season simulations
│   │   └── standings.py      # compute standings, playoff odds
│   ├── output/
│   │   ├── reports.py        # generate summary reports
│   │   └── visualize.py      # charts and distributions
│   ├── main.py               # CLI entry point
│   └── dashboard.py          # Streamlit web dashboard
├── data/
│   ├── roster_changes/       # input CSV files
│   └── cache/                # cached API responses
├── tests/
│   ├── test_projections.py   # projection unit tests
│   ├── test_schedule.py      # SOS calculation tests
│   ├── test_roster_sync.py   # roster sync tests
│   └── test_player_variance.py  # injury/upside variance tests
├── requirements.txt
└── claude.md                 # this file
```

## Development Phases

### Phase 1: Data Pipeline
- Fetch historical player data via pybaseball
- Parse roster change inputs
- Build current roster snapshots for each team

### Phase 2: Player Projections
- Implement weighted historical averaging
- Add regression to the mean
- Implement aging curves
- Add park factor adjustments
- **Project playing time** (PA/IP) for each player based on role and history

### Phase 3: Team Projections
- Aggregate player projections to team level
- Implement Pythagorean expectation
- Model pitching staff innings distribution (rotation + bullpen)
- Validate against recent seasons

### Phase 4: Monte Carlo Simulation
- **Variance modeling** with two components:
  - *Player-level variance*: Based on historical year-to-year consistency (some players are volatile, others stable)
  - *Team-level residual variance*: ~6-8 wins of unexplained variance even after accounting for player performance
- Run bulk simulations (10,000 iterations default)
- Compute playoff odds and confidence intervals

### Phase 5: Output & Polish
- Generate readable reports
- Add visualizations
- CLI interface for running projections

## Validation

- Backtest against recent completed seasons (2022, 2023, 2024)
- Compare projection accuracy to public systems (ZiPS, Steamer)
- Check calibration of probability estimates

## Implemented Features

### Strength of Schedule (SOS)
Adjusts win projections based on opponent quality:
- Calculates weighted SOS from divisional (52 games), league (66 games), and interleague (44 games) opponents
- Applies win adjustment (~1-3 wins) based on schedule difficulty
- Teams in tough divisions (AL East) get adjusted down, weak divisions adjusted up

### Automatic Roster Sync
Fetches current rosters from MLB Stats API:
- `--auto-sync` flag on project command
- Compares current 40-man rosters to historical data
- Detects trades, signings, and team changes automatically
- Manual roster changes CSV takes precedence over auto-detected changes

### Prospect Detection
Automatically identifies and boosts playing time for young breakout players:
- Detects players <25 years old with <900 career PA
- Boosts playing time if performance warrants it
- Separate logic for batters vs pitchers

### Park Factor Adjustments
Adjusts projections when players change teams:
- Uses FanGraphs park factors
- Applies adjustment to batting stats for team changes

### Injury Variance Modeling
Samples playing time with age-based variance instead of using deterministic projections:
- Young players (<26): ~8% coefficient of variation (most durable)
- Prime (26-30): ~12% coefficient of variation
- Veteran (31-34): ~18% coefficient of variation
- Old (35+): ~25% coefficient of variation (highest injury risk)
- Each simulation samples PA/IP from a normal distribution around the projection
- Allows for realistic injury scenarios (e.g., 32-year-old gets 350 PA instead of projected 550)

### Young Player Upside
Widens performance distributions for breakout candidates:
- Applies to players <26 years old with <2 seasons of MLB data
- Stat variance (wRC+, FIP) increased by 50% for qualifying players
- Models the higher uncertainty and potential breakout/bust scenarios for prospects
- Combined with injury variance, creates realistic outcome ranges for young players

### Streamlit Dashboard (v2)
Interactive web interface for projections:
- Run with: `streamlit run src/dashboard.py`
- **Standings & Playoff Odds**: Division standings, playoff probability leaderboard
- **Team Details**: Win distributions, roster breakdowns, outcome probabilities
- **Player Projections**: Filterable tables for batters and pitchers, stat leaders
- **Analytics**: Win distribution comparisons, run differential scatter plots, division strength

## Out of Scope

- In-season updates
- Injury projections (beyond age-based decline)
- Full minor league/prospect integration
- Trade deadline simulation
- Game-by-game simulation

## Notes for Development

- Cache all API calls to avoid rate limiting and speed up iteration
- Start simple—get the pipeline working end-to-end before adding sophistication
- Log intermediate outputs so you can debug projection issues
- Use 2024 season as primary test case since roster changes are fresh
