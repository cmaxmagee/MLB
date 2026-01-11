# Baseball Season Prediction Model

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
baseball-predictor/
├── src/
│   ├── data/
│   │   ├── fetch.py          # pybaseball wrappers
│   │   ├── cache.py          # local caching of fetched data
│   │   └── roster_changes.py # parse roster change inputs
│   ├── projections/
│   │   ├── player.py         # individual player projections
│   │   ├── playing_time.py   # PA/IP projections by role
│   │   ├── pitching_staff.py # rotation/bullpen innings allocation
│   │   ├── team.py           # aggregate to team level
│   │   └── adjustments.py    # aging curves, park factors, regression
│   ├── simulation/
│   │   ├── monte_carlo.py    # run season simulations
│   │   └── standings.py      # compute standings, playoff odds
│   ├── output/
│   │   ├── reports.py        # generate summary reports
│   │   └── visualize.py      # charts and distributions
│   └── main.py               # CLI entry point
├── data/
│   ├── roster_changes/       # input CSV files
│   └── cache/                # cached API responses
├── tests/
├── requirements.txt
└── README.md
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

## Out of Scope (for v1)

- In-season updates
- Injury projections
- Minor league/prospect integration
- Trade deadline simulation
- Game-by-game simulation
- Web interface (Streamlit could be v2)

## Notes for Development

- Cache all API calls to avoid rate limiting and speed up iteration
- Start simple—get the pipeline working end-to-end before adding sophistication
- Log intermediate outputs so you can debug projection issues
- Use 2024 season as primary test case since roster changes are fresh
