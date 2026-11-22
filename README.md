# Genetic Evolution and Emergent Learning in Multi-Agent Systems

Two populations of agents live on a grid with wild food and farms. Each agent carries five genes that
set how it explores, farms, avoids, cooperates and fights. There is no fitness function: an agent
has children only if it survives, gathers enough food and finds a partner, so the genes that spread
are the ones that work. The project asks how **resource scarcity** changes which traits evolve and
how the two populations treat each other.

- **[DESIGN.md](DESIGN.md)**: the rules, the genes, why this is a genetic algorithm, every parameter,
  the calibration, and the limitations. Start here.
- **[RESULTS.md](RESULTS.md)**: each experiment's question, figure, numbers and finding.

## Setup

Needs Python 3.12 or newer (Mesa 3.5.1 requires it).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

```bash
python -m pytest -q
```

51 tests, about 15 seconds. `test_reproduction.py` covers the model's rules; `test_experiment.py`
covers seeds, headless mode, conservation of resources, the neutral control, lifetime records and
gene pools.

## Live demo (dashboard)

```bash
python app.py
```

Then open http://127.0.0.1:5000. (If port 5000 is taken, on a Mac often by AirPlay Receiver, run
`flask --app app run --port 5050` and open http://127.0.0.1:5050.)

Controls: **World** (SMALL 3×3 debug, LARGE 30×30, EXPERIMENT 50×50), **Environment** (the five
abundance levels; ABUNDANT = level 5, SCARCE = level 1), **Starting agents**, **Seed**, then **Reset**.

"Starting agents" loads saved gene pools from `gene_pools/`:

| Choice | Population A | Population B |
|---|---|---|
| Random (generation 0) | random genes | random genes |
| A evolved in plenty · B random | evolved 2,000 steps at level 5 | random genes |
| A evolved in scarcity · B random | evolved 2,000 steps at level 1 | random genes |
| A evolved in plenty · B evolved in scarcity | level-5 veterans | level-1 veterans |

A suggested demo: EXPERIMENT world, Level 2, "A evolved in plenty · B random", speed 40. Watch the
"Average farming" chart: B starts at 0.5 and climbs towards A. Then switch to Level 3 and watch
every agent settle on a farm and stop moving within about 50 steps.

### Fallback animation

`demo.mp4` (44 seconds) shows evolved agents against random generation-0 agents in one world at
level 2. To render it again (needs ffmpeg, about 2 minutes):

```bash
python analysis/make_demo.py            # writes demo.mp4
python analysis/make_demo.py demo.gif   # no ffmpeg needed
```

## Experiments

Each script runs all its simulations on every CPU core and writes compressed CSV files to
`results/`. All use the 50×50 EXPERIMENT world and seeds 0 to 29. Times are for an 8-core laptop.

```bash
python experiments/e1_drift.py          # E1: selection vs neutral control        60 runs,  2 min
python experiments/e2_gradient.py       # E2: five abundance levels              150 runs,  6 min
python experiments/e2_neutral.py        # E2: neutral control at the other four levels  120 runs, 6 min
python experiments/e5_selection.py      # E5: gene vs offspring count (uses E2's lifetime records)
python experiments/e3_shared_world.py   # E3: plenty veterans meet scarcity veterans (uses E2's gene pools)  300 runs, 12 min
python experiments/e4_shock.py          # E4: environmental shock                120 runs, 15 min
python experiments/stretch.py           # Stretch: fight cost sweep, farms off   180 runs, 10 min
python experiments/calibrate.py         # How the abundance levels were chosen (already done; levels are frozen)
```

Run E2 before E5 and E3. `results/e2_lifetimes.csv.gz` (one row per dead agent, about 60 MB) is not
in git; `e2_gradient.py` recreates it. Everything else in `results/` is committed, so the figures
can be regenerated without rerunning anything.

## Figures

```bash
python analysis/make_figures.py         # all figures and tables
python analysis/make_figures.py e2 e5   # only some
```

Writes 300 dpi PNGs to `figures/` and the numbers behind them to `results/tables/`.

| Figure | Shows |
|---|---|
| `e1_selection_vs_neutral.png` | Gene averages over time, with selection and in the neutral control |
| `e2_genes_by_level.png` | Final gene averages at each abundance level, against the neutral control |
| `e2_behaviour_by_level.png` | Population size, encounter rate, cooperation, fights, raids, farms, theft, starvation, extinction |
| `e3_share_over_time.png`, `e3_outcome_by_level.png` | Who dominates when plenty veterans and scarcity veterans share a world |
| `e4_shock.png` | Gene averages before and after the environment switches |
| `e5_gene_vs_offspring.png` | How each gene relates to offspring count at each level |
| `stretch_fight_cost.png` | Risk tolerance, cooperation and fight rate against the cost of a fight |

## Files

| Path | What it is |
|---|---|
| `model.py` | The simulation (Mesa 3.5.1): `SocietyAgent`, `SocietyModel` |
| `config.py` | Every parameter, the abundance levels, the worlds |
| `app.py`, `templates/`, `static/` | The Flask dashboard |
| `experiments/` | One script per experiment, plus `common.py` (parallel runner, saving, gene pools) |
| `analysis/make_figures.py` | Figures and tables from saved results |
| `analysis/make_demo.py` | The fallback animation |
| `gene_pools/` | Saved evolved genomes for the dashboard |
| `results/`, `figures/` | Saved data and figures |
| `test_reproduction.py`, `test_experiment.py` | Tests |

## Using the model from Python

```python
from model import SocietyModel

model = SocietyModel(3, "EXPERIMENT", seed=7, headless=True, record_every=10)
for _ in range(2000):
    model.step()
data = model.datacollector.get_model_vars_dataframe()   # one row per 10 timesteps
lives = model.lifetime_records                          # one dictionary per dead agent
pool = model.living_genomes("A")                        # population A's gene pool
```

Other options: `neutral=True` (neutral control), `overrides={"FIGHT_COST": 0.5}`,
`gene_pools={"A": pool}`, `frozen_genomes=True`, and `model.set_environment(1)` to switch level
mid-run. See DESIGN.md, section 6.
