# Genetic Evolution and Emergent Learning in Multi-Agent Systems

Two populations of agents live on a grid with wild food and farms. Each agent carries genes that set
how it explores, farms, avoids rivals, cooperates and fights. Agents that survive and gather enough
food have children, who inherit their parents' genes with crossover and mutation. The project asks
how resource scarcity changes which traits evolve.

There are two versions of the model. Model 1 has five genes. Model 2 (`version=2` in `model.py`)
adds body genes, an explicit fitness function, soil that wears out and a trade bonus.

Model 2 also has an optional seed rule, off by default (`SEED_MODE` in `config.py`): planting a farm
needs a seed, which agents pick up by eating wild food in the shared land.

## Setup

Needs Python 3.12 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the dashboard

```bash
python app.py
```

Then open http://127.0.0.1:5000. Choose "Model 2" in the Model menu and press Reset for the
three-zone map; its "Raiders vs Farmers" start is a hand-set scenario, not an evolved one.

`demo.mp4` is a recording of that scenario (`python analysis/make_demo.py` renders it again).
`demo_model1.mp4` is a recorded Model 1 run.

## Run the experiments

```bash
python experiments/e1_drift.py            # Model 1: e1 to e5, one script each
python experiments/m_experiments.py m2    # Model 2: m0, m2, m3, m4, m5, m7
python experiments/m_experiments.py m8    # Raiders vs Farmers (hand-set genomes, not evolved)
python experiments/m_experiments.py m9    # Optional seed rule: off, to start, always
```

Results are saved in `results/`. They are already in the repository, so this step can be skipped.

## Make the figures

```bash
python analysis/make_figures.py           # Model 1
python analysis/make_figures_model2.py    # Model 2, with tables in results/model2/tables/
```

Figures are saved in `figures/`.

## Tests

```bash
python -m unittest -v test_reproduction
```

## Files

| Path | What it is |
|---|---|
| `model.py` | The simulation |
| `config.py` | Every parameter |
| `app.py`, `templates/`, `static/` | The dashboard |
| `experiments/` | Experiment scripts |
| `analysis/` | Figure scripts |
| `results/`, `figures/` | Saved data and figures |
