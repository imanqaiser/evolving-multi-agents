# Genetic Evolution and Emergent Learning in Multi-Agent Systems

Two populations of agents live on a grid with wild food and farms. Each agent carries genes that set
how it explores, farms, avoids rivals, cooperates and fights. Agents that survive and gather enough
food have children, who inherit their parents' genes with crossover and mutation. The project asks
how resource scarcity changes which traits evolve.

There are two versions of the model. Model 1 has five genes. Model 2 (`version=2` in `model.py`)
adds body genes, an explicit fitness function, soil that wears out and a trade bonus.

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

Then open http://127.0.0.1:5000. `demo.mp4` is a recorded run.

## Run the experiments

```bash
python experiments/e1_drift.py            # Model 1: e1 to e5, one script each
python experiments/m_experiments.py m2    # Model 2: m0, m2, m3, m4, m5, m7
```

Results are saved in `results/`. They are already in the repository, so this step can be skipped.

## Make the figures

```bash
python analysis/make_figures.py
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
