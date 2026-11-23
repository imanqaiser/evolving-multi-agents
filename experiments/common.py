# Shared code for the experiment scripts: run many headless simulations on all
# CPU cores and save the results as compressed CSV files in results/.
#
# Each run is described by a "spec": a plain dictionary. The keys are
#   experiment, condition   labels copied into every saved row
#   level                   abundance level 1-5
#   seed                    random seed (the run is fully determined by its spec)
#   steps                   how many timesteps to run (default STEPS)
#   neutral                 True for the neutral control
#   overrides               parameter overrides, see SocietyModel
#   gene_pools              starting gene pools, see SocietyModel
#   frozen_genomes          True to stop evolution, see SocietyModel
#   shock                   (timestep, new level): switch abundance level mid-run
#   keep_lifetimes          True to return one row per dead agent
#   labels                  extra {column: value} pairs to copy into every row
#   version                 1 (default) for Model 1, 2 for Model 2 (level is then 1-7)
#   zones                   True for Model 2's three-zone map (uses the ZONED world)
#   world                   world name, if not the default for the map
import json
import multiprocessing
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from model import SocietyModel  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
GENE_POOLS = os.path.join(ROOT, "gene_pools")

WORLD = "EXPERIMENT"
ZONED_WORLD = "ZONED"
STEPS = 2000         # About 150 generations.
RECORD_EVERY = 10    # Save a data row every 10 timesteps.
SEEDS = list(range(30))  # 30 seeds per condition. Calibration used seeds 9000+.


# Run one simulation and return its data.
def run_one(spec):
    zones = spec.get("zones", False)
    model = SocietyModel(
        spec["level"], spec.get("world", ZONED_WORLD if zones else WORLD), seed=spec["seed"], headless=True, zones=zones,
        neutral=spec.get("neutral", False), overrides=spec.get("overrides"),
        gene_pools=spec.get("gene_pools"), frozen_genomes=spec.get("frozen_genomes", False),
        record_every=RECORD_EVERY, version=spec.get("version", 1),
    )
    shock = spec.get("shock")
    for _ in range(spec.get("steps", STEPS)):
        if shock and model.steps == shock[0]:
            model.set_environment(shock[1])
        model.step()

    labels = {"experiment": spec["experiment"], "condition": spec["condition"],
              "level": spec["level"], "seed": spec["seed"]}
    labels.update(spec.get("labels", {}))

    timeseries = model.datacollector.get_model_vars_dataframe()
    for position, (column, value) in enumerate(labels.items()):
        timeseries.insert(position, column, value)

    lifetimes = None
    if spec.get("keep_lifetimes"):
        lifetimes = pd.DataFrame(model.lifetime_records)
        for position, (column, value) in enumerate(labels.items()):
            lifetimes.insert(position, column, value)

    # "selection_summary": True returns, instead of every lifetime row, how
    # strongly each gene goes with having offspring (see selection_summary()).
    selection = None
    if spec.get("selection_summary"):
        selection = selection_summary(pd.DataFrame(model.lifetime_records), model.genes,
                                      spec.get("steps", STEPS) // 2)
        selection.update(labels)

    pools = {"A": model.living_genomes("A"), "B": model.living_genomes("B")}
    return {"spec": spec, "timeseries": timeseries, "lifetimes": lifetimes, "pools": pools,
            "encounter_rows": list(model.encounter_rows), "selection": selection}


# Selection strength in one run, from the agents born after "born_after" that
# have died: the rank correlation (Spearman) between each gene and the number
# of offspring, and the offspring of the fittest quarter against the least fit.
def selection_summary(lives, genes, born_after):
    out = {"agents": 0}
    if len(lives) == 0:
        return out
    lives = lives[lives["birth_step"] >= born_after]
    out["agents"] = len(lives)
    if len(lives) < 40:
        return out
    ranks = lives[genes + ["offspring"]].rank()
    for gene in genes:
        out[f"rho {gene}"] = ranks[gene].corr(ranks["offspring"])
    low, high = lives["fitness"].quantile([0.25, 0.75])
    out["offspring top quarter"] = lives[lives["fitness"] >= high]["offspring"].mean()
    out["offspring bottom quarter"] = lives[lives["fitness"] <= low]["offspring"].mean()
    out["mean offspring"] = lives["offspring"].mean()
    return out


# Run every spec, using all CPU cores, and return the results in spec order.
def run_batch(specs, processes=None):
    started = time.time()
    results = []
    # PROCESSES=3 python ... limits how many CPU cores a batch uses.
    with multiprocessing.Pool(processes or int(os.environ.get("PROCESSES", os.cpu_count()))) as pool:
        for count, result in enumerate(pool.imap(run_one, specs), start=1):
            results.append(result)
            if count % 10 == 0 or count == len(specs):
                print(f"  {count}/{len(specs)} runs done ({time.time() - started:.0f} s)", flush=True)
    return results


# Save the per-timestep rows (and lifetime rows, if kept) of a batch.
def save_results(name, results, folder=RESULTS):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}_timeseries.csv.gz")
    pd.concat([r["timeseries"] for r in results]).to_csv(path, index=False, float_format="%.5g")
    print(f"Saved {path}")
    selection = [r["selection"] for r in results if r.get("selection")]
    if selection:
        path = os.path.join(folder, f"{name}_selection.csv")
        pd.DataFrame(selection).to_csv(path, index=False, float_format="%.5g")
        print(f"Saved {path}")
    rows = [{"experiment": r["spec"]["experiment"], "condition": r["spec"]["condition"],
             "level": r["spec"]["level"], "seed": r["spec"]["seed"], **r["spec"].get("labels", {}),
             **{f"row {i}": count for i, count in enumerate(r["encounter_rows"])}}
            for r in results if r["spec"].get("zones")]
    if rows:
        path = os.path.join(folder, f"{name}_encounter_rows.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Saved {path}")
    lifetimes = [r["lifetimes"] for r in results if r["lifetimes"] is not None]
    if lifetimes:
        path = os.path.join(folder, f"{name}_lifetimes.csv.gz")
        pd.concat(lifetimes).to_csv(path, index=False, float_format="%.4g")
        print(f"Saved {path}")


# A gene pool is a list of genomes (dictionaries), stored as JSON.
def save_gene_pool(path, genomes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rounded = [{gene: round(value, 4) for gene, value in genome.items()} for genome in genomes]
    with open(path, "w") as file:
        json.dump(rounded, file)


def load_gene_pool(path):
    with open(path) as file:
        return json.load(file)
