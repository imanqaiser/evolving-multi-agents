# E2: the scarcity gradient.
# All five abundance levels, 30 seeds each. Saves the per-timestep data, one
# row per dead agent (used by E5), and every run's final gene pools (used by
# E3 and by the dashboard's "evolved populations" option).
#
# Run with: python experiments/e2_gradient.py
import gzip
import json
import os
import random

from common import GENE_POOLS, RESULTS, SEEDS, run_batch, save_gene_pool, save_results
import config

if __name__ == "__main__":
    specs = []
    for level in config.ABUNDANCE_MULTIPLIERS:
        for seed in SEEDS:
            specs.append({"experiment": "E2", "condition": f"level {level}", "level": level,
                          "seed": seed, "keep_lifetimes": True})
    print(f"E2: {len(specs)} runs")
    results = run_batch(specs)
    save_results("e2", results)

    # Final gene pools of every run: {level: {seed: {"A": [...], "B": [...]}}}.
    pools = {}
    for result in results:
        spec = result["spec"]
        rounded = {}
        for population, genomes in result["pools"].items():
            rounded[population] = [{gene: round(value, 4) for gene, value in genome.items()}
                                   for genome in genomes]
        pools.setdefault(str(spec["level"]), {})[str(spec["seed"])] = rounded
    path = os.path.join(RESULTS, "e2_gene_pools.json.gz")
    with gzip.open(path, "wt") as file:
        json.dump(pools, file)
    print(f"Saved {path}")

    # Two small pools for the dashboard: 300 genomes sampled from all the
    # scarcest-level runs ("scarcity veterans") and all the most plentiful
    # ("plenty veterans").
    sampler = random.Random(0)
    for name, level in [("scarcity_veterans", min(config.ABUNDANCE_MULTIPLIERS)),
                        ("plenty_veterans", max(config.ABUNDANCE_MULTIPLIERS))]:
        everyone = []
        for run in pools[str(level)].values():
            everyone.extend(run["A"] + run["B"])
        save_gene_pool(os.path.join(GENE_POOLS, f"{name}.json"), sampler.sample(everyone, 300))
        print(f"Saved gene_pools/{name}.json")
