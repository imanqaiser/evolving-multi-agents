# E3: the question from the proposal slides. A population that evolved in
# plenty and a population that evolved in scarcity share one world. What happens?
#
# Needs E2's saved gene pools. For seed s, the "plenty veterans" are founded
# from the final gene pool of E2's most plentiful run with seed s, and the
# "scarcity veterans" from E2's scarcest run with seed s, so the 30 seeds are
# 30 independently evolved pairs. The pair meets at each of the five levels.
#   - Even seeds put the plenty veterans in the top half (population A), odd
#     seeds in the bottom half (population B), so position cannot explain who wins.
#   - "evolving" runs keep mutation and inheritance as normal.
#   - "frozen" runs copy every child's genome from its side's founders, so the
#     outcome reflects only what each side brought with it.
#
# Run with: python experiments/e3_shared_world.py
import gzip
import json
import os

from common import RESULTS, SEEDS, run_batch, save_results
import config


# One evolved population from an E2 run: A, or B if A died out.
def veteran_pool(pools, level, seed):
    run = pools[str(level)][str(seed)]
    return run["A"] if run["A"] else run["B"]


if __name__ == "__main__":
    with gzip.open(os.path.join(RESULTS, "e2_gene_pools.json.gz"), "rt") as file:
        pools = json.load(file)
    scarcest, most_plentiful = min(config.ABUNDANCE_MULTIPLIERS), max(config.ABUNDANCE_MULTIPLIERS)
    specs = []
    for level in config.ABUNDANCE_MULTIPLIERS:
        for mode in ["evolving", "frozen"]:
            for seed in SEEDS:
                plenty_side = "A" if seed % 2 == 0 else "B"
                scarcity_side = "B" if plenty_side == "A" else "A"
                specs.append({
                    "experiment": "E3", "condition": mode, "level": level, "seed": seed,
                    "frozen_genomes": mode == "frozen",
                    "gene_pools": {plenty_side: veteran_pool(pools, most_plentiful, seed),
                                   scarcity_side: veteran_pool(pools, scarcest, seed)},
                    "labels": {"plenty_side": plenty_side},
                })
    print(f"E3: {len(specs)} runs")
    save_results("e3", run_batch(specs))
