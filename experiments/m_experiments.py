# The Model 2 experiments. Model 2 = body genes + explicit fitness + soil that
# wears out + trade bonus (version=2), on the three-zone map (zones=True):
# two homelands with farms only and shared land with wild food in between.
# Scarcity is homeland productivity: level 1 (SCARCE) to level 7 (PLENTIFUL).
#
# Run one experiment with: python experiments/m_experiments.py NAME
#   m2   the scarcity gradient: 7 levels, selection and neutral control.
#        (M1, "evolution or drift", uses m2's runs at levels 1, 4 and 7.)
#   m7   unequal homelands: one population rich, the other poor, in one world.
#   m0   competitive exclusion on an OPEN map (no zones): selection vs neutral.
#   m4   environmental shock: plentiful to scarce and the reverse.
#   m5   trade bonus: when does cooperation survive?
#   m3   veterans of plenty against veterans of scarcity on an open map (needs m2).
#   m8   Raiders vs Farmers: a hand-set (not evolved) starting point, see config.py.
#   pools  writes gene_pools/raiders.json and farmers.json for m8 and the dashboard.
# Results go to results/model2/. PROCESSES=3 limits the CPU cores used.
import gzip
import json
import multiprocessing
import os
import pickle
import random
import sys
import time

from common import GENE_POOLS, RESULTS, load_gene_pool, run_batch, run_one, save_gene_pool, save_results
import config

FOLDER = os.path.join(RESULTS, "model2")
SEEDS = list(range(20))   # 20 seeds per condition. Calibration used seeds 9000+.
STEPS = 1500              # About 200 generations in Model 2.
NAMED = config.M2_NAMED_LEVELS   # {"SCARCE": 1, "MIDDLE": 4, "PLENTIFUL": 7}
SCARCE, MIDDLE, PLENTIFUL = NAMED["SCARCE"], NAMED["MIDDLE"], NAMED["PLENTIFUL"]


# The settings every zoned Model 2 run shares.
def zoned(experiment, condition, level, seed, **extra):
    spec = {"experiment": experiment, "condition": condition, "level": level, "seed": seed,
            "version": 2, "zones": True, "steps": STEPS, "selection_summary": True}
    spec.update(extra)
    return spec


# An open-map Model 2 run: wild food everywhere, no zones, 50 by 50 world.
def open_map(experiment, condition, level, seed, **extra):
    spec = {"experiment": experiment, "condition": condition, "level": level, "seed": seed,
            "version": 2, "steps": STEPS}
    spec.update(extra)
    return spec


def round_pool(genomes):
    return [{gene: round(value, 4) for gene, value in genome.items()} for genome in genomes]


# The Raiders vs Farmers starting pools: the hand-set genome from config.py
# plus a little noise on every gene, always from the same seed.
def make_scenario_pools():
    sampler = random.Random(config.SCENARIO_POOL_SEED)
    for name, base in [("raiders", config.RAIDER_GENOME), ("farmers", config.FARMER_GENOME)]:
        pool = []
        for _ in range(config.SCENARIO_POOL_SIZE):
            pool.append({gene: min(1.0, max(0.0, value + sampler.gauss(0, config.SCENARIO_NOISE)))
                         for gene, value in base.items()})
        path = os.path.join(GENE_POOLS, f"{name}.json")
        save_gene_pool(path, pool)
        print(f"Saved {path}")
    # For the dashboard's "evolved" option: a sample of the agents alive at the
    # end of m2's runs at the scarcest and the most plentiful level.
    with gzip.open(os.path.join(FOLDER, "m2_gene_pools.json.gz"), "rt") as file:
        evolved = json.load(file)
    for name, level in [("m2_scarcity_veterans", SCARCE), ("m2_plenty_veterans", PLENTIFUL)]:
        genomes = [genome for run in evolved[str(level)].values() for side in run.values() for genome in side]
        path = os.path.join(GENE_POOLS, f"{name}.json")
        save_gene_pool(path, sampler.sample(genomes, config.SCENARIO_POOL_SIZE))
        print(f"Saved {path}")


def run_one_indexed(item):
    index, spec = item
    return index, run_one(spec)


def run_resumable(name, specs, limit):
    parts = os.path.join(FOLDER, "parts", name)
    os.makedirs(parts, exist_ok=True)
    started = time.time()
    todo = [(i, spec) for i, spec in enumerate(specs) if not os.path.exists(os.path.join(parts, f"{i}.pkl"))]
    if todo:
        pool = multiprocessing.Pool(int(os.environ.get("PROCESSES", os.cpu_count())))
        for index, result in pool.imap_unordered(run_one_indexed, todo):
            with open(os.path.join(parts, f"{index}.tmp"), "wb") as file:
                pickle.dump(result, file)
            os.replace(os.path.join(parts, f"{index}.tmp"), os.path.join(parts, f"{index}.pkl"))
            if time.time() - started > limit:
                break
        pool.terminate()
    done = sum(os.path.exists(os.path.join(parts, f"{i}.pkl")) for i in range(len(specs)))
    print(f"{name}: {done}/{len(specs)} runs done", flush=True)
    if done < len(specs):
        return None
    results = []
    for i in range(len(specs)):
        with open(os.path.join(parts, f"{i}.pkl"), "rb") as file:
            results.append(pickle.load(file))
    return results


if __name__ == "__main__":
    name = sys.argv[1]
    if name == "pools":
        make_scenario_pools()
        raise SystemExit(0)
    specs = []
    if name == "m2":
        for level in config.M2_MULTIPLIERS:
            for seed in SEEDS:
                specs.append(zoned("M2", "selection", level, seed))
                specs.append(zoned("M2", "neutral", level, seed, neutral=True))
    elif name == "m7":
        # Even seeds make A the rich side, odd seeds B, so position cannot explain the result.
        for seed in SEEDS:
            rich = "A" if seed % 2 == 0 else "B"
            poor = "B" if rich == "A" else "A"
            levels = {rich: PLENTIFUL, poor: SCARCE}
            for condition, neutral in [("selection", False), ("neutral", True)]:
                specs.append(zoned("M7", condition, MIDDLE, seed, neutral=neutral,
                                   overrides={"HOMELAND_LEVELS": levels}, labels={"rich_side": rich}))
    elif name == "m0":
        for level in NAMED.values():
            for seed in SEEDS:
                specs.append(open_map("M0", "selection", level, seed))
                specs.append(open_map("M0", "neutral", level, seed, neutral=True))
    elif name == "m4":
        # 2,400 steps; the level switches at step 1,200. Two controls never switch.
        for seed in SEEDS[:15]:
            for condition, start, shock in [("plenty then scarcity", PLENTIFUL, SCARCE),
                                            ("scarcity then plenty", SCARCE, PLENTIFUL),
                                            ("always plenty", PLENTIFUL, None),
                                            ("always scarcity", SCARCE, None)]:
                extra = {"steps": 2400, "selection_summary": False}
                if shock is not None:
                    extra["shock"] = (1200, shock)
                specs.append(zoned("M4", condition, start, seed, **extra))
    elif name == "m5":
        # The default bonus (0.5) is already in m2, so only the other values run here.
        for bonus in [0.0, 1.0, 2.0]:
            for level in NAMED.values():
                for seed in SEEDS[:12]:
                    specs.append(zoned("M5", f"bonus {bonus}", level, seed, selection_summary=False,
                                       overrides={"TRADE_BONUS": bonus}, labels={"bonus": bonus}))
    elif name == "m3":
        # Veterans come from m2: for seed s, the "plenty veterans" are a final
        # population of m2's PLENTIFUL run with seed s, the "scarcity veterans"
        # one of its SCARCE run. They meet on an open map at three levels.
        with gzip.open(os.path.join(FOLDER, "m2_gene_pools.json.gz"), "rt") as file:
            pools = json.load(file)

        def veterans(level, seed):
            run = pools[str(level)][str(seed)]
            return run["A"] if len(run["A"]) >= len(run["B"]) else run["B"]

        for level in NAMED.values():
            for mode in ["evolving", "frozen"]:
                for seed in SEEDS:
                    plenty_side = "A" if seed % 2 == 0 else "B"
                    scarcity_side = "B" if plenty_side == "A" else "A"
                    specs.append(open_map(
                        "M3", mode, level, seed, steps=1000, frozen_genomes=mode == "frozen",
                        gene_pools={plenty_side: veterans(PLENTIFUL, seed), scarcity_side: veterans(SCARCE, seed)},
                        labels={"plenty_side": plenty_side}))
    elif name == "m8":
        # Raiders vs Farmers (hand-set genomes, not evolved). The sides alternate
        # as in m7: even seeds make A the rich side. A fourth condition, "random"
        # (random genes, same unequal homelands), is m7's "selection" runs: the
        # settings are identical, so the figures read them from m7's results.
        raiders = load_gene_pool(os.path.join(GENE_POOLS, "raiders.json"))
        farmers = load_gene_pool(os.path.join(GENE_POOLS, "farmers.json"))
        for seed in SEEDS:
            rich = "A" if seed % 2 == 0 else "B"
            poor = "B" if rich == "A" else "A"
            levels = {rich: config.FARMER_HOMELAND_LEVEL, poor: config.RAIDER_HOMELAND_LEVEL}
            #   evolving  raiders poor, farmers rich, genes inherit and mutate
            #   frozen    the same, but children copy the starting genomes
            #   swapped   raiders get the rich homeland, farmers the poor one
            for condition, raider_side in [("evolving", poor), ("frozen", poor), ("swapped", rich)]:
                farmer_side = "B" if raider_side == "A" else "A"
                specs.append(zoned("M8", condition, MIDDLE, seed, frozen_genomes=condition == "frozen",
                                   gene_pools={raider_side: raiders, farmer_side: farmers},
                                   overrides={"HOMELAND_LEVELS": levels},
                                   labels={"rich_side": rich, "raider_side": raider_side}))
    else:
        raise SystemExit(f"Unknown experiment: {name}")

    print(f"{name}: {len(specs)} runs", flush=True)
    # TIME_LIMIT=150 (seconds) runs as many of the remaining runs as fit, keeps
    # each finished run in results/model2/parts/, and stops. Repeat the same
    # command until it reports that every run is done; then it saves as usual.
    limit = float(os.environ.get("TIME_LIMIT", 0))
    if limit:
        results = run_resumable(name, specs, limit)
        if results is None:
            raise SystemExit(0)
    else:
        results = run_batch(specs)
    for result in results:
        result["spec"].pop("gene_pools", None)   # Not needed any more, and large.
    save_results(name, results, FOLDER)

    if name == "m2":
        # Final gene pools of the selection runs at the three named levels:
        # {level: {seed: {"A": [...], "B": [...]}}}. Used by m3, the trait-space
        # figure and the dashboard's "evolved populations" option.
        pools = {}
        for result in results:
            spec = result["spec"]
            if spec["condition"] == "selection" and spec["level"] in NAMED.values():
                pools.setdefault(str(spec["level"]), {})[str(spec["seed"])] = {
                    population: round_pool(genomes) for population, genomes in result["pools"].items()}
        path = os.path.join(FOLDER, "m2_gene_pools.json.gz")
        with gzip.open(path, "wt") as file:
            json.dump(pools, file)
        print(f"Saved {path}")
