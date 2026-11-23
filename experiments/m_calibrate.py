# Model 2 calibration pilots. Run BEFORE the Model 2 experiments, on seeds the
# experiments never use (9000+). Every table printed here is also appended to
# results/model2/calibration.txt. After that the values in config.MODEL_2 and
# config.M2_MULTIPLIERS are frozen.
#
# Each pilot checks that a MECHANISM works (a gene is not stuck at 0 or 1
# everywhere, agents keep moving, the populations keep meeting, selection is
# strong enough to see). None of them looks at which way a result points.
#
# Run with: python experiments/m_calibrate.py STAGE
#   costs      vision cost × body cost × soil recovery, at five multipliers
#   stakes     reproduction threshold / cost / cooldown
#   scale      sweep of the abundance multiplier
#   gate       Gate 1: the three named levels with the frozen settings
import itertools
import os
import sys

import numpy as np

from common import RESULTS, STEPS, run_batch
import config

FOLDER = os.path.join(RESULTS, "model2")
ZONES = os.environ.get("ZONES") == "1"   # ZONES=1 python m_calibrate.py ... pilots the three-zone map.
GENES = config.GENES + config.BODY_GENE_NAMES + (config.ZONE_GENE_NAMES if ZONES else [])
SHORT = {"exploratory_tendency": "expl", "risk_tolerance": "risk", "sociability": "soc",
         "cooperative_tendency": "coop", "farming_tendency": "farm", "vision": "vis", "body_size": "body", "expansion": "expa"}


# The settings of one pilot run at a given abundance multiplier.
def pilot_spec(label, multiplier, seed, overrides=None, neutral=False, steps=STEPS):
    settings = {"regeneration": config.BASE_REGENERATION * multiplier,
                "farm_growth": config.BASE_FARM_GROWTH * multiplier}
    settings.update(overrides or {})
    spec = {"experiment": "m_calibration", "condition": label, "level": 4, "seed": seed, "version": 2,
            "overrides": settings, "neutral": neutral, "keep_lifetimes": True, "steps": steps,
            "labels": {"multiplier": multiplier}}
    if ZONES:
        # Three-zone map: the multiplier scales homeland farm growth only.
        spec["zones"] = True
        del settings["regeneration"]
    return spec


# The numbers every pilot reports for one finished run (second half of the run).
def measure(result):
    data = result["timeseries"]
    steps = data["Timestep"].max()
    late = data[data["Timestep"] >= steps // 2]
    first, last = late.iloc[0], late.iloc[-1]

    def change(column):
        return last[column] - first[column]

    lived = change("Population A agent steps") + change("Population B agent steps")
    out = {
        "agents": (late["Population A"] + late["Population B"]).mean(),
        "smaller": min(late["Population A"].mean(), late["Population B"].mean()),
        "both": float(last["Population A"] > 0 and last["Population B"] > 0),
        "enc": change("Cumulative encounters"),
        "enc/100": 100 * change("Cumulative encounters") / max(1, lived),
        "move": (change("Population A moves") + change("Population B moves")) / max(1, lived),
        "soil": late["Average fertility"].mean(),
        "gens": last["Average generation"],
        "starve": change("Deaths from starvation") / max(1, change("Deaths from starvation")
                                                         + change("Deaths from old age")),
    }
    if ZONES:
        food = {s: sum(change(f"Population {p} food {s}") for p in "AB") for s in ["farm", "wild", "raid"]}
        where = {s: sum(change(f"Population {p} steps {s}") for p in "AB") for s in ["home", "shared", "rival"]}
        out["farmfood"] = food["farm"] / max(1e-9, sum(food.values()))
        out["raidfood"] = food["raid"] / max(1e-9, sum(food.values()))
        out["rivaltime"] = where["rival"] / max(1, sum(where.values()))
        out["homeenc"] = (change("Encounters in homeland A") + change("Encounters in homeland B")) / max(
            1, change("Cumulative encounters"))
    total = last["Population A"] + last["Population B"]
    for gene in GENES:
        out[SHORT[gene]] = sum(last[f"Population {p}"] * last[f"Population {p} average {gene}"]
                               for p in "AB") / max(1, total)
    # Selection strength: offspring of the top quarter of fitness against the
    # bottom quarter, among agents born in the second half that have died.
    lives = result["lifetimes"]
    out["top/bottom"] = np.nan
    out["max|rho|"] = np.nan
    if lives is not None and len(lives):
        lives = lives[lives["birth_step"] >= steps // 2]
        if len(lives) >= 40:
            low, high = lives["fitness"].quantile([0.25, 0.75])
            bottom = lives[lives["fitness"] <= low]["offspring"].mean()
            top = lives[lives["fitness"] >= high]["offspring"].mean()
            out["top"] = top
            out["bottom"] = bottom
            out["top/bottom"] = top / bottom if bottom > 0 else np.inf
            ranks = lives[GENES + ["offspring"]].rank()
            out["max|rho|"] = ranks[GENES].corrwith(ranks["offspring"]).abs().max()
    return out


# Run the specs, average the measurements over seeds for each (condition,
# multiplier), print the table and append it to calibration.txt.
def report(title, specs, columns):
    results = run_batch(specs)
    groups = {}
    for result in results:
        key = (result["spec"]["condition"], result["spec"]["labels"]["multiplier"])
        groups.setdefault(key, []).append(measure(result))
    lines = [f"== {title} ==  (mean of {len(specs) // len(groups)} calibration seeds per row, "
             f"second half of a {specs[0]['steps']}-step run)",
             f"{'condition':<34}{'mult':>6}" + "".join(f"{column:>11}" for column in columns)]
    for (condition, multiplier), rows in groups.items():
        line = f"{condition:<34}{multiplier:>6.2f}"
        for column in columns:
            values = [row.get(column, np.nan) for row in rows]
            line += f"{np.nanmean(values) if not np.all(np.isnan(values)) else np.nan:>11.3g}"
        lines.append(line)
    text = "\n".join(lines) + "\n"
    print(text)
    os.makedirs(FOLDER, exist_ok=True)
    with open(os.path.join(FOLDER, "calibration.txt"), "a") as file:
        file.write(text + "\n")
    return groups


BASIC = ["agents", "smaller", "both", "enc", "enc/100", "move", "soil", "gens", "starve"]
if ZONES:
    BASIC += ["farmfood", "raidfood", "rivaltime", "homeenc"]
GENE_COLUMNS = [SHORT[gene] for gene in GENES]
SELECTION = ["top/bottom", "max|rho|"]

if __name__ == "__main__":
    stage = sys.argv[1]
    seeds = list(range(9000, 9000 + int(sys.argv[2]) if len(sys.argv) > 2 else 9003))
    specs = []
    if stage == "costs":
        # Body-gene costs and soil recovery. Looking for: vision and body size
        # not at the same extreme at every multiplier; agents moving; soil in use.
        for vision_cost, body_cost, recovery in itertools.product([0.01, 0.02, 0.04], [0.1, 0.2], [0.01, 0.02]):
            label = f"vis {vision_cost} body {body_cost} rec {recovery}"
            for multiplier in [0.5, 1.0, 2.0, 4.0, 8.0]:
                for seed in seeds:
                    specs.append(pilot_spec(label, multiplier, seed, {
                        "VISION_COST": vision_cost, "BODY_COST": body_cost, "SOIL_RECOVERY": recovery}))
        report("Body costs and soil recovery", specs, ["agents", "smaller", "both", "enc/100", "move", "soil",
                                                         "vis", "body", "farm", "gens"])
    elif stage == "custom":
        # python m_calibrate.py custom SEEDS "NAME=value,..." mult mult ...   (NAME neutral=1 for the control)
        settings = {}
        neutral = False
        for item in sys.argv[3].split(","):
            if not item:
                continue
            name, value = item.split("=")
            if name == "neutral":
                neutral = True
            elif name == "SELECTION_MODE":
                settings[name] = value
            else:
                settings[name] = float(value) if "." in value else int(value)
            if name == "HOMELAND_LEVELS":   # written as HOMELAND_LEVELS=71 for A at level 7, B at level 1
                settings[name] = {"A": int(value[0]), "B": int(value[1])}
        for multiplier in [float(value) for value in sys.argv[4:]]:
            for seed in seeds:
                specs.append(pilot_spec(sys.argv[3] or "frozen settings", multiplier, seed, settings, neutral))
        report("Custom pilot", specs, BASIC + GENE_COLUMNS + SELECTION)
    elif stage == "scale":
        for multiplier in [float(value) for value in sys.argv[3:]]:
            for seed in seeds:
                specs.append(pilot_spec("frozen settings", multiplier, seed))
                specs.append(pilot_spec("frozen settings, neutral", multiplier, seed, neutral=True))
        report("Abundance multiplier sweep", specs, BASIC + SELECTION)
    elif stage == "gate":
        for name, level in config.M2_NAMED_LEVELS.items():
            for seed in seeds:
                for neutral in [False, True]:
                    spec = pilot_spec(f"{name} (level {level})" + (", neutral" if neutral else ""),
                                      config.M2_MULTIPLIERS[level], seed, neutral=neutral)
                    specs.append(spec)
        report("Gate 1", specs, BASIC + ["top", "bottom"] + SELECTION)
