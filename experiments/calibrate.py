# Calibration of the abundance levels. Run once, BEFORE the experiments, on
# seeds the experiments never use (9000+). It sweeps the productivity
# multiplier and prints, for each value, how many agents the world holds and
# how many deaths are from starvation. The five levels in config.py were picked
# from this table and then frozen.
#
# Run with: python experiments/calibrate.py
import sys

import numpy as np

from common import STEPS, run_batch
import config

MULTIPLIERS = [0.5, 0.6, 0.7, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0, 4.0]
CALIBRATION_SEEDS = list(range(9000, 9008))

if __name__ == "__main__":
    multipliers = [float(value) for value in sys.argv[1:]] or MULTIPLIERS
    specs = []
    for multiplier in multipliers:
        for seed in CALIBRATION_SEEDS:
            specs.append({
                "experiment": "calibration", "condition": str(multiplier), "level": 3, "seed": seed,
                "overrides": {"regeneration": config.BASE_REGENERATION * multiplier,
                              "farm_growth": config.BASE_FARM_GROWTH * multiplier},
                "labels": {"multiplier": multiplier},
            })
    results = run_batch(specs)
    print("multiplier  total agents  smaller population  both survive  starvation share  "
          "starved per 1000 agent-steps  generations")
    for multiplier in multipliers:
        totals, smaller, survive, starved, hazard, generations = [], [], [], [], [], []
        for result in results:
            if result["spec"]["labels"]["multiplier"] != multiplier:
                continue
            data = result["timeseries"]
            late = data[data["Timestep"] >= STEPS // 2]
            first, last = late.iloc[0], late.iloc[-1]
            totals.append((late["Population A"] + late["Population B"]).mean())
            smaller.append(min(late["Population A"].mean(), late["Population B"].mean()))
            survive.append(last["Population A"] > 0 and last["Population B"] > 0)
            starvation = last["Deaths from starvation"] - first["Deaths from starvation"]
            old_age = last["Deaths from old age"] - first["Deaths from old age"]
            starved.append(starvation / max(1, starvation + old_age))
            lived = sum(last[f"Population {p} agent steps"] - first[f"Population {p} agent steps"] for p in "AB")
            hazard.append(1000 * starvation / max(1, lived))
            generations.append(last["Average generation"])
        print(f"{multiplier:10.2f}  {np.mean(totals):12.0f}  {np.mean(smaller):18.0f}  "
              f"{np.mean(survive):12.2f}  {np.mean(starved):16.2f}  {np.mean(hazard):28.2f}  {np.mean(generations):11.0f}")
