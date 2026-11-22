# E4: environmental shock. Do the genes track the environment, and how fast?
# Runs last 4,000 timesteps. The two shock conditions switch abundance level
# at timestep 2,000 (most plentiful -> scarcest, and the reverse). The two
# constant conditions never switch: they show where the genes would have gone
# without the shock.
#
# Run with: python experiments/e4_shock.py
from common import SEEDS, STEPS, run_batch, save_results
import config

if __name__ == "__main__":
    scarcest, most_plentiful = min(config.ABUNDANCE_MULTIPLIERS), max(config.ABUNDANCE_MULTIPLIERS)
    conditions = [
        ("plenty then scarcity", most_plentiful, scarcest),
        ("scarcity then plenty", scarcest, most_plentiful),
        ("always plenty", most_plentiful, None),
        ("always scarcity", scarcest, None),
    ]
    specs = []
    for condition, start, after in conditions:
        for seed in SEEDS:
            spec = {"experiment": "E4", "condition": condition, "level": start, "seed": seed,
                    "steps": 2 * STEPS}
            if after is not None:
                spec["shock"] = (STEPS, after)
            specs.append(spec)
    print(f"E4: {len(specs)} runs")
    save_results("e4", run_batch(specs))
