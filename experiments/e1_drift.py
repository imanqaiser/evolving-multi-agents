# E1: is it evolution or drift?
# Middle abundance level, 30 seeds with selection (normal runs) and 30 seeds of
# the neutral control (genes inherited, behaviour random). A gene "evolved"
# only if its average moves away from where the neutral runs wander.
#
# Run with: python experiments/e1_drift.py
from common import SEEDS, run_batch, save_results
import config

if __name__ == "__main__":
    specs = []
    for condition, neutral in [("selection", False), ("neutral", True)]:
        for seed in SEEDS:
            specs.append({"experiment": "E1", "condition": condition, "level": config.MIDDLE_LEVEL,
                          "seed": seed, "neutral": neutral})
    print(f"E1: {len(specs)} runs")
    save_results("e1", run_batch(specs))
