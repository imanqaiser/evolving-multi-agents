# Stretch experiments (run after E1-E5).
#   1. A sweep of FIGHT_COST, to test the claim in config.py that cheap fights
#      make everyone evolve to fight. Run at level 2, because E2 showed that is
#      the only level where the two populations keep meeting; fights cannot
#      matter where nobody meets.
#   2. Farms disabled, at every level, to see how much farming shapes the result.
# 20 seeds per condition.
#
# Run with: python experiments/stretch.py
from common import run_batch, save_results
import config

STRETCH_SEEDS = list(range(20))
FIGHT_COSTS = [0.0, 0.5, 1.5, 3.0]
CONTACT_LEVEL = 2

if __name__ == "__main__":
    specs = []
    for cost in FIGHT_COSTS:
        for seed in STRETCH_SEEDS:
            specs.append({"experiment": "stretch", "condition": f"fight cost {cost:g}", "level": CONTACT_LEVEL,
                          "seed": seed, "overrides": {"FIGHT_COST": cost}})
    for level in config.ABUNDANCE_MULTIPLIERS:
        for seed in STRETCH_SEEDS:
            specs.append({"experiment": "stretch", "condition": f"no farms level {level}", "level": level,
                          "seed": seed, "overrides": {"FARMS_ENABLED": False}})
    print(f"Stretch: {len(specs)} runs")
    save_results("stretch", run_batch(specs))
