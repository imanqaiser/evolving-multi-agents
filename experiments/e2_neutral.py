# Neutral control at the other four abundance levels (E1 already ran it at the
# middle level). Population size differs a lot between levels, and smaller
# populations drift more, so each level's evolved genes are compared with the
# neutral runs of the same level.
#
# Run with: python experiments/e2_neutral.py
from common import SEEDS, run_batch, save_results
import config

if __name__ == "__main__":
    specs = []
    for level in config.ABUNDANCE_MULTIPLIERS:
        if level == config.MIDDLE_LEVEL:
            continue
        for seed in SEEDS:
            specs.append({"experiment": "E2 neutral", "condition": "neutral", "level": level,
                          "seed": seed, "neutral": True})
    print(f"E2 neutral: {len(specs)} runs")
    save_results("e2_neutral", run_batch(specs))
