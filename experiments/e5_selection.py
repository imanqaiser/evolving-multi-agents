# E5: what makes a trait advantageous?
# Uses the lifetime records saved by E2 (one row per dead agent). Offspring
# count is an agent's realized fitness. For every run and every gene this
# script measures how the gene relates to offspring count:
#   - Spearman rank correlation between the gene and offspring count;
#   - a multiple regression of offspring count on all five genes at once
#     (genes standardized, so a coefficient is "extra offspring per one
#     standard deviation of the gene, holding the other genes fixed").
# Both are computed per run and then averaged across runs in the analysis, so
# the unit of replication is the run (30 per level), not the agent.
# Only agents born into the population are used (the founders start with
# extra resources), and the two populations of a run are pooled.
# It also saves mean offspring by gene value (five bins) for plotting.
#
# Run with: python experiments/e5_selection.py     (after e2_gradient.py)
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import RESULTS
import config

BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001]

if __name__ == "__main__":
    lifetimes = pd.read_csv(os.path.join(RESULTS, "e2_lifetimes.csv.gz"))
    lifetimes = lifetimes[lifetimes["generation"] > 0]
    rows = []
    bin_rows = []
    for (level, seed), run in lifetimes.groupby(["level", "seed"]):
        offspring = run["offspring"].to_numpy(dtype=float)
        genes = run[config.GENES].to_numpy(dtype=float)
        standardized = (genes - genes.mean(axis=0)) / genes.std(axis=0)
        design = np.column_stack([np.ones(len(run)), standardized])
        coefficients = np.linalg.lstsq(design, offspring, rcond=None)[0][1:]
        for index, gene in enumerate(config.GENES):
            rows.append({
                "level": level, "seed": seed, "gene": gene, "agents": len(run),
                "mean_offspring": offspring.mean(),
                "gene_std": genes[:, index].std(),
                "spearman": spearmanr(genes[:, index], offspring)[0],
                "regression": coefficients[index],
            })
            binned = pd.cut(run[gene], BINS, right=False, labels=False)
            for bin_index, group in run.groupby(binned):
                bin_rows.append({
                    "level": level, "seed": seed, "gene": gene,
                    "bin_middle": BINS[int(bin_index)] + 0.1, "agents": len(group),
                    "mean_offspring": group["offspring"].mean(),
                    "relative_offspring": group["offspring"].mean() / offspring.mean(),
                })
    path = os.path.join(RESULTS, "e5_selection.csv")
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.5g")
    print(f"Saved {path}")
    path = os.path.join(RESULTS, "e5_offspring_by_gene.csv")
    pd.DataFrame(bin_rows).to_csv(path, index=False, float_format="%.5g")
    print(f"Saved {path}")
