# Regenerates every figure in figures/ and every table in results/tables/
# from the saved experiment results.
#
# Run with: python analysis/make_figures.py
#
# Conventions used everywhere:
#   - The unit of replication is the RUN (one seed). Every mean and every 95%
#     confidence interval is taken across runs, using the t distribution.
#   - Behaviour is always a rate (per encounter, per opportunity, per agent),
#     never a raw count, because scarce worlds hold fewer agents.
#   - "Final" means the average over the last FINAL_WINDOW timesteps of a run.
#   - Population A is blue and population B is orange, as in the dashboard.
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
TABLES = os.path.join(RESULTS, "tables")
FIGURES = os.path.join(ROOT, "figures")

GENES = config.GENES
GENE_LABELS = {
    "exploratory_tendency": "Exploration",
    "risk_tolerance": "Risk tolerance",
    "sociability": "Sociability",
    "cooperative_tendency": "Cooperation",
    "farming_tendency": "Farming",
}
LEVELS = sorted(config.ABUNDANCE_MULTIPLIERS)
LEVEL_LABELS = [f"{level}\n(×{config.ABUNDANCE_MULTIPLIERS[level]:g})" for level in LEVELS]
COLORS = {"A": "#1556b8", "B": "#c2560c"}      # Same as static/dashboard.js.
NEUTRAL = "#6b7280"                             # Neutral control: grey.
PLENTY, SCARCITY = "#2a7f3f", "#7b3fa0"         # Veterans of plenty / of scarcity.
FINAL_WINDOW = 200
# A rate (e.g. "share of meetings that were cooperative") is only computed for
# a run with at least MIN_EVENTS such events, and only plotted for a level
# where at least MIN_RUNS runs have a value. Otherwise it would rest on noise.
MIN_EVENTS = 20
MIN_RUNS = 10
STEPS = 2000

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14, "legend.fontsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e5e7eb",
    "grid.linewidth": 0.8, "lines.linewidth": 2.2, "figure.dpi": 100, "savefig.dpi": 300,
    "legend.frameon": False, "axes.titleweight": "bold",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(name):
    data = pd.read_csv(os.path.join(RESULTS, name))
    if "Population A" not in data:
        return data
    # The model stores 0 as the gene average of an extinct population. Replace
    # it with "no value", then add the gene average over both populations.
    size_a, size_b = data["Population A"], data["Population B"]
    for gene in GENES:
        column_a, column_b = f"Population A average {gene}", f"Population B average {gene}"
        data.loc[size_a == 0, column_a] = np.nan
        data.loc[size_b == 0, column_b] = np.nan
        total = data[column_a].fillna(0) * size_a + data[column_b].fillna(0) * size_b
        data[f"gene {gene}"] = total / (size_a + size_b).replace(0, np.nan)
    data["Total population"] = size_a + size_b
    return data


# Mean and half-width of the 95% confidence interval across runs.
def mean_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return (values.mean() if len(values) else np.nan), np.nan, len(values)
    half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return values.mean(), half, len(values)


def text_ci(values, digits=3):
    mean, half, count = mean_ci(values)
    return f"{mean:.{digits}f} ± {half:.{digits}f} (n={count})"


# Cohen's d: difference between two groups in pooled standard deviations.
def cohens_d(first, second):
    first, second = np.asarray(first, float), np.asarray(second, float)
    first, second = first[~np.isnan(first)], second[~np.isnan(second)]
    pooled = np.sqrt(((len(first) - 1) * first.var(ddof=1) + (len(second) - 1) * second.var(ddof=1))
                     / (len(first) + len(second) - 2))
    return (first.mean() - second.mean()) / pooled


# Draw the across-run mean of a column over time, with its 95% band.
def draw_band(axis, data, column, color, label, style="-"):
    grouped = data.groupby("Timestep")[column]
    mean, count = grouped.mean(), grouped.count()
    half = stats.t.ppf(0.975, (count - 1).clip(lower=1)) * grouped.std() / np.sqrt(count)
    axis.plot(mean.index, mean.values, color=color, linestyle=style, label=label)
    axis.fill_between(mean.index, (mean - half).values, (mean + half).values, color=color,
                      alpha=0.18, linewidth=0)


# Draw mean ± CI at each abundance level. "values" maps level -> per-run values.
def draw_levels(axis, values, color, label, offset=0.0, marker="o", min_runs=2):
    means, halves = [], []
    for level in LEVELS:
        mean, half, count = mean_ci(values[level])
        if count < min_runs:
            mean, half = np.nan, np.nan
        means.append(mean)
        halves.append(half)
    positions = np.array(LEVELS) + offset
    axis.errorbar(positions, means, yerr=halves, color=color, marker=marker, markersize=8,
                  capsize=4, label=label, markeredgecolor="white", markeredgewidth=1)


def level_axis(axis, label=True):
    axis.set_xticks(LEVELS)
    axis.set_xticklabels(LEVEL_LABELS)
    axis.set_xlim(0.6, 5.4)
    if label:
        axis.set_xlabel("Abundance level (scarce → plentiful)")


# The average of each run's last FINAL_WINDOW rows: one row per run.
def final_rows(data, keys, steps=STEPS):
    late = data[data["Timestep"] > steps - FINAL_WINDOW]
    return late.groupby(keys, as_index=False).mean(numeric_only=True)


def save(figure, name):
    os.makedirs(FIGURES, exist_ok=True)
    figure.savefig(os.path.join(FIGURES, name), bbox_inches="tight")
    plt.close(figure)
    print(f"Saved figures/{name}")


def save_table(table, name):
    os.makedirs(TABLES, exist_ok=True)
    table.to_csv(os.path.join(TABLES, name), index=False, float_format="%.4g")
    print(f"\n--- {name} ---")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


# Hide the unused sixth panel of a 2 by 3 grid and put the legend there.
def legend_panel(axis, source_axis, title=None):
    axis.axis("off")
    handles, labels = source_axis.get_legend_handles_labels()
    axis.legend(handles, labels, loc="center left", title=title, fontsize=14, title_fontsize=14)


# ---------------------------------------------------------------------------
# E1: selection versus the neutral control
# ---------------------------------------------------------------------------

def figure_e1():
    data = load("e1_timeseries.csv.gz")
    selection = data[data["condition"] == "selection"]
    neutral = data[data["condition"] == "neutral"]

    figure, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    axes = axes.ravel()
    for axis, gene in zip(axes, GENES):
        draw_band(axis, neutral, f"gene {gene}", NEUTRAL, "Neutral control (no selection)", "--")
        for population in ["A", "B"]:
            draw_band(axis, selection, f"Population {population} average {gene}", COLORS[population],
                      f"Population {population}, with selection")
        axis.set_title(GENE_LABELS[gene])
        axis.set_ylim(0, 1)
    for axis in axes[3:5]:
        axis.set_xlabel("Timestep")
    axes[2].set_xlabel("Timestep")
    axes[2].tick_params(labelbottom=True)
    for axis in [axes[0], axes[3]]:
        axis.set_ylabel("Average gene value")
    legend_panel(axes[5], axes[0])
    figure.suptitle("E1. Evolution or drift? Selection runs against the neutral control\n"
                    "(middle abundance level, mean and 95% band across 30 runs)", fontsize=18)
    figure.tight_layout()
    save(figure, "e1_selection_vs_neutral.png")

    final = final_rows(data, ["condition", "seed"])
    rows = []
    for gene in GENES:
        chosen = final[final["condition"] == "selection"][f"gene {gene}"]
        drift = final[final["condition"] == "neutral"][f"gene {gene}"]
        difference = chosen.mean() - drift.mean()
        test = stats.ttest_ind(chosen, drift, equal_var=False)
        interval = test.confidence_interval(0.95)
        rows.append({
            "gene": gene, "selection_mean": chosen.mean(), "selection_ci": mean_ci(chosen)[1],
            "neutral_mean": drift.mean(), "neutral_ci": mean_ci(drift)[1],
            "difference": difference, "difference_low": interval.low, "difference_high": interval.high,
            "cohens_d": cohens_d(chosen, drift), "p_value": test.pvalue,
            "n_selection": len(chosen), "n_neutral": len(drift),
        })
    save_table(pd.DataFrame(rows), "e1_final_genes.csv")
    sizes = final.groupby("condition")["Total population"].agg(["mean", "std", "count"]).reset_index()
    save_table(sizes, "e1_population_size.csv")


# ---------------------------------------------------------------------------
# E2: the scarcity gradient
# ---------------------------------------------------------------------------

# Rates over the second half of each run, from the running totals.
def behaviour_rates(data, start=STEPS // 2, end=STEPS):
    rows = []
    for (level, seed), run in data.groupby(["level", "seed"]):
        run = run.set_index("Timestep")
        first, last = run.loc[start], run.loc[end]
        window = run.loc[start + 1:end]
        encounters = last["Cumulative encounters"] - first["Cumulative encounters"]
        fights = last["Cumulative fights"] - first["Cumulative fights"]
        lived_total = 0
        for population in ["A", "B"]:
            prefix = f"Population {population}"

            def gained(column):
                return last[f"{prefix} {column}"] - first[f"{prefix} {column}"]

            lived = gained("agent steps")
            lived_total += lived
            meetings = gained("cumulative cooperate") + gained("cumulative compete")
            chances = gained("cumulative raid") + gained("cumulative respect")
            sightings = gained("cumulative avoid") + gained("cumulative ignore")
            deaths = gained("starvation deaths") + gained("old age deaths")
            alive = window[window[prefix] > 0]
            rows.append({
                "level": level, "seed": seed, "population": population,
                "population_size": window[prefix].mean(),
                "extinct": float(last[prefix] == 0),
                "cooperate_rate": gained("cumulative cooperate") / meetings if meetings >= MIN_EVENTS else np.nan,
                "raid_rate": gained("cumulative raid") / chances if chances >= MIN_EVENTS else np.nan,
                "avoid_rate": gained("cumulative avoid") / sightings if sightings >= MIN_EVENTS else np.nan,
                "farms_per_agent": (alive[f"{prefix} farms"] / alive[prefix]).mean() if len(alive) else np.nan,
                "stolen_per_agent": 100 * gained("cumulative stolen") / lived if lived else np.nan,
                "starvation_share": gained("starvation deaths") / deaths if deaths else np.nan,
                "starved_per_1000": 1000 * gained("starvation deaths") / lived if lived else np.nan,
                "meetings_per_agent": 100 * meetings / lived if lived else np.nan,
            })
        for row in rows[-2:]:
            row["fights_per_encounter"] = fights / encounters if encounters >= MIN_EVENTS else np.nan
            row["encounters"] = encounters
            row["encounters_per_agent"] = 100 * encounters / lived_total if lived_total else np.nan
    return pd.DataFrame(rows)


def figure_e2():
    data = load("e2_timeseries.csv.gz")
    final = final_rows(data, ["level", "seed"])
    # Neutral control runs: level 3 from E1, the other levels from e2_neutral.py.
    neutral = load("e1_timeseries.csv.gz")
    neutral = neutral[neutral["condition"] == "neutral"]
    if os.path.exists(os.path.join(RESULTS, "e2_neutral_timeseries.csv.gz")):
        neutral = pd.concat([neutral, load("e2_neutral_timeseries.csv.gz")])
    neutral = final_rows(neutral, ["level", "seed"])

    # Figure: final genes by level.
    figure, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    axes = axes.ravel()
    for axis, gene in zip(axes, GENES):
        axis.axhline(0.5, color=NEUTRAL, linewidth=1, linestyle=":")
        draw_levels(axis, {level: neutral[neutral["level"] == level][f"gene {gene}"] for level in LEVELS},
                    NEUTRAL, "Neutral control (drift only)", 0.2, marker="D")
        for population, offset in [("A", -0.07), ("B", 0.07)]:
            values = {level: final[final["level"] == level][f"Population {population} average {gene}"]
                      for level in LEVELS}
            draw_levels(axis, values, COLORS[population], f"Population {population}", offset)
        axis.set_title(GENE_LABELS[gene])
        axis.set_ylim(0, 1)
        level_axis(axis)
    for axis in [axes[0], axes[3]]:
        axis.set_ylabel(f"Final average gene value")
    legend_panel(axes[5], axes[0])
    figure.suptitle("E2. Evolved genes at each abundance level\n"
                    "(average of the last 200 timesteps, mean and 95% CI across 30 runs)", fontsize=18)
    figure.tight_layout()
    save(figure, "e2_genes_by_level.png")

    rows = []
    for gene in GENES:
        column = f"gene {gene}"
        for level in LEVELS:
            mean, half, count = mean_ci(final[final["level"] == level][column])
            rows.append({"gene": gene, "level": level, "mean": mean, "ci": half, "n": count})
    save_table(pd.DataFrame(rows), "e2_final_genes.csv")

    rows = []
    for gene in GENES:
        column = f"gene {gene}"
        scarce = final[final["level"] == LEVELS[0]][column]
        plenty = final[final["level"] == LEVELS[-1]][column]
        test = stats.ttest_ind(plenty, scarce, equal_var=False)
        interval = test.confidence_interval(0.95)
        trend = stats.spearmanr(final["level"], final[column], nan_policy="omit")
        rows.append({
            "gene": gene, "scarcest": scarce.mean(), "most_plentiful": plenty.mean(),
            "difference": plenty.mean() - scarce.mean(), "difference_low": interval.low,
            "difference_high": interval.high, "cohens_d": cohens_d(plenty, scarce),
            "p_value": test.pvalue, "trend_spearman": trend[0], "trend_p": trend[1],
        })
    save_table(pd.DataFrame(rows), "e2_gene_scarcity_effect.csv")

    # Each level's evolved genes against the neutral control at the same level.
    rows = []
    for level in LEVELS:
        for gene in GENES:
            chosen = final[final["level"] == level][f"gene {gene}"].dropna()
            drift = neutral[neutral["level"] == level][f"gene {gene}"].dropna()
            if len(drift) < 2:
                continue
            test = stats.ttest_ind(chosen, drift, equal_var=False)
            interval = test.confidence_interval(0.95)
            rows.append({"level": level, "gene": gene, "selection": chosen.mean(), "neutral": drift.mean(),
                         "difference": chosen.mean() - drift.mean(), "difference_low": interval.low,
                         "difference_high": interval.high, "cohens_d": cohens_d(chosen, drift),
                         "p_value": test.pvalue, "n_selection": len(chosen), "n_neutral": len(drift)})
    save_table(pd.DataFrame(rows), "e2_genes_vs_neutral.csv")

    # Figure: behaviour rates by level.
    rates = behaviour_rates(data)
    panels = [
        ("population_size", "Population size", "agents"),
        ("encounters_per_agent", "Meetings between the populations", "encounters per 100 timesteps lived"),
        ("starvation_share", "Deaths that were starvation", "share of deaths"),
        ("cooperate_rate", "Chose to cooperate\nwhen meeting a rival", "share of meetings"),
        ("fights_per_encounter", "Fights per encounter", "fights / encounters"),
        ("raid_rate", "Raided a rival farm\nwhen one was in reach", "share of opportunities"),
        ("farms_per_agent", "Farms per agent", "farms / agents"),
        ("stolen_per_agent", "Resources stolen per agent", "per 100 timesteps lived"),
        ("extinct", "How often a population died out", "share of runs"),
    ]
    rate_columns = ["cooperate_rate", "fights_per_encounter", "raid_rate"]
    figure, axes = plt.subplots(3, 3, figsize=(18, 14))
    for axis, (column, title, unit) in zip(axes.ravel(), panels):
        needed = MIN_RUNS if column in rate_columns else 2
        if column in ["fights_per_encounter", "encounters_per_agent"]:
            runs = rates[rates["population"] == "A"]
            draw_levels(axis, {level: runs[runs["level"] == level][column] for level in LEVELS},
                        "#374151", "Both populations", marker="s", min_runs=needed)
        elif column == "extinct":
            for population, offset in [("A", -0.17), ("B", 0.17)]:
                runs = rates[rates["population"] == population]
                shares = [runs[runs["level"] == level][column].mean() for level in LEVELS]
                axis.bar(np.array(LEVELS) + offset, shares, width=0.3, color=COLORS[population],
                         label=f"Population {population}")
            axis.set_ylim(0, 1)
        else:
            for population, offset in [("A", -0.07), ("B", 0.07)]:
                runs = rates[rates["population"] == population]
                draw_levels(axis, {level: runs[runs["level"] == level][column] for level in LEVELS},
                            COLORS[population], f"Population {population}", offset, min_runs=needed)
        if column in rate_columns:
            # Say so where the populations met too rarely for a rate to mean anything.
            counted = rates.groupby("level")[column].count()
            for level in LEVELS:
                if counted[level] < MIN_RUNS * (1 if column != "fights_per_encounter" else 2):
                    axis.text(level, 0.04, "too few\nmeetings", ha="center", va="bottom", fontsize=11,
                              color=NEUTRAL, transform=axis.get_xaxis_transform())
        axis.set_title(title, fontsize=15)
        axis.set_ylabel(unit)
        axis.set_ylim(bottom=0)
        if column.endswith("_rate") or column.endswith("_share"):
            axis.set_ylim(0, 1)
        level_axis(axis, label=False)
    for axis in axes[2]:
        axis.set_xlabel("Abundance level (scarce → plentiful)")
    axes[0, 0].legend(loc="upper left")
    axes[0, 1].legend(loc="upper right")
    figure.suptitle("E2. Behaviour at each abundance level, as rates\n(timesteps 1,000 to 2,000, "
                    "mean and 95% CI across 30 runs)", fontsize=18)
    figure.tight_layout()
    save(figure, "e2_behaviour_by_level.png")

    # Table: one value per run (the two populations averaged), then across runs.
    per_run = rates.groupby(["level", "seed"], as_index=False).mean(numeric_only=True)
    any_extinct = rates.groupby(["level", "seed"])["extinct"].max().reset_index()
    rows = []
    for level in LEVELS:
        runs = per_run[per_run["level"] == level]
        row = {"level": level, "multiplier": config.ABUNDANCE_MULTIPLIERS[level],
               "runs_with_extinction": any_extinct[any_extinct["level"] == level]["extinct"].sum(),
               "median_encounters": runs["encounters"].median(),
               "runs_with_20_meetings": runs["fights_per_encounter"].notna().sum()}
        for column in ["population_size", "starvation_share", "starved_per_1000", "cooperate_rate",
                       "fights_per_encounter", "raid_rate", "avoid_rate", "farms_per_agent",
                       "stolen_per_agent", "encounters_per_agent"]:
            mean, half, count = mean_ci(runs[column])
            enough = count >= (MIN_RUNS if column in rate_columns + ["avoid_rate"] else 2)
            row[column] = mean if enough else np.nan
            row[f"{column}_ci"] = half if enough else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    table["population_size"] *= 2  # Both populations together.
    table["population_size_ci"] *= 2
    save_table(table, "e2_behaviour.csv")

    rows = []
    for column in ["farms_per_agent", "stolen_per_agent", "starvation_share", "starved_per_1000",
                   "encounters_per_agent"]:
        scarce = per_run[per_run["level"] == LEVELS[0]][column]
        plenty = per_run[per_run["level"] == LEVELS[-1]][column]
        test = stats.ttest_ind(plenty, scarce, equal_var=False, nan_policy="omit")
        trend = stats.spearmanr(per_run["level"], per_run[column], nan_policy="omit")
        rows.append({"measure": column, "scarcest": scarce.mean(), "most_plentiful": plenty.mean(),
                     "difference": plenty.mean() - scarce.mean(), "cohens_d": cohens_d(plenty, scarce),
                     "p_value": test.pvalue, "trend_spearman": trend[0], "trend_p": trend[1]})
    save_table(pd.DataFrame(rows), "e2_behaviour_scarcity_effect.csv")

    # The populations only keep meeting at levels 1 and 2, so the encounter
    # rates can only be compared between those two.
    rows = []
    for column in ["cooperate_rate", "fights_per_encounter", "raid_rate", "avoid_rate", "stolen_per_agent",
                   "encounters_per_agent"]:
        first = per_run[per_run["level"] == 1][column].dropna()
        second = per_run[per_run["level"] == 2][column].dropna()
        test = stats.ttest_ind(second, first, equal_var=False)
        interval = test.confidence_interval(0.95)
        rows.append({"measure": column, "level_1": first.mean(), "n_1": len(first), "level_2": second.mean(),
                     "n_2": len(second), "difference": second.mean() - first.mean(),
                     "difference_low": interval.low, "difference_high": interval.high,
                     "cohens_d": cohens_d(second, first), "p_value": test.pvalue})
    save_table(pd.DataFrame(rows), "e2_behaviour_level1_vs_level2.csv")


# ---------------------------------------------------------------------------
# E3: veterans of plenty meet veterans of scarcity
# ---------------------------------------------------------------------------

def figure_e3():
    data = load("e3_timeseries.csv.gz")
    plenty_is_a = data["plenty_side"] == "A"
    data["plenty"] = np.where(plenty_is_a, data["Population A"], data["Population B"])
    data["scarcity"] = np.where(plenty_is_a, data["Population B"], data["Population A"])
    data["plenty_share"] = data["plenty"] / (data["plenty"] + data["scarcity"]).replace(0, np.nan)
    modes = [("evolving", "-", "Evolution continues"), ("frozen", "--", "Genomes frozen (no evolution)")]

    figure, axes = plt.subplots(1, 5, figsize=(22, 5.2), sharey=True)
    for axis, level in zip(axes, LEVELS):
        for mode, style, label in modes:
            runs = data[(data["level"] == level) & (data["condition"] == mode)]
            draw_band(axis, runs, "plenty_share", PLENTY if mode == "evolving" else "#111827", label, style)
        axis.axhline(0.5, color=NEUTRAL, linewidth=1, linestyle=":")
        axis.set_title(f"Level {level} (×{config.ABUNDANCE_MULTIPLIERS[level]:g})")
        axis.set_ylim(0, 1)
        axis.set_xlabel("Timestep")
    axes[0].set_ylabel("Plenty veterans' share\nof all agents")
    axes[0].legend(loc="lower left")
    axes[0].text(40, 0.96, "plenty veterans dominate ↑", fontsize=12, color="#374151", va="top")
    axes[0].text(40, 0.2, "scarcity veterans dominate ↓", fontsize=12, color="#374151", va="bottom")
    figure.suptitle("E3. Veterans of plenty and veterans of scarcity in a shared world "
                    "(mean and 95% band across 30 runs)", fontsize=18)
    figure.tight_layout()
    save(figure, "e3_share_over_time.png")

    final = final_rows(data, ["condition", "level", "seed", "plenty_side"])
    figure, axes = plt.subplots(1, 3, figsize=(20, 6))
    for mode, style, label in modes:
        runs = final[final["condition"] == mode]
        color = PLENTY if mode == "evolving" else "#111827"
        offset = -0.07 if mode == "evolving" else 0.07
        draw_levels(axes[0], {level: runs[runs["level"] == level]["plenty_share"] for level in LEVELS},
                    color, label, offset, marker="o" if mode == "evolving" else "s")
    axes[0].axhline(0.5, color=NEUTRAL, linewidth=1, linestyle=":")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Plenty veterans' final share of all agents")
    axes[0].set_title("Who ends up more numerous?")
    axes[0].legend(loc="upper right")
    level_axis(axes[0])
    # Who is bigger at the end: counts of runs.
    evolving = final[final["condition"] == "evolving"]
    plenty_wins = [(evolving[evolving["level"] == level]["plenty_share"] > 0.5).sum() for level in LEVELS]
    scarcity_wins = [(evolving[evolving["level"] == level]["plenty_share"] < 0.5).sum() for level in LEVELS]
    axes[1].bar(np.array(LEVELS) - 0.19, plenty_wins, width=0.36, color=PLENTY, label="Plenty veterans larger")
    axes[1].bar(np.array(LEVELS) + 0.19, scarcity_wins, width=0.36, color=SCARCITY, label="Scarcity veterans larger")
    axes[1].set_ylabel("Runs (out of 30)")
    axes[1].set_ylim(0, 30)
    axes[1].set_title("Larger population at the end\n(evolution continues)")
    axes[1].legend(loc="upper center", fontsize=12)
    level_axis(axes[1])
    for side, color in [("plenty", PLENTY), ("scarcity", SCARCITY)]:
        draw_levels(axes[2], {level: evolving[evolving["level"] == level][side] for level in LEVELS},
                    color, f"{side.capitalize()} veterans", -0.07 if side == "plenty" else 0.07)
    axes[2].set_ylabel("Final population size (agents)")
    axes[2].set_ylim(bottom=0)
    axes[2].set_title("Population sizes\n(evolution continues)")
    axes[2].legend(loc="upper left")
    level_axis(axes[2])
    figure.suptitle("E3. Outcome of the shared world at each abundance level "
                    "(last 200 timesteps, mean and 95% CI across 30 runs)", fontsize=18)
    figure.tight_layout()
    save(figure, "e3_outcome_by_level.png")

    rows = []
    for mode, _, _ in modes:
        for level in LEVELS:
            runs = final[(final["condition"] == mode) & (final["level"] == level)]
            share = runs["plenty_share"].dropna()
            mean, half, count = mean_ci(share)
            wins = int((share > 0.5).sum())
            losses = int((share < 0.5).sum())
            top = runs[runs["plenty_side"] == "A"]["plenty_share"].mean()
            bottom = runs[runs["plenty_side"] == "B"]["plenty_share"].mean()
            rows.append({
                "mode": mode, "level": level, "plenty_share": mean, "ci": half, "n": count,
                "p_vs_half": stats.ttest_1samp(share, 0.5).pvalue,
                "plenty_larger": wins, "scarcity_larger": losses,
                "sign_test_p": stats.binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else np.nan,
                "plenty_extinct": int((runs["plenty"] == 0).sum()),
                "scarcity_extinct": int((runs["scarcity"] == 0).sum()),
                "share_when_plenty_on_top": top, "share_when_plenty_on_bottom": bottom,
                "plenty_size": runs["plenty"].mean(), "scarcity_size": runs["scarcity"].mean(),
            })
    save_table(pd.DataFrame(rows), "e3_outcome.csv")

    # How different were the two founding gene pools? (timestep 0 of the frozen runs)
    start = data[(data["Timestep"] == 0) & (data["condition"] == "frozen") & (data["level"] == LEVELS[0])]
    rows = []
    for gene in GENES:
        column_a, column_b = f"Population A average {gene}", f"Population B average {gene}"
        plenty = np.where(start["plenty_side"] == "A", start[column_a], start[column_b])
        scarcity = np.where(start["plenty_side"] == "A", start[column_b], start[column_a])
        difference = plenty - scarcity
        mean, half, count = mean_ci(difference)
        rows.append({"gene": gene, "plenty_veterans": plenty.mean(), "scarcity_veterans": scarcity.mean(),
                     "difference": mean, "ci": half, "n": count})
    save_table(pd.DataFrame(rows), "e3_founding_genes.csv")


# ---------------------------------------------------------------------------
# E4: environmental shock
# ---------------------------------------------------------------------------

def figure_e4():
    data = load("e4_timeseries.csv.gz")
    styles = {
        "plenty then scarcity": (PLENTY, "-", "Plenty, then scarcity"),
        "scarcity then plenty": (SCARCITY, "-", "Scarcity, then plenty"),
        "always plenty": (PLENTY, ":", "Always plenty (no shock)"),
        "always scarcity": (SCARCITY, ":", "Always scarcity (no shock)"),
    }
    figure, axes = plt.subplots(2, 3, figsize=(17, 9.5), sharex=True)
    axes = axes.ravel()
    for axis, gene in zip(axes, GENES):
        for condition, (color, style, label) in styles.items():
            draw_band(axis, data[data["condition"] == condition], f"gene {gene}", color, label, style)
        axis.axvline(STEPS, color="#111827", linewidth=1.2)
        axis.set_title(GENE_LABELS[gene])
        axis.set_ylim(0, 1)
    for condition, (color, style, label) in styles.items():
        draw_band(axes[5], data[data["condition"] == condition], "Total population", color, label, style)
    axes[5].axvline(STEPS, color="#111827", linewidth=1.2)
    axes[5].set_title("Population size (both populations)")
    axes[5].set_ylabel("agents")
    axes[5].set_ylim(bottom=0)
    axes[5].legend(loc="center left", fontsize=11)
    axes[0].text(STEPS + 40, 0.03, "environment switches", fontsize=12, color="#111827")
    for axis in axes[3:]:
        axis.set_xlabel("Timestep")
    for axis in [axes[0], axes[3]]:
        axis.set_ylabel("Average gene value")
    figure.suptitle("E4. Genes before and after an environmental shock at timestep 2,000\n"
                    "(both populations pooled, mean and 95% band across 30 runs)", fontsize=18)
    figure.tight_layout()
    save(figure, "e4_shock.png")

    before = final_rows(data, ["condition", "seed"], steps=STEPS)
    after = final_rows(data, ["condition", "seed"], steps=2 * STEPS)
    rows = []
    for gene in GENES:
        column = f"gene {gene}"
        row = {"gene": gene}
        for condition in styles:
            name = condition.replace(" ", "_")
            row[f"{name}_before"] = before[before["condition"] == condition][column].mean()
            mean, half, _ = mean_ci(after[after["condition"] == condition][column])
            row[f"{name}_after"] = mean
            row[f"{name}_after_ci"] = half
        for shocked, target in [("plenty then scarcity", "always scarcity"),
                                ("scarcity then plenty", "always plenty")]:
            first = after[after["condition"] == shocked][column]
            second = after[after["condition"] == target][column]
            row[f"{shocked.replace(' ', '_')}_vs_target_p"] = stats.ttest_ind(first, second, equal_var=False).pvalue
        rows.append(row)
    save_table(pd.DataFrame(rows), "e4_genes_before_after.csv")

    # Within-run change across the shock, compared with the unshocked runs' change.
    merged = before.merge(after, on=["condition", "seed"], suffixes=("_before", "_after"))
    rows = []
    for gene in GENES:
        merged[f"change {gene}"] = merged[f"gene {gene}_after"] - merged[f"gene {gene}_before"]
        for shocked, control in [("plenty then scarcity", "always plenty"),
                                 ("scarcity then plenty", "always scarcity")]:
            first = merged[merged["condition"] == shocked][f"change {gene}"]
            second = merged[merged["condition"] == control][f"change {gene}"]
            test = stats.ttest_ind(first, second, equal_var=False)
            interval = test.confidence_interval(0.95)
            rows.append({"gene": gene, "shock": shocked, "change_with_shock": first.mean(),
                         "change_without_shock": second.mean(),
                         "difference": first.mean() - second.mean(), "difference_low": interval.low,
                         "difference_high": interval.high, "cohens_d": cohens_d(first, second),
                         "p_value": test.pvalue})
    save_table(pd.DataFrame(rows), "e4_shock_effect.csv")
    sizes = []
    for condition in styles:
        runs = data[data["condition"] == condition]
        for label, low, high in [("before", STEPS - FINAL_WINDOW, STEPS), ("200 steps after", STEPS, STEPS + 200),
                                 ("end", 2 * STEPS - FINAL_WINDOW, 2 * STEPS)]:
            window = runs[(runs["Timestep"] > low) & (runs["Timestep"] <= high)]
            sizes.append({"condition": condition, "window": label,
                          "population": window.groupby("seed")["Total population"].mean().mean()})
    save_table(pd.DataFrame(sizes), "e4_population_size.csv")


# ---------------------------------------------------------------------------
# E5: which genes pay off, at which level
# ---------------------------------------------------------------------------

# Offspring per agent in each gene-value bin, relative to the overall average,
# pooling the agents of all runs at one level. (Averaging per-run ratios
# instead would be biased in small populations: a run's gene pool drifts into
# one bin, and that crowded bin is by definition where the successful families
# are.) The 95% interval is a bootstrap over runs.
def pooled_relative_offspring(chosen):
    chosen = chosen.assign(total=chosen["mean_offspring"] * chosen["agents"])
    agents = chosen.pivot(index="seed", columns="bin_middle", values="agents").fillna(0).to_numpy()
    totals = chosen.pivot(index="seed", columns="bin_middle", values="total").fillna(0).to_numpy()
    middles = np.array(sorted(chosen["bin_middle"].unique()))

    def relative(rows):
        return (totals[rows].sum(axis=0) / agents[rows].sum(axis=0)) / (totals[rows].sum() / agents[rows].sum())

    everyone = np.arange(len(agents))
    sampler = np.random.default_rng(0)
    resampled = np.array([relative(sampler.choice(everyone, len(everyone))) for _ in range(1000)])
    low, high = np.nanpercentile(resampled, [2.5, 97.5], axis=0)
    return middles, relative(everyone), low, high


def figure_e5():
    selection = pd.read_csv(os.path.join(RESULTS, "e5_selection.csv"))
    bins = pd.read_csv(os.path.join(RESULTS, "e5_offspring_by_gene.csv"))
    shades = {1: "#a8c5ee", 2: "#4f86d6", 5: "#0f3d85"}  # One hue, light = scarce, dark = plentiful.

    figure, axes = plt.subplots(2, 5, figsize=(22, 9.5), sharey="row")
    for column, gene in enumerate(GENES):
        runs = selection[selection["gene"] == gene]
        axis = axes[0, column]
        draw_levels(axis, {level: runs[runs["level"] == level]["spearman"] for level in LEVELS},
                    "#111827", "Mean across runs", marker="s")
        axis.axhline(0, color=NEUTRAL, linewidth=1.2)
        axis.set_title(GENE_LABELS[gene])
        level_axis(axis)
        axis = axes[1, column]
        for level, shade in shades.items():
            chosen = bins[(bins["gene"] == gene) & (bins["level"] == level)]
            middles, means, lows, highs = pooled_relative_offspring(chosen)
            axis.errorbar(middles, means, yerr=[means - lows, highs - means], color=shade, marker="o",
                          markersize=7, capsize=3,
                          label=f"Level {level} (×{config.ABUNDANCE_MULTIPLIERS[level]:g})")
        axis.axhline(1, color=NEUTRAL, linewidth=1.2)
        axis.set_xlim(0, 1)
        axis.set_xlabel(f"{GENE_LABELS[gene]} gene value")
    axes[0, 0].set_ylabel("Rank correlation between\ngene and offspring count")
    axes[1, 0].set_ylabel("Offspring relative to the\nlevel's average (1 = average)")
    axes[1, 0].legend(loc="best", fontsize=12)
    figure.suptitle("E5. Which genes pay off? Gene value against offspring count (realized fitness)\n"
                    "Top: above the grey line = a higher gene value means more offspring. "
                    "Mean and 95% CI across 30 runs.", fontsize=18)
    figure.tight_layout()
    save(figure, "e5_gene_vs_offspring.png")

    rows = []
    for gene in GENES:
        for level in LEVELS:
            runs = selection[(selection["gene"] == gene) & (selection["level"] == level)]
            rho, rho_half, count = mean_ci(runs["spearman"])
            slope, slope_half, _ = mean_ci(runs["regression"])
            rows.append({"gene": gene, "level": level, "spearman": rho, "spearman_ci": rho_half,
                         "p_vs_zero": stats.ttest_1samp(runs["spearman"].dropna(), 0).pvalue,
                         "regression": slope, "regression_ci": slope_half, "n": count,
                         "agents_per_run": runs["agents"].mean(), "gene_std": runs["gene_std"].mean(),
                         "mean_offspring": runs["mean_offspring"].mean()})
    save_table(pd.DataFrame(rows), "e5_selection_by_level.csv")


# ---------------------------------------------------------------------------
# Stretch experiments (only drawn if their results exist)
# ---------------------------------------------------------------------------

def figure_stretch():
    path = os.path.join(RESULTS, "stretch_timeseries.csv.gz")
    if not os.path.exists(path):
        return
    data = load("stretch_timeseries.csv.gz")
    final = final_rows(data, ["condition", "seed"])
    fights = data[data["condition"].str.startswith("fight cost")]
    rows = []
    for condition, runs in data.groupby("condition"):
        ends = final[final["condition"] == condition]
        row = {"condition": condition, "n": len(ends), "population": ends["Total population"].mean()}
        for gene in GENES:
            mean, half, _ = mean_ci(ends[f"gene {gene}"])
            row[gene] = mean
            row[f"{gene}_ci"] = half
        rates = []
        for seed, run in runs.groupby("seed"):
            run = run.set_index("Timestep")
            first, last = run.loc[STEPS // 2], run.loc[STEPS]
            encounters = last["Cumulative encounters"] - first["Cumulative encounters"]
            rates.append((last["Cumulative fights"] - first["Cumulative fights"]) / encounters
                         if encounters else np.nan)
        row["fights_per_encounter"], row["fights_per_encounter_ci"], _ = mean_ci(rates)
        rows.append(row)
    table = pd.DataFrame(rows)
    save_table(table, "stretch.csv")

    costs = sorted(float(name.split()[-1]) for name in fights["condition"].unique())
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for axis, column, title in [(axes[0], "risk_tolerance", "Risk tolerance gene"),
                                (axes[1], "cooperative_tendency", "Cooperation gene"),
                                (axes[2], "fights_per_encounter", "Fights per encounter")]:
        means = [table[table["condition"] == f"fight cost {cost:g}"][column].iloc[0] for cost in costs]
        halves = [table[table["condition"] == f"fight cost {cost:g}"][f"{column}_ci"].iloc[0] for cost in costs]
        axis.errorbar(costs, means, yerr=halves, color="#111827", marker="s", markersize=8, capsize=4)
        axis.axvline(config.FIGHT_COST, color=NEUTRAL, linestyle=":", linewidth=1.5)
        axis.set_title(title)
        axis.set_xlabel("Fight cost (dotted line = default)")
        axis.set_ylim(0, 1 if column != "fights_per_encounter" else None)
    axes[0].set_ylabel("Final value")
    figure.suptitle("Stretch. Does a cheap fight make everyone evolve to fight? "
                    "(level 2, mean and 95% CI across 20 runs)", fontsize=18)
    figure.tight_layout()
    save(figure, "stretch_fight_cost.png")


if __name__ == "__main__":
    chosen = sys.argv[1:] or ["e1", "e2", "e3", "e4", "e5", "stretch"]
    for name in chosen:
        globals()[f"figure_{name}"]()
