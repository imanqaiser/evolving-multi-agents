# Regenerates every Model 2 figure (figures/model2/) and table
# (results/model2/tables/) from the saved Model 2 experiment results.
#
# Run with: python analysis/make_figures_model2.py
#
# Same conventions as analysis/make_figures.py:
#   - The unit of replication is the RUN (one seed): every mean and every 95%
#     confidence interval is taken across runs.
#   - Behaviour is always a rate (per encounter, per agent, a share of food),
#     never a raw count, and population size is shown next to it.
#   - Every selection result is compared with the NEUTRAL control at the same level.
import gzip
import json
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from make_figures import cohens_d, draw_band, mean_ci  # noqa: E402  (also sets the plot style)

RESULTS = os.path.join(ROOT, "results", "model2")
TABLES = os.path.join(RESULTS, "tables")
FIGURES = os.path.join(ROOT, "figures", "model2")

GENES = config.GENES + config.BODY_GENE_NAMES + config.ZONE_GENE_NAMES
OPEN_GENES = config.GENES + config.BODY_GENE_NAMES      # The open map has no expansion gene.
LABELS = {
    "exploratory_tendency": "Exploration", "risk_tolerance": "Risk tolerance",
    "sociability": "Sociability", "cooperative_tendency": "Cooperation",
    "farming_tendency": "Farming", "vision": "Vision", "body_size": "Body size",
    "expansion": "Expansion",
}
LEVELS = sorted(config.M2_MULTIPLIERS)
NAMED = config.M2_NAMED_LEVELS
SCARCE, MIDDLE, PLENTIFUL = NAMED["SCARCE"], NAMED["MIDDLE"], NAMED["PLENTIFUL"]
LEVEL_NAME = {SCARCE: "Scarce", MIDDLE: "Middle", PLENTIFUL: "Plentiful"}
LEVEL_COLOR = {SCARCE: "#7b3fa0", MIDDLE: "#b45309", PLENTIFUL: "#2a7f3f"}
COLORS = {"A": "#1556b8", "B": "#c2560c"}
NEUTRAL = "#6b7280"
STEPS = 1500
FINAL_WINDOW = 200
MIN_EVENTS = 20
ALL_TABLES = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(name, genes=GENES):
    data = pd.read_csv(os.path.join(RESULTS, name))
    size_a, size_b = data["Population A"], data["Population B"]
    for gene in genes:
        column_a, column_b = f"Population A average {gene}", f"Population B average {gene}"
        data.loc[size_a == 0, column_a] = np.nan
        data.loc[size_b == 0, column_b] = np.nan
        total = data[column_a].fillna(0) * size_a + data[column_b].fillna(0) * size_b
        data[f"gene {gene}"] = total / (size_a + size_b).replace(0, np.nan)
    data["Total population"] = size_a + size_b
    return data


# One row per run: the average of its last FINAL_WINDOW timesteps.
def final_rows(data, keys, steps=STEPS):
    late = data[(data["Timestep"] > steps - FINAL_WINDOW) & (data["Timestep"] <= steps)]
    return late.groupby(keys, as_index=False).mean(numeric_only=True)


# One row per run: behaviour rates over timesteps start..end, from the running totals.
def behaviour(data, keys, start, end):
    first = data[data["Timestep"] == start].set_index(keys)
    last = data[data["Timestep"] == end].set_index(keys)
    window = data[(data["Timestep"] >= start) & (data["Timestep"] <= end)]

    def change(column):
        return last[column] - first[column]

    def both(template):
        return change(template.format("A")) + change(template.format("B"))

    out = pd.DataFrame(index=last.index)
    lived = both("Population {} agent steps")
    encounters = change("Cumulative encounters")
    decisions = both("Population {} cumulative cooperate") + both("Population {} cumulative compete")
    raids = both("Population {} cumulative raid") + both("Population {} cumulative respect")
    out["agents"] = window.groupby(keys)["Total population"].mean()
    out["smaller population"] = window.groupby(keys)[["Population A", "Population B"]].mean().min(axis=1)
    out["both survive"] = ((last["Population A"] > 0) & (last["Population B"] > 0)).astype(float)
    out["encounters"] = encounters
    out["encounters per 100 agent-steps"] = 100 * encounters / lived.replace(0, np.nan)
    out["cooperation rate"] = (both("Population {} cumulative cooperate") / decisions).where(decisions >= MIN_EVENTS)
    out["fights per encounter"] = (change("Cumulative fights") / encounters).where(encounters >= MIN_EVENTS)
    out["raid rate"] = (both("Population {} cumulative raid") / raids).where(raids >= MIN_EVENTS)
    out["stolen per agent per 100 steps"] = 100 * both("Population {} cumulative stolen") / lived.replace(0, np.nan)
    out["trades per agent per 100 steps"] = 100 * both("Population {} trades") / lived.replace(0, np.nan)
    deaths = change("Deaths from starvation") + change("Deaths from old age")
    out["starvation share of deaths"] = change("Deaths from starvation") / deaths.replace(0, np.nan)
    if "Population A food farm" in data:
        food = {s: both("Population {} food " + s) for s in ["farm", "wild", "raid"]}
        total_food = (food["farm"] + food["wild"] + food["raid"]).replace(0, np.nan)
        for source in food:
            out[f"food share {source}"] = food[source] / total_food
        where = {z: both("Population {} steps " + z) for z in ["home", "shared", "rival"]}
        total_time = (where["home"] + where["shared"] + where["rival"]).replace(0, np.nan)
        for zone in where:
            out[f"time share {zone}"] = where[zone] / total_time
        home = change("Encounters in homeland A") + change("Encounters in homeland B")
        out["share of encounters in a homeland"] = (home / encounters).where(encounters >= MIN_EVENTS)
        home_fights = change("Fights in homeland A") + change("Fights in homeland B")
        fights = change("Cumulative fights")
        out["share of fights in a homeland"] = (home_fights / fights).where(fights >= MIN_EVENTS)
    return out.reset_index()


def compare(first, second):
    """Difference of means with 95% CI (Welch), Cohen's d and p value."""
    first = np.asarray(first, float)
    second = np.asarray(second, float)
    first, second = first[~np.isnan(first)], second[~np.isnan(second)]
    if len(first) < 3 or len(second) < 3:
        return {"difference": np.nan, "low": np.nan, "high": np.nan, "d": np.nan, "p": np.nan}
    test = stats.ttest_ind(first, second, equal_var=False)
    interval = test.confidence_interval(0.95)
    return {"difference": first.mean() - second.mean(), "low": interval.low, "high": interval.high,
            "d": cohens_d(first, second), "p": test.pvalue}


def save(figure, name):
    os.makedirs(FIGURES, exist_ok=True)
    figure.savefig(os.path.join(FIGURES, name), bbox_inches="tight")
    plt.close(figure)
    print(f"Saved figures/model2/{name}")


def save_table(table, name):
    os.makedirs(TABLES, exist_ok=True)
    table.to_csv(os.path.join(TABLES, name), index=False, float_format="%.4g")
    text = f"\n--- {name} ---\n" + table.to_string(index=False, float_format=lambda value: f"{value:.3f}")
    print(text)
    ALL_TABLES.append(text)


def level_axis(axis, label=True):
    axis.set_xticks(LEVELS)
    axis.set_xticklabels([f"{level}\n×{config.M2_MULTIPLIERS[level]:g}" for level in LEVELS])
    axis.set_xlim(0.5, len(LEVELS) + 0.5)
    if label:
        axis.set_xlabel("Homeland productivity (scarce → plentiful)")


def draw_levels(axis, table, column, color, label, offset=0.0, marker="o"):
    means, halves = [], []
    for level in LEVELS:
        mean, half, count = mean_ci(table[table["level"] == level][column])
        means.append(mean if count >= 5 else np.nan)
        halves.append(half if count >= 5 else np.nan)
    axis.errorbar(np.array(LEVELS) + offset, means, yerr=halves, color=color, marker=marker, markersize=8,
                  capsize=4, label=label, markeredgecolor="white", markeredgewidth=1)


# ---------------------------------------------------------------------------
# M1 and M2: evolution or drift, and the scarcity gradient
# ---------------------------------------------------------------------------

def figures_m1_m2():
    data = load("m2_timeseries.csv.gz")
    keys = ["condition", "level", "seed"]
    selection = data[data["condition"] == "selection"]
    neutral = data[data["condition"] == "neutral"]

    # --- M1: gene trajectories, scarce and plentiful worlds on the same axes.
    figure, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True, sharey=True)
    for axis, gene in zip(axes.ravel(), GENES):
        draw_band(axis, neutral[neutral["level"].isin([SCARCE, PLENTIFUL])], f"gene {gene}", NEUTRAL,
                  "Neutral control (genes do nothing)", "--")
        for level in [SCARCE, PLENTIFUL]:
            draw_band(axis, selection[selection["level"] == level], f"gene {gene}", LEVEL_COLOR[level],
                      f"{LEVEL_NAME[level]} homelands, with selection")
        axis.set_title(LABELS[gene])
        axis.set_ylim(0, 1)
    for axis in axes[1]:
        axis.set_xlabel("Timestep (about 7 per generation)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Average gene value")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=15, bbox_to_anchor=(0.5, -0.04))
    count = selection[selection["level"] == SCARCE]["seed"].nunique()
    figure.suptitle("M1. How the genes evolve in scarce and in plentiful worlds\n"
                    f"(mean and 95% band across {count} runs each; grey = drift alone)", fontsize=19)
    figure.tight_layout()
    save(figure, "m1_gene_trajectories.png")

    final = final_rows(data, keys)
    chosen, drift = final[final["condition"] == "selection"], final[final["condition"] == "neutral"]

    # Table: every gene at every level against its neutral control.
    rows = []
    for level in LEVELS:
        for gene in GENES:
            with_selection = chosen[chosen["level"] == level][f"gene {gene}"]
            control = drift[drift["level"] == level][f"gene {gene}"]
            row = {"level": level, "gene": LABELS[gene], "selection": with_selection.mean(),
                   "neutral": control.mean()}
            row.update(compare(with_selection, control))
            row["separates"] = bool(row["low"] > 0 or row["high"] < 0)
            rows.append(row)
    versus = pd.DataFrame(rows)
    save_table(versus, "m2_genes_vs_neutral.csv")
    counts = versus.groupby("level")["separates"].sum().reset_index(name="genes that separate from drift (of 8)")
    save_table(counts, "m1_genes_under_selection.csv")

    # Table: does the gene change along the gradient? Spearman across all selection runs.
    rows = []
    for gene in GENES:
        rho, p = stats.spearmanr(chosen["level"], chosen[f"gene {gene}"], nan_policy="omit")
        scarce, plentiful = chosen[chosen["level"] == SCARCE][f"gene {gene}"], chosen[chosen["level"] == PLENTIFUL][f"gene {gene}"]
        row = {"gene": LABELS[gene], "scarce": scarce.mean(), "plentiful": plentiful.mean(), "trend rho": rho, "trend p": p}
        row.update({f"plentiful - scarce {k}": v for k, v in compare(plentiful, scarce).items()})
        rows.append(row)
    save_table(pd.DataFrame(rows), "m2_gene_trend.csv")

    # --- Trait profiles: what an agent evolved in each world looks like.
    figure, axis = plt.subplots(figsize=(17, 7))
    width = 0.26
    positions = np.arange(len(GENES))
    for index, level in enumerate([SCARCE, MIDDLE, PLENTIFUL]):
        means, halves = [], []
        for gene in GENES:
            mean, half, _ = mean_ci(chosen[chosen["level"] == level][f"gene {gene}"])
            means.append(mean)
            halves.append(half)
        axis.bar(positions + (index - 1) * width, means, width, yerr=halves, capsize=3, color=LEVEL_COLOR[level],
                 label=f"{LEVEL_NAME[level]} homelands (×{config.M2_MULTIPLIERS[level]:g})")
        for position, gene in zip(positions, GENES):
            mean = drift[drift["level"] == level][f"gene {gene}"].mean()
            axis.plot([position + (index - 1) * width - width / 2, position + (index - 1) * width + width / 2],
                      [mean, mean], color="black", linewidth=2.5,
                      label="Neutral control (drift alone)" if index == 0 and gene == GENES[0] else None)
    axis.set_xticks(positions)
    axis.set_xticklabels([LABELS[gene] for gene in GENES], fontsize=15)
    axis.set_ylabel("Evolved gene value (0 to 1)")
    axis.set_ylim(0, 1)
    axis.grid(axis="x", visible=False)
    axis.legend(loc="upper right", ncol=2)
    axis.set_title(f"What evolves in each world: the average agent after {STEPS:,} timesteps\n"
                   f"(mean and 95% interval across {count} runs; black line = what drift alone gives)")
    save(figure, "m2_trait_profiles.png")

    # --- Genes across the whole gradient.
    figure, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True, sharey=True)
    for axis, gene in zip(axes.ravel(), GENES):
        draw_levels(axis, drift, f"gene {gene}", NEUTRAL, "Neutral control", 0.08, "D")
        draw_levels(axis, chosen, f"gene {gene}", "#111827", "With selection", -0.08)
        axis.set_title(LABELS[gene])
        axis.set_ylim(0, 1)
        level_axis(axis, label=False)
    for axis in axes[1]:
        axis.set_xlabel("Homeland productivity level")
    for axis in axes[:, 0]:
        axis.set_ylabel("Final gene value")
    axes[0, 0].legend(loc="upper left")
    figure.suptitle("M2. Evolved genes across the scarcity gradient (level 1 = poorest homelands)\n"
                    f"(mean and 95% interval across {count} runs per level)", fontsize=19)
    figure.tight_layout()
    save(figure, "m2_genes_by_level.png")

    # --- Behaviour across the gradient (second half of each run).
    rates = behaviour(data, keys, STEPS // 2, STEPS)
    chosen_rates, drift_rates = rates[rates["condition"] == "selection"], rates[rates["condition"] == "neutral"]
    panels = [("food share farm", "Share of food from own farms"),
              ("time share shared", "Share of time spent in the shared land"),
              ("encounters per 100 agent-steps", "Meetings with rivals\nper 100 agent-timesteps"),
              ("cooperation rate", "Share of meetings where the agent cooperated"),
              ("fights per encounter", "Fights per meeting"),
              ("stolen per agent per 100 steps", "Resources stolen\nper agent per 100 timesteps"),
              ("share of encounters in a homeland", "Share of meetings that happen\ninside a homeland"),
              ("agents", "Agents alive (both populations)")]
    figure, axes = plt.subplots(2, 4, figsize=(21, 9.5), sharex=True)
    for axis, (column, title) in zip(axes.ravel(), panels):
        draw_levels(axis, drift_rates, column, NEUTRAL, "Neutral control", 0.08, "D")
        draw_levels(axis, chosen_rates, column, "#111827", "With selection", -0.08)
        axis.set_title(title, fontsize=14)
        axis.set_ylim(bottom=0)
        level_axis(axis, label=False)
    for axis in axes[1]:
        axis.set_xlabel("Homeland productivity level")
    axes[0, 0].legend(loc="best")
    figure.suptitle("M2. Behaviour across the scarcity gradient (second half of each run)", fontsize=19)
    figure.tight_layout()
    save(figure, "m2_behaviour_by_level.png")

    columns = [c for c in rates.columns if c not in keys]
    summary = rates.groupby(["condition", "level"])[columns].mean().reset_index()
    save_table(summary, "m2_behaviour.csv")
    rows = []
    for column in columns:
        scarce, plentiful = chosen_rates[chosen_rates["level"] == SCARCE][column], chosen_rates[chosen_rates["level"] == PLENTIFUL][column]
        rho, p = stats.spearmanr(chosen_rates["level"], chosen_rates[column], nan_policy="omit")
        row = {"measure": column, "scarce": scarce.mean(), "plentiful": plentiful.mean(), "trend rho": rho, "trend p": p}
        row.update(compare(plentiful, scarce))
        rows.append(row)
    save_table(pd.DataFrame(rows), "m2_behaviour_trend.csv")

    # --- Food sources, stacked.
    figure, axis = plt.subplots(figsize=(11, 6.5))
    bottom = np.zeros(len(LEVELS))
    for source, color, label in [("farm", "#2a7f3f", "Own farms (homeland)"), ("wild", "#d4a017", "Wild food (shared land)"),
                                 ("raid", "#b91c1c", "Raided from rival farms")]:
        shares = np.array([chosen_rates[chosen_rates["level"] == level][f"food share {source}"].mean() for level in LEVELS])
        axis.bar(LEVELS, shares, bottom=bottom, color=color, label=label, width=0.7)
        bottom += shares
    level_axis(axis)
    axis.set_ylabel("Share of all food eaten")
    axis.set_ylim(0, 1)
    axis.grid(axis="x", visible=False)
    axis.legend(loc="lower right", framealpha=0.9, frameon=True)
    axis.set_title("Where the food comes from, by homeland productivity")
    save(figure, "m2_food_sources.png")

    # --- Where the populations meet: encounters by grid row.
    rows = pd.read_csv(os.path.join(RESULTS, "m2_encounter_rows.csv"))
    rows = rows[rows["condition"] == "selection"]
    row_columns = [c for c in rows.columns if c.startswith("row ")]
    size = len(row_columns)
    shared_first = (size - config.MODEL_2_ZONES["SHARED_ROWS"]) // 2
    shared_last = shared_first + config.MODEL_2_ZONES["SHARED_ROWS"] - 1
    figure, axis = plt.subplots(figsize=(13, 6))
    axis.axvspan(-0.5, shared_first - 0.5, color=COLORS["A"], alpha=0.08)
    axis.axvspan(shared_first - 0.5, shared_last + 0.5, color="#d4a017", alpha=0.12)
    axis.axvspan(shared_last + 0.5, size - 0.5, color=COLORS["B"], alpha=0.08)
    for level in [SCARCE, MIDDLE, PLENTIFUL]:
        counts = rows[rows["level"] == level][row_columns]
        share = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).mean()
        axis.plot(range(size), share.values, color=LEVEL_COLOR[level], label=f"{LEVEL_NAME[level]} homelands")
    top = axis.get_ylim()[1]
    for x, text in [(shared_first / 2, "Homeland A"), (size / 2, "Shared land"), ((shared_last + size) / 2, "Homeland B")]:
        axis.text(x, top * 0.95, text, ha="center", va="top", fontsize=14, fontweight="bold")
    axis.set_xlim(-0.5, size - 0.5)
    axis.set_xlabel("Grid row (top to bottom)")
    axis.set_ylabel("Share of all meetings between the populations")
    axis.legend(loc="center left")
    axis.set_title("Where the two populations meet")
    save(figure, "m2_where_they_meet.png")

    # --- Trait space: vision against body size, before and after.
    with gzip.open(os.path.join(RESULTS, "m2_gene_pools.json.gz"), "rt") as file:
        pools = json.load(file)
    sampler = random.Random(0)
    figure, axes = plt.subplots(1, 3, figsize=(18, 6.3), sharex=True, sharey=True)
    start = [(sampler.random(), sampler.random()) for _ in range(1500)]
    axes[0].scatter(*zip(*start), s=10, color=NEUTRAL, alpha=0.5)
    axes[0].set_title("Starting agents (random genes)")
    for axis, level in zip(axes[1:], [SCARCE, PLENTIFUL]):
        genomes = [g for run in pools[str(level)].values() for population in run.values() for g in population]
        sample = sampler.sample(genomes, min(1500, len(genomes)))
        axis.scatter([g["vision"] for g in sample], [g["body_size"] for g in sample], s=10,
                     color=LEVEL_COLOR[level], alpha=0.5)
        axis.set_title(f"After evolution: {LEVEL_NAME[level].lower()} homelands")
    for axis in axes:
        axis.set_xlabel("Vision gene")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
    axes[0].set_ylabel("Body size gene")
    figure.suptitle("Trait space before and after evolution (1,500 agents sampled from all runs)", fontsize=18)
    figure.tight_layout()
    save(figure, "m2_trait_space.png")

    # --- Selection strength (also used by M6).
    strength = pd.read_csv(os.path.join(RESULTS, "m2_selection.csv"))
    rows = []
    for (condition, level), group in strength.groupby(["condition", "level"]):
        row = {"condition": condition, "level": level}
        for gene in GENES:
            mean, half, _ = mean_ci(group[f"rho {gene}"])
            row[LABELS[gene]] = mean
            row[LABELS[gene] + " ±"] = half
        row["offspring top / bottom fitness quarter"] = (group["offspring top quarter"] / group["offspring bottom quarter"]).replace(np.inf, np.nan).mean()
        rows.append(row)
    save_table(pd.DataFrame(rows), "m2_selection_strength.csv")
    return versus, chosen, drift, strength


# ---------------------------------------------------------------------------
# M7: unequal homelands (one population rich, one poor, in the same world)
# ---------------------------------------------------------------------------

def figures_m7():
    data = pd.read_csv(os.path.join(RESULTS, "m7_timeseries.csv.gz"))
    # Relabel the two populations as "rich" and "poor" instead of A and B.
    for side, own in [("rich", lambda rich: rich), ("poor", lambda rich: "B" if rich == "A" else "A")]:
        letter = data["rich_side"].map(own)
        for column in [c for c in data.columns if c.startswith("Population A")]:
            suffix = column[len("Population A"):]
            data[side + suffix] = np.where(letter == "A", data["Population A" + suffix], data["Population B" + suffix])
    for side in ["rich", "poor"]:
        for gene in GENES:
            data.loc[data[side] == 0, f"{side} average {gene}"] = np.nan
    keys = ["condition", "seed"]
    final = final_rows(data, keys)
    first = data[data["Timestep"] == STEPS // 2].set_index(keys)
    last = data[data["Timestep"] == STEPS].set_index(keys)
    rates = pd.DataFrame(index=last.index)
    for side in ["rich", "poor"]:
        change = lambda column: last[f"{side} {column}"] - first[f"{side} {column}"]  # noqa: E731
        food = change("food farm") + change("food wild") + change("food raid")
        time = change("steps home") + change("steps shared") + change("steps rival")
        rates[f"{side} population"] = data[data["Timestep"] >= STEPS // 2].groupby(keys)[side].mean()
        rates[f"{side} food from farms"] = change("food farm") / food.replace(0, np.nan)
        rates[f"{side} food from raids"] = change("food raid") / food.replace(0, np.nan)
        rates[f"{side} time in shared land"] = change("steps shared") / time.replace(0, np.nan)
        rates[f"{side} time on rival land"] = change("steps rival") / time.replace(0, np.nan)
        rates[f"{side} stolen per agent per 100 steps"] = 100 * change("cumulative stolen") / change("agent steps").replace(0, np.nan)
        decisions = change("cumulative cooperate") + change("cumulative compete")
        rates[f"{side} cooperation rate"] = change("cumulative cooperate") / decisions.replace(0, np.nan)
        rates[f"{side} survived"] = (last[side] > 0).astype(float)
    rates = rates.reset_index()

    chosen, drift = final[final["condition"] == "selection"], final[final["condition"] == "neutral"]
    figure, axis = plt.subplots(figsize=(17, 7))
    positions = np.arange(len(GENES))
    width = 0.36
    rows = []
    for index, (side, color) in enumerate([("rich", LEVEL_COLOR[PLENTIFUL]), ("poor", LEVEL_COLOR[SCARCE])]):
        means, halves = [], []
        for gene in GENES:
            mean, half, _ = mean_ci(chosen[f"{side} average {gene}"])
            means.append(mean)
            halves.append(half)
        axis.bar(positions + (index - 0.5) * width, means, width, yerr=halves, capsize=3, color=color,
                 label=f"Population with the {side} homeland")
        for position, gene in zip(positions, GENES):
            mean = drift[f"{side} average {gene}"].mean()
            axis.plot([position + (index - 1) * width, position + index * width], [mean, mean], color="black",
                      linewidth=2.5, label="Neutral control" if index == 0 and gene == GENES[0] else None)
    for gene in GENES:
        paired = (chosen[f"rich average {gene}"] - chosen[f"poor average {gene}"]).dropna()
        test = stats.ttest_1samp(paired, 0) if len(paired) > 2 else None
        interval = test.confidence_interval(0.95) if test else (np.nan, np.nan)
        rows.append({"gene": LABELS[gene], "rich side": chosen[f"rich average {gene}"].mean(),
                     "poor side": chosen[f"poor average {gene}"].mean(), "rich - poor": paired.mean(),
                     "low": interval[0], "high": interval[1], "p": test.pvalue if test else np.nan,
                     "rich neutral": drift[f"rich average {gene}"].mean(), "poor neutral": drift[f"poor average {gene}"].mean()})
    axis.set_xticks(positions)
    axis.set_xticklabels([LABELS[gene] for gene in GENES], fontsize=15)
    axis.set_ylabel("Evolved gene value (0 to 1)")
    axis.set_ylim(0, 1)
    axis.grid(axis="x", visible=False)
    axis.legend(loc="upper right")
    count = chosen["seed"].nunique()
    axis.set_title("M7. Rich homeland against poor homeland in one world: what each side evolves\n"
                   f"(mean and 95% interval across {count} runs)")
    save(figure, "m7_unequal_profiles.png")
    save_table(pd.DataFrame(rows), "m7_genes_rich_vs_poor.csv")

    summary = rates.groupby("condition").mean(numeric_only=True).drop(columns="seed").T.reset_index()
    summary.columns = ["measure"] + list(summary.columns[1:])
    save_table(summary, "m7_behaviour.csv")

    # Population of each side over time.
    figure, axis = plt.subplots(figsize=(11, 6))
    selection = data[data["condition"] == "selection"]
    draw_band(axis, selection, "rich", LEVEL_COLOR[PLENTIFUL], "Rich homeland")
    draw_band(axis, selection, "poor", LEVEL_COLOR[SCARCE], "Poor homeland")
    axis.set_xlabel("Timestep")
    axis.set_ylabel("Agents alive")
    axis.set_ylim(bottom=0)
    axis.legend()
    axis.set_title("M7. Population of the rich and the poor side")
    save(figure, "m7_unequal_population.png")


# ---------------------------------------------------------------------------
# M0: competitive exclusion on an open map
# ---------------------------------------------------------------------------

def figures_m0():
    data = load("m0_timeseries.csv.gz", OPEN_GENES)
    data["both alive"] = ((data["Population A"] > 0) & (data["Population B"] > 0)).astype(float)
    data["smaller share"] = data[["Population A", "Population B"]].min(axis=1) / data["Total population"].replace(0, np.nan)
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharey=True)
    rows = []
    for axis, level in zip(axes, [SCARCE, MIDDLE, PLENTIFUL]):
        for condition, color, label, style in [("neutral", NEUTRAL, "Neutral control (no selection)", "--"),
                                               ("selection", "#b91c1c", "With selection", "-")]:
            group = data[(data["level"] == level) & (data["condition"] == condition)]
            alive = group.groupby("Timestep")["both alive"].mean()
            axis.plot(alive.index, alive.values, color=color, linestyle=style, label=label)
            last = group[group["Timestep"] == group["Timestep"].max()]
            gone = group[group["both alive"] == 0].groupby("seed")["Timestep"].min()
            rows.append({"level": level, "condition": condition, "runs": last["seed"].nunique(),
                         "runs where both survive": int(last["both alive"].sum()),
                         "median timestep of exclusion": gone.median() if len(gone) else np.nan,
                         "smaller side's share at the end": last["smaller share"].mean()})
        axis.set_title(f"{LEVEL_NAME[level]} (level {level})")
        axis.set_xlabel("Timestep")
        axis.set_ylim(0, 1.05)
    axes[0].set_ylabel("Share of runs with both populations alive")
    axes[0].legend(loc="lower left")
    figure.suptitle("M0. On an open map, evolution makes one population eliminate the other", fontsize=18)
    figure.tight_layout()
    save(figure, "m0_exclusion.png")
    save_table(pd.DataFrame(rows), "m0_exclusion.csv")


# ---------------------------------------------------------------------------
# M3: veterans of plenty against veterans of scarcity, on an open map
# ---------------------------------------------------------------------------

def figures_m3():
    data = pd.read_csv(os.path.join(RESULTS, "m3_timeseries.csv.gz"))
    plenty = np.where(data["plenty_side"] == "A", data["Population A"], data["Population B"])
    scarcity = np.where(data["plenty_side"] == "A", data["Population B"], data["Population A"])
    data["plenty share"] = plenty / pd.Series(plenty + scarcity).replace(0, np.nan).values
    data["plenty alive"], data["scarcity alive"] = plenty > 0, scarcity > 0
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharey=True)
    rows = []
    for axis, level in zip(axes, [SCARCE, MIDDLE, PLENTIFUL]):
        for mode, style in [("frozen", "-"), ("evolving", "--")]:
            group = data[(data["level"] == level) & (data["condition"] == mode)]
            draw_band(axis, group, "plenty share", "#111827" if mode == "frozen" else "#2563eb",
                      "Genomes frozen" if mode == "frozen" else "Evolution continues", style)
            last = group[group["Timestep"] == group["Timestep"].max()]
            share = last["plenty share"]
            mean, half, count = mean_ci(share)
            wins = int((share > 0.5).sum())
            sign = stats.binomtest(wins, int((share != 0.5).sum()), 0.5).pvalue if (share != 0.5).sum() else np.nan
            rows.append({"level": level, "mode": mode, "runs": count, "plenty veterans' final share": mean, "±": half,
                         "runs plenty veterans larger": wins, "runs scarcity veterans larger": int((share < 0.5).sum()),
                         "sign test p": sign, "plenty veterans died out": int((~last["plenty alive"]).sum()),
                         "scarcity veterans died out": int((~last["scarcity alive"]).sum())})
        axis.axhline(0.5, color=NEUTRAL, linewidth=1, linestyle=":")
        axis.set_title(f"{LEVEL_NAME[level]} world (level {level})")
        axis.set_xlabel("Timestep")
        axis.set_ylim(0, 1)
    axes[0].set_ylabel("Plenty veterans' share of all agents")
    axes[0].legend(loc="best")
    figure.suptitle("M3. Veterans of plenty against veterans of scarcity in one open world\n"
                    "(above 0.5 = plenty veterans winning; mean and 95% band)", fontsize=18)
    figure.tight_layout()
    save(figure, "m3_veterans.png")
    save_table(pd.DataFrame(rows), "m3_outcome.csv")


# ---------------------------------------------------------------------------
# M4: environmental shock
# ---------------------------------------------------------------------------

def figures_m4():
    data = load("m4_timeseries.csv.gz")
    styles = {"plenty then scarcity": ("#b91c1c", "-"), "scarcity then plenty": ("#2563eb", "-"),
              "always plenty": (LEVEL_COLOR[PLENTIFUL], "--"), "always scarcity": (LEVEL_COLOR[SCARCE], "--")}
    panels = [("Total population", "Agents alive"), ("gene body_size", "Body size gene"),
              ("gene vision", "Vision gene"), ("gene farming_tendency", "Farming gene"),
              ("gene cooperative_tendency", "Cooperation gene"), ("gene risk_tolerance", "Risk tolerance gene")]
    figure, axes = plt.subplots(2, 3, figsize=(19, 9.5), sharex=True)
    for axis, (column, title) in zip(axes.ravel(), panels):
        for condition, (color, style) in styles.items():
            draw_band(axis, data[data["condition"] == condition], column, color, condition.capitalize(), style)
        axis.axvline(1200, color="black", linewidth=1, linestyle=":")
        axis.set_title(title)
        if column != "Total population":
            axis.set_ylim(0, 1)
    for axis in axes[1]:
        axis.set_xlabel("Timestep (the environment switches at 1,200)")
    axes[0, 0].legend(loc="best", fontsize=11)
    figure.suptitle("M4. Environmental shock: do the genes follow the environment?", fontsize=19)
    figure.tight_layout()
    save(figure, "m4_shock.png")

    rows = []
    for condition in styles:
        group = data[data["condition"] == condition]
        row = {"condition": condition, "runs": group["seed"].nunique()}
        for step in [1200, 1300, 1400, 1800, 2400]:
            row[f"agents at {step}"] = group[group["Timestep"] == step]["Total population"].mean()
        last = group[group["Timestep"] == 2400]
        row["runs that lost a population"] = int(((last["Population A"] == 0) | (last["Population B"] == 0)).sum())
        rows.append(row)
    save_table(pd.DataFrame(rows), "m4_population.csv")
    rows = []
    for gene in GENES:
        for shocked, start_control, end_control in [("plenty then scarcity", "always plenty", "always scarcity"),
                                                    ("scarcity then plenty", "always scarcity", "always plenty")]:
            def at(condition, step):
                group = data[(data["condition"] == condition) & (data["Timestep"] > step - 100) & (data["Timestep"] <= step)]
                return group.groupby("seed")[f"gene {gene}"].mean()
            before, after = at(shocked, 1200), at(shocked, 2400)
            target, stay = at(end_control, 2400), at(start_control, 2400)
            moved = compare(after, stay)
            gap = target.mean() - stay.mean()
            rows.append({"gene": LABELS[gene], "shock": shocked, "before": before.mean(), "after": after.mean(),
                         "if it had not switched": stay.mean(), "where it was heading": target.mean(),
                         "shift vs no switch": moved["difference"], "low": moved["low"], "high": moved["high"],
                         "p": moved["p"], "share of the gap covered": moved["difference"] / gap if abs(gap) > 0.05 else np.nan})
    save_table(pd.DataFrame(rows), "m4_gene_shift.csv")


# ---------------------------------------------------------------------------
# M5: when does cooperation survive? Trade bonus against abundance.
# ---------------------------------------------------------------------------

def figures_m5(m2_final, m2_rates_source):
    data = load("m5_timeseries.csv.gz")
    seeds = sorted(data["seed"].unique())
    default = m2_rates_source[(m2_rates_source["condition"] == "selection") & m2_rates_source["level"].isin(NAMED.values())
                              & m2_rates_source["seed"].isin(seeds)].copy()
    default["bonus"] = config.MODEL_2["TRADE_BONUS"]
    data = pd.concat([data, default])
    keys = ["bonus", "level", "seed"]
    final = final_rows(data, keys)
    rates = behaviour(data, keys, STEPS // 2, STEPS)
    neutral = m2_final[m2_final["condition"] == "neutral"].groupby("level")["gene cooperative_tendency"].mean()
    bonuses = sorted(final["bonus"].unique())
    levels = [SCARCE, MIDDLE, PLENTIFUL]
    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    tables = []
    for axis, (source, column, title) in zip(axes, [(final, "gene cooperative_tendency", "Cooperation gene"),
                                                    (rates, "cooperation rate", "Share of meetings where the agent cooperated")]):
        grid = np.array([[source[(source["bonus"] == b) & (source["level"] == l)][column].mean() for b in bonuses] for l in levels])
        image = axis.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto", origin="lower")
        for i in range(len(levels)):
            for j in range(len(bonuses)):
                axis.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=16, fontweight="bold")
        axis.set_xticks(range(len(bonuses)))
        axis.set_xticklabels([f"{b:g}" for b in bonuses])
        axis.set_yticks(range(len(levels)))
        axis.set_yticklabels([LEVEL_NAME[l] for l in levels])
        axis.set_xlabel("Trade bonus each side gets when both cooperate")
        axis.set_title(title)
        axis.grid(False)
        figure.colorbar(image, ax=axis, fraction=0.046)
    axes[0].set_ylabel("Homelands")
    count = len(seeds)
    figure.suptitle(f"M5. When does cooperation survive? (mean of {count} runs per cell; drift alone gives about "
                    f"{neutral.mean():.2f} for the gene)", fontsize=17)
    figure.tight_layout()
    save(figure, "m5_cooperation_map.png")
    rows = []
    for level in levels:
        for bonus in bonuses:
            gene = final[(final["bonus"] == bonus) & (final["level"] == level)]["gene cooperative_tendency"]
            rate = rates[(rates["bonus"] == bonus) & (rates["level"] == level)]["cooperation rate"]
            mean, half, n = mean_ci(gene)
            rows.append({"level": level, "bonus": bonus, "runs": n, "cooperation gene": mean, "±": half,
                         "neutral control": neutral[level], "cooperation rate": rate.mean(),
                         "trades per agent per 100 steps": rates[(rates["bonus"] == bonus) & (rates["level"] == level)]["trades per agent per 100 steps"].mean(),
                         "agents": rates[(rates["bonus"] == bonus) & (rates["level"] == level)]["agents"].mean()})
    save_table(pd.DataFrame(rows), "m5_cooperation.csv")
    rho, p = stats.spearmanr(final["bonus"], final["gene cooperative_tendency"], nan_policy="omit")
    save_table(pd.DataFrame([{"trend of cooperation gene with bonus (rho)": rho, "p": p, "runs": len(final)}]), "m5_trend.csv")


# ---------------------------------------------------------------------------
# M6: Model 1 against Model 2
# ---------------------------------------------------------------------------

def figures_m6():
    old_tables = os.path.join(ROOT, "results", "tables")
    old_genes = pd.read_csv(os.path.join(old_tables, "e2_genes_vs_neutral.csv"))
    old_genes["separates"] = (old_genes["difference_low"] > 0) | (old_genes["difference_high"] < 0)
    old_strength = pd.read_csv(os.path.join(old_tables, "e5_selection_by_level.csv"))
    new_genes = pd.read_csv(os.path.join(TABLES, "m2_genes_vs_neutral.csv"))
    new_strength = pd.read_csv(os.path.join(TABLES, "m2_selection_strength.csv"))
    new_strength = new_strength[new_strength["condition"] == "selection"].set_index("level")
    names = [LABELS[gene] for gene in GENES]

    rows = []
    for level, group in old_genes.groupby("level"):
        strength = old_strength[old_strength["level"] == level]
        strongest = strength.loc[strength["spearman"].abs().idxmax()]
        rows.append({"model": "Model 1", "level": level, "genes": group["gene"].nunique(),
                     "genes that separate from drift": int(group["separates"].sum()),
                     "largest gene-to-offspring correlation": abs(strongest["spearman"]),
                     "gene with the largest correlation": LABELS[strongest["gene"]]})
    for level, group in new_genes.groupby("level"):
        correlations = new_strength.loc[level, names].astype(float)
        rows.append({"model": "Model 2", "level": level, "genes": group["gene"].nunique(),
                     "genes that separate from drift": int(group["separates"].sum()),
                     "largest gene-to-offspring correlation": correlations.abs().max(),
                     "gene with the largest correlation": correlations.abs().idxmax()})
    table = pd.DataFrame(rows)
    save_table(table, "m6_model1_vs_model2.csv")

    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    panels = [("genes that separate from drift", "Genes that separate from the neutral control"),
              ("largest gene-to-offspring correlation", "Largest gene-to-offspring correlation (|Spearman|)")]
    for axis, (column, title) in zip(axes, panels):
        for model, color, marker in [("Model 1", NEUTRAL, "D"), ("Model 2", "#111827", "o")]:
            group = table[table["model"] == model]
            genes = int(group["genes"].iloc[0])
            axis.plot(group["level"], group[column], color=color, marker=marker, markersize=9,
                      label=f"{model} ({genes} genes, {len(group)} levels)")
        axis.set_title(title)
        axis.set_xlabel("Abundance level (1 = scarcest; each model has its own scale)")
        axis.set_xticks(LEVELS)
        axis.set_ylim(bottom=0)
    axes[0].set_ylabel("Number of genes")
    axes[0].legend(loc="best")
    figure.suptitle("M6. Model 1 against Model 2: how much the genetic algorithm has to work with", fontsize=18)
    figure.tight_layout()
    save(figure, "m6_model1_vs_model2.png")


# ---------------------------------------------------------------------------
# M8: Raiders vs Farmers. A hand-set starting point, NOT an evolved one.
# ---------------------------------------------------------------------------

def figures_m8():
    data = pd.read_csv(os.path.join(RESULTS, "m8_timeseries.csv.gz"))
    # The "random" condition is m7's selection runs: random genes, same unequal
    # homelands. There the poor side stands where the raiders stand.
    random_start = pd.read_csv(os.path.join(RESULTS, "m7_timeseries.csv.gz"))
    random_start = random_start[random_start["condition"] == "selection"].copy()
    random_start["condition"] = "random"
    random_start["raider_side"] = np.where(random_start["rich_side"] == "A", "B", "A")
    data = pd.concat([data, random_start], ignore_index=True)

    # Relabel the two populations as "raiders" and "farmers" instead of A and B.
    for side, own in [("raiders", lambda raider: raider), ("farmers", lambda raider: "B" if raider == "A" else "A")]:
        letter = data["raider_side"].map(own)
        for column in [c for c in data.columns if c.startswith("Population A")]:
            suffix = column[len("Population A"):]
            data[side + suffix] = np.where(letter == "A", data["Population A" + suffix], data["Population B" + suffix])
    sides = ["raiders", "farmers"]
    for side in sides:
        for gene in GENES:
            data.loc[data[side] == 0, f"{side} average {gene}"] = np.nan
    keys = ["condition", "seed"]
    conditions = ["evolving", "frozen", "swapped", "random"]
    side_color = {"raiders": "#b91c1c", "farmers": "#2a7f3f"}
    start = {"raiders": config.RAIDER_GENOME, "farmers": config.FARMER_GENOME}

    # --- Do raiders stay raiders? Genes over time in the evolving condition.
    evolving = data[data["condition"] == "evolving"]
    count = evolving["seed"].nunique()
    shown = ["expansion", "cooperative_tendency", "body_size", "risk_tolerance"]
    figure, axes = plt.subplots(1, 4, figsize=(21, 5.6), sharey=True)
    for axis, gene in zip(axes, shown):
        for side in sides:
            draw_band(axis, evolving, f"{side} average {gene}", side_color[side], side.capitalize())
            axis.axhline(start[side][gene], color=side_color[side], linewidth=1.2, linestyle=":")
        axis.set_title(LABELS[gene])
        axis.set_xlabel("Timestep")
        axis.set_ylim(0, 1)
    axes[0].set_ylabel("Average gene value")
    axes[0].legend(loc="center right")
    figure.suptitle("M8. Raiders vs Farmers (hand-set start, not evolved): do raiders stay raiders?\n"
                    f"(mean and 95% band across {count} runs; dotted = the starting value)", fontsize=18)
    figure.tight_layout()
    save(figure, "m8_raider_genes.png")

    # --- Behaviour of each side, per run, over the second half of the run.
    first = data[data["Timestep"] == STEPS // 2].set_index(keys)
    last = data[data["Timestep"] == STEPS].set_index(keys)
    early = data[data["Timestep"] == 300].set_index(keys)
    late = data[data["Timestep"] >= STEPS // 2]
    measures = ["share of food from raids", "share of time on rival land",
                "stolen per agent per 100 steps", "population"]
    rates = {}
    for side in sides:
        def change(column, side=side):
            return last[f"{side} {column}"] - first[f"{side} {column}"]
        food = change("food farm") + change("food wild") + change("food raid")
        time = change("steps home") + change("steps shared") + change("steps rival")
        table = pd.DataFrame(index=last.index)
        table["share of food from raids"] = change("food raid") / food.replace(0, np.nan)
        table["share of time on rival land"] = change("steps rival") / time.replace(0, np.nan)
        table["stolen per agent per 100 steps"] = 100 * change("cumulative stolen") / change("agent steps").replace(0, np.nan)
        table["population"] = late.groupby(keys)[side].mean()
        early_food = sum(early[f"{side} food {source}"] for source in ["farm", "wild", "raid"])
        table["share of food from raids, first 300 steps"] = early[f"{side} food raid"] / early_food.replace(0, np.nan)
        table["survived"] = (last[side] > 0).astype(float)
        table["final size"] = last[side]
        for gene in shown:
            table[f"final {LABELS[gene].lower()} gene"] = final_rows(data, keys).set_index(keys)[f"{side} average {gene}"]
        rates[side] = table.reset_index()

    figure, axes = plt.subplots(1, 4, figsize=(22, 6))
    positions = np.arange(len(conditions))
    width = 0.38
    for axis, measure in zip(axes, measures):
        for index, side in enumerate(sides):
            means, halves = [], []
            for condition in conditions:
                mean, half, _ = mean_ci(rates[side][rates[side]["condition"] == condition][measure])
                means.append(mean)
                halves.append(half)
            axis.bar(positions + (index - 0.5) * width, means, width, yerr=halves, capsize=3, color=side_color[side],
                     label="Raiders (in \"random\": the poor side)" if side == "raiders"
                     else "Farmers (in \"random\": the rich side)")
        axis.set_xticks(positions)
        axis.set_xticklabels(["Evolving", "Frozen", "Swapped\nhomelands", "Random\ngenes"], fontsize=13)
        axis.set_title(measure.capitalize(), fontsize=15)
        axis.set_ylim(bottom=0)
        axis.grid(axis="x", visible=False)
    axes[2].legend(loc="upper right", fontsize=11)
    figure.suptitle("M8. Raiders vs Farmers: what each side does (second half of each run; "
                    f"mean and 95% interval across {count} runs)", fontsize=18)
    figure.tight_layout()
    save(figure, "m8_behaviour.png")

    rows = []
    for condition in conditions:
        size = {side: rates[side][rates[side]["condition"] == condition].set_index("seed")["final size"] for side in sides}
        for side in sides:
            group = rates[side][rates[side]["condition"] == condition]
            other = "farmers" if side == "raiders" else "raiders"
            row = {"condition": condition, "side": side, "runs": len(group)}
            for measure in measures + ["share of food from raids, first 300 steps"]:
                mean, half, _ = mean_ci(group[measure])
                row[measure] = mean
                row[measure + " ±"] = half
            row["runs survived"] = int(group["survived"].sum())
            row["runs ended larger"] = int((size[side] > size[other]).sum())
            for gene in shown:
                row[f"final {LABELS[gene].lower()} gene"] = group[f"final {LABELS[gene].lower()} gene"].mean()
            rows.append(row)
    save_table(pd.DataFrame(rows), "m8_raiders_vs_farmers.csv")


# Regenerate every Model 2 figure and table. Name some to make only those,
# for example: python analysis/make_figures_model2.py m6 m8
if __name__ == "__main__":
    wanted = sys.argv[1:]

    def want(name):
        return not wanted or name in wanted

    if want("m1") or want("m2") or want("m5"):
        versus, chosen, drift, strength = figures_m1_m2()
    if want("m7"):
        figures_m7()
    if want("m0"):
        figures_m0()
    if want("m3"):
        figures_m3()
    if want("m4"):
        figures_m4()
    if want("m5"):
        # figures_m5 needs m2's final gene values and m2's raw rows (for the default trade bonus).
        figures_m5(pd.concat([chosen, drift]), load("m2_timeseries.csv.gz"))
    if want("m6"):
        figures_m6()
    if want("m8"):
        figures_m8()
    if not wanted:
        path = os.path.join(TABLES, "all_tables.txt")
        with open(path, "w") as file:
            file.write("\n".join(ALL_TABLES) + "\n")
        print(f"Saved {path}")
