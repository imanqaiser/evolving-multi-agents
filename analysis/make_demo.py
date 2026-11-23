# Renders demo.mp4: the "Raiders vs Farmers" run of Model 2, the same one the
# dashboard shows with Model 2 and "Raiders vs Farmers".
#
# This is a designed scenario, not an evolved one: the starting genomes were
# written by hand (config.RAIDER_GENOME and config.FARMER_GENOME). Population A,
# the raiders, has the poor homeland at the top; population B, the farmers,
# has the rich homeland at the bottom.
#
# Run with: python analysis/make_demo.py            (needs ffmpeg; about 3 minutes)
#           python analysis/make_demo.py demo.gif   (no ffmpeg needed, larger file)
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config  # noqa: E402
from model import SocietyModel  # noqa: E402

SEED = 3
STEPS = 720               # One timestep per frame.
FRAMES_PER_SECOND = 16    # 720 frames / 16 = 45 seconds.
WINDOW = 25               # The charts show the last 25 timesteps.
COLORS = {"A": "#1556b8", "B": "#c2560c"}
SOURCES = {"farm": ("#2a7f3f", "Own farms"), "wild": ("#d4a017", "Wild food"), "raid": ("#b91c1c", "Raided from farmers")}
# The dashboard's cell colours (red, green, blue from 0 to 1).
GROUND = np.array([243, 239, 230]) / 255
FOOD_EMPTY, FOOD_FULL = np.array([225, 236, 216]) / 255, np.array([61, 139, 61]) / 255
FARM_EMPTY = {"A": np.array([214, 226, 245]) / 255, "B": np.array([248, 226, 210]) / 255}
FARM_FULL = {"A": np.array([127, 159, 214]) / 255, "B": np.array([223, 159, 116]) / 255}
HOMELAND = {"A": np.array([21, 86, 184]) / 255, "B": np.array([194, 86, 12]) / 255}


def load_pool(name):
    with open(os.path.join(ROOT, "gene_pools", f"{name}.json")) as file:
        return json.load(file)


# The grid as a picture: the two tinted homelands, wild food, and farms.
def grid_image(model):
    image = np.empty((model.grid_size, model.grid_size, 3))
    image[:] = GROUND
    for row, zone in enumerate(model.row_zone):
        if zone != "S":
            image[row] = 0.9 * GROUND + 0.1 * HOMELAND[zone]
    for position, capacity in model.capacity.items():
        if capacity == 0:
            continue
        fraction = min(1.0, model.food[position] / capacity)
        owner = model.farms.get(position)
        empty, full = (FARM_EMPTY[owner], FARM_FULL[owner]) if owner else (FOOD_EMPTY, FOOD_FULL)
        image[position] = empty + (full - empty) * fraction
    return image


# Change of a running total over the last WINDOW timesteps, at every timestep.
def recent(values):
    values = np.array(values, dtype=float)
    earlier = np.concatenate([np.full(WINDOW, values[0]), values])[:len(values)]
    return values - earlier


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "demo.mp4"
    model = SocietyModel(
        config.M2_NAMED_LEVELS["MIDDLE"], "ZONED", seed=SEED, headless=True, version=2, zones=True,
        gene_pools={"A": load_pool("raiders"), "B": load_pool("farmers")},
        overrides={"HOMELAND_LEVELS": {"A": config.RAIDER_HOMELAND_LEVEL, "B": config.FARMER_HOMELAND_LEVEL}})

    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    figure = plt.figure(figsize=(12.8, 7.2), dpi=100)
    layout = figure.add_gridspec(3, 2, width_ratios=[1.25, 1], left=0.02, right=0.97, top=0.88,
                                 bottom=0.08, wspace=0.16, hspace=0.62)
    world = figure.add_subplot(layout[:, 0])
    sizes = figure.add_subplot(layout[0, 1])
    food = figure.add_subplot(layout[1, 1])
    stolen = figure.add_subplot(layout[2, 1])
    figure.suptitle("Raiders (A, blue, poor homeland) vs Farmers (B, orange, rich homeland)\n"
                    "A designed starting point, not an evolved one", fontsize=15, fontweight="bold")

    picture = world.imshow(grid_image(model), interpolation="nearest")
    dots = world.scatter([], [])
    for row in [model.shared_first - 0.5, model.shared_last + 0.5]:
        world.axhline(row, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    edge = model.grid_size - 1.5
    for row, text in [(1.5, "Homeland of the raiders"), (model.shared_first + 1, "Shared land (wild food)"),
                      (model.shared_last + 2, "Homeland of the farmers")]:
        world.text(edge, row, text, ha="right", va="top", fontsize=10, color="#374151", zorder=5,
                   bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2})
    world.set_xticks([])
    world.set_yticks([])
    caption = world.set_title("", fontsize=11)
    world.set_xlabel("dots = agents (bigger = bigger body) · red ring = standing on the rival homeland", fontsize=11)

    lines = {}
    for population, label in [("A", "Raiders"), ("B", "Farmers")]:
        lines[("size", population)], = sizes.plot([], [], color=COLORS[population], linewidth=2.2, label=label)
        lines[("stolen", population)], = stolen.plot([], [], color=COLORS[population], linewidth=2.2, label=label)
    for source, (color, label) in SOURCES.items():
        lines[("food", source)], = food.plot([], [], color=color, linewidth=2.2, label=label)
    sizes.set_title("Population size", fontsize=13, fontweight="bold")
    food.set_title(f"Where the raiders' food comes from (%, last {WINDOW} steps)", fontsize=13, fontweight="bold")
    stolen.set_title(f"Resources stolen from the other side (last {WINDOW} steps)", fontsize=13, fontweight="bold")
    food.set_ylim(0, 100)
    for axis in [sizes, food, stolen]:
        axis.set_xlim(0, STEPS)
        axis.grid(color="#e5e7eb")
    sizes.legend(loc="upper right", frameon=False, fontsize=10, ncol=2)
    food.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=9, ncol=3)
    stolen.set_xlabel("Timestep")
    # Running totals at every timestep.
    history = {"t": [], "A": [], "B": [], "stolen A": [], "stolen B": [], "farm": [], "wild": [], "raid": []}

    def record():
        history["t"].append(model.steps)
        counts = {"A": 0, "B": 0}
        for agent in model.agents:
            counts[agent.population] += 1
        for population in "AB":
            history[population].append(counts[population])
            history[f"stolen {population}"].append(model.cumulative_stolen[population])
        for source in SOURCES:
            history[source].append(model.food_from["A"][source])

    def draw(frame):
        if frame > 0:
            model.step()
        record()
        picture.set_data(grid_image(model))
        agents = list(model.agents)
        away = [model.row_zone[a.position[0]] not in ("S", a.population) for a in agents]
        dots.set_offsets(np.array([[a.position[1], a.position[0]] for a in agents]).reshape(-1, 2))
        dots.set_sizes([10 + 34 * a.genome["body_size"] for a in agents])
        dots.set_facecolors([COLORS[a.population] for a in agents])
        dots.set_edgecolors(["#e11d1d" if outside else "white" for outside in away])
        dots.set_linewidths([1.6 if outside else 0.3 for outside in away])
        caption.set_text(f"Timestep {model.steps} · raiders {history['A'][-1]} · farmers {history['B'][-1]}"
                         f" · raiders on farmer land: {sum(away)}")
        steps = history["t"]
        gained = {source: recent(history[source]) for source in SOURCES}
        total = gained["farm"] + gained["wild"] + gained["raid"]
        total[total == 0] = np.nan
        for source in SOURCES:
            lines[("food", source)].set_data(steps, 100 * gained[source] / total)
        taken = {population: recent(history[f"stolen {population}"]) for population in "AB"}
        for population in "AB":
            lines[("size", population)].set_data(steps, history[population])
            lines[("stolen", population)].set_data(steps, taken[population])
        sizes.set_ylim(0, 1.25 * max(history["A"] + history["B"]))
        stolen.set_ylim(0, max(10.0, 1.15 * max(taken["A"].max(), taken["B"].max())))
        if frame % 100 == 0:
            print(f"  frame {frame}/{STEPS}", flush=True)

    movie = animation.FuncAnimation(figure, draw, frames=STEPS + 1, repeat=False)
    path = os.path.join(ROOT, output)
    if output.endswith(".gif"):
        movie.save(path, writer=animation.PillowWriter(fps=FRAMES_PER_SECOND))
    else:
        movie.save(path, writer=animation.FFMpegWriter(
            fps=FRAMES_PER_SECOND, extra_args=["-crf", "27", "-pix_fmt", "yuv420p"]))
    print(f"Saved {output}")
