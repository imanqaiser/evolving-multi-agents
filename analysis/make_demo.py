# Renders demo.mp4: a fallback for the live dashboard demo.
#
# One shared world, the same one the dashboard shows with
# "A evolved in plenty · B random": population A starts from genomes that
# already evolved for 2,000 timesteps (gene_pools/plenty_veterans.json),
# population B starts as random generation-0 agents.
#
# Run with: python analysis/make_demo.py            (needs ffmpeg; about 2 minutes)
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

LEVEL = 2             # The level where the two populations keep meeting.
SEED = 1
STEPS = 700          # One timestep per frame.
FRAMES_PER_SECOND = 16   # 700 frames / 16 = about 44 seconds.
COLORS = {"A": "#1556b8", "B": "#c2560c"}
# The dashboard's cell colours (red, green, blue from 0 to 1).
GROUND = np.array([243, 239, 230]) / 255
FOOD_EMPTY, FOOD_FULL = np.array([225, 236, 216]) / 255, np.array([61, 139, 61]) / 255
FARM_EMPTY = {"A": np.array([214, 226, 245]) / 255, "B": np.array([248, 226, 210]) / 255}
FARM_FULL = {"A": np.array([127, 159, 214]) / 255, "B": np.array([223, 159, 116]) / 255}


# The grid as a picture: barren land, wild food, and farms.
def grid_image(model):
    image = np.empty((model.grid_size, model.grid_size, 3))
    image[:] = GROUND
    for position, capacity in model.capacity.items():
        if capacity == 0:
            continue
        fraction = min(1.0, model.food[position] / capacity)
        owner = model.farms.get(position)
        empty, full = (FARM_EMPTY[owner], FARM_FULL[owner]) if owner else (FOOD_EMPTY, FOOD_FULL)
        image[position] = empty + (full - empty) * fraction
    return image


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "demo.mp4"
    with open(os.path.join(ROOT, "gene_pools", "plenty_veterans.json")) as file:
        veterans = json.load(file)
    model = SocietyModel(LEVEL, "EXPERIMENT", seed=SEED, headless=True, gene_pools={"A": veterans})

    plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
    figure = plt.figure(figsize=(12.8, 7.2), dpi=100)
    layout = figure.add_gridspec(2, 2, width_ratios=[1.25, 1], left=0.02, right=0.97, top=0.88,
                                 bottom=0.09, wspace=0.16, hspace=0.42)
    world = figure.add_subplot(layout[:, 0])
    sizes = figure.add_subplot(layout[0, 1])
    farming = figure.add_subplot(layout[1, 1])
    figure.suptitle("Evolved agents (A, blue) against random generation-0 agents (B, orange) in one world",
                    fontsize=16, fontweight="bold")

    picture = world.imshow(grid_image(model), interpolation="nearest")
    dots = world.scatter([], [], s=26, edgecolors="white", linewidths=0.4)
    world.set_xticks([])
    world.set_yticks([])
    caption = world.set_title("", fontsize=14)
    world.set_xlabel("green = wild food · blue / orange squares = farms · dots = agents", fontsize=12)

    lines = {}
    for axis, title, top in [(sizes, "Population size", None), (farming, "Average farming gene", 1)]:
        axis.set_title(title, fontsize=14, fontweight="bold")
        axis.set_xlim(0, STEPS)
        axis.grid(color="#e5e7eb")
        if top:
            axis.set_ylim(0, top)
        for population in ["A", "B"]:
            label = "A: evolved in plenty" if population == "A" else "B: random generation 0"
            lines[(axis, population)], = axis.plot([], [], color=COLORS[population], linewidth=2.4, label=label)
    sizes.set_ylim(0, 700)
    sizes.legend(loc="upper left", frameon=False, fontsize=12)
    farming.set_xlabel("Timestep")
    history = {"t": [], "A": [], "B": [], "gene A": [], "gene B": []}

    def record():
        history["t"].append(model.steps)
        for population, size in [("A", model.population_a), ("B", model.population_b)]:
            history[population].append(size)
            gene = getattr(model, f"{population}_average_farming_tendency")
            history[f"gene {population}"].append(gene if size > 0 else np.nan)

    def draw(frame):
        if frame > 0:
            model.step()
        record()
        picture.set_data(grid_image(model))
        agents = list(model.agents)
        dots.set_offsets(np.array([[a.position[1], a.position[0]] for a in agents]).reshape(-1, 2))
        dots.set_facecolors([COLORS[a.population] for a in agents])
        caption.set_text(f"Timestep {model.steps}   ·   A: {model.population_a} agents   ·   "
                         f"B: {model.population_b} agents")
        sizes.set_ylim(0, max(700, 1.1 * max(history["A"] + history["B"])))
        for population in ["A", "B"]:
            lines[(sizes, population)].set_data(history["t"], history[population])
            lines[(farming, population)].set_data(history["t"], history[f"gene {population}"])
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
