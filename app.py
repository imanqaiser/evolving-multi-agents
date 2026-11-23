# Flask serves the pages. Mesa handles the simulation in model.py.
# Run this file with: python app.py
import json
import os

from flask import Flask, jsonify, redirect, render_template, request, url_for

import config
from model import SocietyModel

app = Flask(__name__)

# One simulation for this local demo. All browser tabs see this same world.
model = SocietyModel()

# The "Starting agents" menu. "random" gives every agent random genes
# (generation 0). The other choices start a population from a saved gene pool:
# genomes that already evolved for 2,000 timesteps in experiment E2, at the
# most plentiful level ("plenty veterans") or the scarcest ("scarcity veterans").
# Each entry: label, then which pool population A and population B start from.
START_OPTIONS = {
    "random": ("Random (generation 0)", None, None),
    "plenty_vs_random": ("A evolved in plenty · B random", "plenty_veterans", None),
    "scarcity_vs_random": ("A evolved in scarcity · B random", "scarcity_veterans", None),
    "plenty_vs_scarcity": ("A evolved in plenty · B evolved in scarcity", "plenty_veterans", "scarcity_veterans"),
}
start_name = "random"

# Model 2's "Starting agents" menu (three-zone map, ZONED world). Each entry:
# label, pool for A, pool for B, and each population's homeland level (None =
# both use the level chosen on the page). "Raiders vs Farmers" is a DESIGNED
# scenario: its genomes were written by hand in config.py, they did not evolve.
M2_START_OPTIONS = {
    "random": ("Random genes (generation 0)", None, None, None),
    "scarcity_vs_plenty": ("A evolved in scarcity · B evolved in plenty",
                           "m2_scarcity_veterans", "m2_plenty_veterans", None),
    "raiders_vs_farmers": ("Raiders (A, poor homeland) vs Farmers (B, rich homeland): designed scenario, not evolved",
                           "raiders", "farmers",
                           {"A": config.RAIDER_HOMELAND_LEVEL, "B": config.FARMER_HOMELAND_LEVEL}),
}
LEVEL_NAMES = {level: name for name, level in config.M2_NAMED_LEVELS.items()}


def start_options(version):
    return M2_START_OPTIONS if version == 2 else START_OPTIONS

# Menu text for the environments: the five abundance levels and the two aliases.
ENVIRONMENT_LABELS = {}
for name, environment in config.ENVIRONMENTS.items():
    if name.startswith("M2_"):
        continue  # Model 2 has its own level menu.
    text = name.replace("_", " ").capitalize()
    ENVIRONMENT_LABELS[name] = f"{text} (level {environment['level']}, productivity ×{environment['multiplier']:g})"


# Read a saved gene pool (a list of genomes) from the gene_pools folder.
def load_pool(name):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gene_pools", f"{name}.json")
    with open(path) as file:
        return json.load(file)


# Replace the old simulation with a fresh starting world.
# Returns an error message for unknown names, or None if the reset worked.
def reset_model(environment_name, world_name, seed=None, start=None, version=None, level=None):
    global model, start_name
    # version 1 is Model 1 (environment and world from the menus). version 2 is
    # Model 2 on the three-zone map: always the ZONED world, with a homeland level 1-7.
    version = model.version if version is None else int(version)
    if version not in (1, 2):
        return "Unknown model version."
    options = start_options(version)
    start = start or start_name
    if start not in options:
        if version == model.version:
            return "Unknown starting population."
        start = "random"  # The other model's menu has different choices.
    try:
        seed = model.seed if seed is None else int(seed)
    except (TypeError, ValueError):
        return "The seed must be a whole number."
    gene_pools = {}
    for population, pool_name in zip(["A", "B"], options[start][1:3]):
        if pool_name:
            try:
                gene_pools[population] = load_pool(pool_name)
            except FileNotFoundError:
                return f"gene_pools/{pool_name}.json is missing. Run: python experiments/m_experiments.py pools"
    if version == 2:
        try:
            level = int(level) if level is not None else (
                model.environment["level"] if model.version == 2 else config.M2_NAMED_LEVELS["MIDDLE"])
        except (TypeError, ValueError):
            return "The homeland level must be a number from 1 to 7."
        if level not in config.M2_MULTIPLIERS:
            return "The homeland level must be a number from 1 to 7."
        overrides = {"HOMELAND_LEVELS": options[start][3]} if options[start][3] else None
        model = SocietyModel(level, "ZONED", seed=seed, gene_pools=gene_pools, version=2, zones=True,
                             overrides=overrides)
    else:
        if model.version == 2 and environment_name not in ENVIRONMENT_LABELS:
            environment_name, world_name = config.DEFAULT_ENVIRONMENT, "LARGE"
        if environment_name not in ENVIRONMENT_LABELS:
            return "Unknown environment preset."
        if world_name not in config.WORLDS:
            return "Unknown world size."
        model = SocietyModel(environment_name, world_name, seed=seed, gene_pools=gene_pools)
    start_name = start
    print(f"Simulation reset to timestep 0: Model {version}, {model.world_name} {model.environment_name}, "
          f"seed {seed}, start {start}.", flush=True)
    return None


# ---------------------------------------------------------------------------
# Dashboard: the main page. Its JavaScript (static/dashboard.js) asks the
# /api/... routes below for the current state as JSON, so the page never
# reloads while the simulation runs.
# ---------------------------------------------------------------------------

# A route connects a web address to a Python function.
# Opening http://127.0.0.1:5000/ calls this function.
@app.route("/")
def show_dashboard():
    # Lists of (name, label), because the order of the menu matters.
    menus = {version: [(name, option[0]) for name, option in start_options(version).items()] for version in (1, 2)}
    levels = {level: f"Level {level} (×{multiplier:g})" + (f" {LEVEL_NAMES[level]}" if level in LEVEL_NAMES else "")
              for level, multiplier in config.M2_MULTIPLIERS.items()}
    return render_template("index.html", model=model, config=config, start_menus=menus,
                           start_name=start_name, environment_labels=ENVIRONMENT_LABELS, levels=levels)


# The small amount of information the grid needs to draw one agent.
def agent_summary(agent):
    return {
        "id": agent.unique_id,
        "name": agent.name,
        "population": agent.population,
        "position": agent.position,
        "age": agent.age,
        "resources": round(agent.resources, 2),
        # Model 2: body size (marker size) and whether it stands on the rival homeland.
        "body": round(agent.genome["body_size"], 2) if model.body_genes else None,
        "away": bool(model.zones and model.row_zone[agent.position[0]] not in ("S", agent.population)),
    }


# Everything the inspector panel shows about one agent, alive or dead.
def agent_details(agent):
    # Look up the parents by ID so the page can show their names.
    parents = []
    if agent.parent_ids:
        for other in model.all_agents:
            if other.unique_id in agent.parent_ids:
                parents.append({"id": other.unique_id, "name": other.name, "alive": other.alive})
    details = agent_summary(agent)
    details.update({
        "alive": agent.alive,
        "status": agent.status,
        "generation": agent.generation,
        "genome": agent.genome,
        "parents": parents,
        "next_reproduction_step": agent.next_reproduction_step,
        "reproduction_status": agent.reproduction_status(),
        "last_action": agent.last_action,
    })
    if model.version == 2:
        details["fitness"] = round(agent.fitness(), 3)
        details["sight_radius"] = agent.sight_radius
    return details


# Build the JSON the dashboard draws from.
# "since" is the last timestep the page already has, so only newer history
# rows and events are sent. "selected_id" is the agent shown in the inspector.
def build_state(since, selected_id):
    food = []
    capacity = []
    farms = []  # "A", "B", or None (no farm) for every cell.
    for row in range(model.grid_size):
        food_row = []
        capacity_row = []
        farm_row = []
        for column in range(model.grid_size):
            food_row.append(round(model.food[(row, column)], 2))
            capacity_row.append(model.capacity[(row, column)])
            farm_row.append(model.farms.get((row, column)))
        food.append(food_row)
        capacity.append(capacity_row)
        farms.append(farm_row)

    # Soil fertility per cell (Model 2), or None for a cell where nothing grows.
    fertility = None
    if model.soil:
        fertility = []
        for row in range(model.grid_size):
            fertility_row = []
            for column in range(model.grid_size):
                value = model.fertility.get((row, column))
                fertility_row.append(None if value is None else round(value, 2))
            fertility.append(fertility_row)
    zones = None
    if model.zones:
        zones = {"shared_first": model.shared_first, "shared_last": model.shared_last}

    agents = []
    for agent in model.agents:
        agents.append(agent_summary(agent))

    # DataCollector keeps one list per column, with row t = timestep t.
    history = {}
    for column, values in model.datacollector.model_vars.items():
        history[column] = values[since + 1:]

    events = []
    for event in model.events:
        if event["timestep"] > since:
            events.append(event)

    selected = None
    for agent in model.all_agents:
        if agent.unique_id == selected_id:
            selected = agent_details(agent)

    environment = model.environment
    parameters = {
        "Grid size": f"{model.grid_size} × {model.grid_size}",
        "Starting resources": environment["starting_resources"],
        "Wild patch capacity": environment["food_per_patch"],
        "Survival cost": environment["survival_cost"],
        "Movement cost": environment["movement_cost"],
        "Maximum age": environment["maximum_age"],
        "Reproduction threshold": model.param("REPRODUCTION_THRESHOLD"),
        "Reproduction cost per parent": model.param("REPRODUCTION_COST_PER_PARENT"),
        "Child starting resources": model.param("CHILD_STARTING_RESOURCES"),
        "Minimum reproduction age": config.MIN_REPRODUCTION_AGE,
        "Reproduction cooldown": config.REPRODUCTION_COOLDOWN,
        "Mutation rate": config.MUTATION_RATE,
        "Mutation size (std)": config.MUTATION_STD,
        "Mating radius": model.param("MATING_RADIUS"),
        "Fight cost": config.FIGHT_COST,
        "Share of resources taken in a robbery": config.STEAL_FRACTION,
        "Wild regrowth per timestep": environment["regeneration"],
        "Farm growth per timestep": environment["farm_growth"],
        "Farm capacity": config.FARM_CAPACITY,
        "Cost to plant a farm": config.PLANT_COST,
        "Resources needed to plant": config.PLANT_MIN_RESOURCES,
        "Farm abandoned after (timesteps unworked)": config.FARM_ABANDON_STEPS,
        "Random seed": model.seed,
        "Starting agents": start_options(model.version)[start_name][0],
    }
    if model.version == 2:
        levels = model.param("HOMELAND_LEVELS")
        parameters.update({
            "Wild regrowth per timestep": f"{environment['regeneration']} (shared land only)",
            "Farm growth per timestep": f"A {model.farm_growth['A']:.2f} · B {model.farm_growth['B']:.2f}",
            "Model": "2 (three zones, eight genes)",
            "Homeland level": f"A {levels['A']} · B {levels['B']}" if levels else environment["level"],
            "Rows of shared land": model.param("SHARED_ROWS"),
            "Deepest raid (rows into the rival homeland)": model.param("EXPANSION_MAX_ROWS"),
            "Defender's advantage in a fight": f"×{model.param('DEFENDER_ADVANTAGE'):g}",
            "Cost of vision (per extra cell of sight)": model.param("VISION_COST"),
            "Cost of body size": model.param("BODY_COST"),
            "Tournament size (partner choice)": model.param("TOURNAMENT_SIZE"),
            "Soil lost per harvest · regained per rest": f"{model.param('SOIL_DEPLETION')} · {model.param('SOIL_RECOVERY')}",
            "Trade bonus": model.param("TRADE_BONUS"),
        })
    if model.world_name != "SMALL":
        parameters["Fertile land (share of cells)"] = environment["patch_density"]
        parameters["Starting agents per population"] = model.world["agents_per_population"]

    return {
        # id() is different for every SocietyModel, so the page can tell
        # when the world was reset (for example from another browser tab).
        "run_id": id(model),
        "timestep": model.steps,
        "running": model.running,
        "world": model.world_name,
        "environment": model.environment_name,
        "seed": model.seed,
        "start": start_name,
        "start_label": start_options(model.version)[start_name][0],
        "version": model.version,
        "level": environment["level"],
        "genes": model.genes,
        "zones": zones,
        "fertility": fertility,
        "grid_size": model.grid_size,
        "food": food,
        "capacity": capacity,
        "farms": farms,
        "agents": agents,
        "stats": {
            "population_a": model.population_a,
            "population_b": model.population_b,
            "births_this_step": model.births_this_step,
            "cumulative_births": model.cumulative_births,
            "starvation_deaths": model.starvation_deaths,
            "old_age_deaths": model.old_age_deaths,
            "highest_generation": model.highest_generation,
            "average_generation": model.average_generation,
            "encounters_per_timestep": model.encounters_per_timestep,
            "cumulative_encounters": model.cumulative_encounters,
            "environmental_resources": model.environmental_resources,
            "average_resources": model.average_resources,
            "minimum_reproduction_age": config.MIN_REPRODUCTION_AGE,
            "reproduction_threshold": model.param("REPRODUCTION_THRESHOLD"),
            "cumulative_actions": model.cumulative_actions,
            "fights_this_step": model.fights_this_step,
            "cumulative_fights": model.cumulative_fights,
            "cumulative_stolen": model.cumulative_stolen,
            "farms_a": model.A_farms,
            "farms_b": model.B_farms,
        },
        "history": history,
        "events": events,
        "selected": selected,
        "parameters": parameters,
    }


@app.route("/api/state")
def get_state():
    since = request.args.get("since", -1, type=int)
    selected_id = request.args.get("selected", None, type=int)
    return jsonify(build_state(since, selected_id))


# Run one or more timesteps. Running a few per request keeps high speeds smooth.
@app.route("/api/step", methods=["POST"])
def step_api():
    data = request.get_json(silent=True) or {}
    count = max(1, min(int(data.get("steps", 1)), 50))
    for _ in range(count):
        # Do nothing if all agents are dead.
        if model.running:
            model.step()
    return jsonify(build_state(int(data.get("since", -1)), data.get("selected")))


@app.route("/api/reset", methods=["POST"])
def reset_api():
    data = request.get_json(silent=True) or {}
    error = reset_model(
        data.get("environment", model.environment_name),
        data.get("world", model.world_name),
        data.get("seed"),
        data.get("start"),
        data.get("version"),
        data.get("level"),
    )
    if error:
        return error, 400
    return jsonify(build_state(-1, None))


# ---------------------------------------------------------------------------
# Debug page: the original server-rendered page with the full turn-by-turn
# explanations and every agent in one table. Useful in the SMALL world.
# ---------------------------------------------------------------------------

@app.route("/debug")
def show_simulation():
    # Build one list of cells per grid row for the HTML page.
    # Each cell gets its position, food amount, and list of living agents.
    grid = []
    for row in range(model.grid_size):
        cells = []
        for column in range(model.grid_size):
            cells.append({
                "position": (row, column),
                "food": model.food[(row, column)],
                "capacity": model.capacity[(row, column)],
                "agents": model.cell_agents[(row, column)],
            })
        grid.append(cells)

    # render_template fills in the values and loops in templates/debug.html.
    return render_template("debug.html", model=model, grid=grid, config=config,
                           environment_labels=ENVIRONMENT_LABELS)


# POST means the button asks the server to change something.
@app.route("/step", methods=["POST"])
def next_timestep():
    # Do nothing if all agents are dead.
    if model.running:
        model.step()

    # Return to the debug page. Refreshing it won't repeat the button press.
    return redirect(url_for("show_simulation"))


@app.route("/reset", methods=["POST"])
def reset_simulation():
    error = reset_model(
        request.form.get("environment", model.environment_name),
        request.form.get("world", model.world_name),
    )
    if error:
        return error, 400
    return redirect(url_for("show_simulation"))


# This runs only when you type 'python app.py', not when another file imports it.
if __name__ == "__main__":
    # Handle one request at a time so two button presses cannot overlap.
    app.run(host="127.0.0.1", port=5000, threaded=False)
