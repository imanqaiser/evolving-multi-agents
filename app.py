# Flask serves the pages. Mesa handles the simulation in model.py.
# Run this file with: python app.py
from flask import Flask, jsonify, redirect, render_template, request, url_for

import config
from model import SocietyModel

app = Flask(__name__)

# One simulation for this local demo. All browser tabs see this same world.
model = SocietyModel()


# Replace the old simulation with a fresh starting world.
# Returns an error message for unknown names, or None if the reset worked.
def reset_model(environment_name, world_name):
    global model
    if environment_name not in config.ENVIRONMENTS:
        return "Unknown environment preset."
    if world_name not in config.WORLDS:
        return "Unknown world size."
    model = SocietyModel(environment_name, world_name)
    print(f"Simulation reset to timestep 0: {world_name} {environment_name}.", flush=True)
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
    return render_template("index.html", model=model, config=config)


# The small amount of information the grid needs to draw one agent.
def agent_summary(agent):
    return {
        "id": agent.unique_id,
        "name": agent.name,
        "population": agent.population,
        "position": agent.position,
        "age": agent.age,
        "resources": round(agent.resources, 2),
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
        "Reproduction threshold": config.REPRODUCTION_THRESHOLD,
        "Reproduction cost per parent": config.REPRODUCTION_COST_PER_PARENT,
        "Child starting resources": config.CHILD_STARTING_RESOURCES,
        "Minimum reproduction age": config.MIN_REPRODUCTION_AGE,
        "Reproduction cooldown": config.REPRODUCTION_COOLDOWN,
        "Mutation rate": config.MUTATION_RATE,
        "Mutation size (std)": config.MUTATION_STD,
        "Mating radius": config.MATING_RADIUS,
        "Fight cost": config.FIGHT_COST,
        "Share of resources taken in a robbery": config.STEAL_FRACTION,
        "Wild regrowth per timestep": environment["regeneration"],
        "Farm growth per timestep": environment["farm_growth"],
        "Farm capacity": config.FARM_CAPACITY,
        "Cost to plant a farm": config.PLANT_COST,
        "Resources needed to plant": config.PLANT_MIN_RESOURCES,
        "Farm abandoned after (timesteps unworked)": config.FARM_ABANDON_STEPS,
        "Random seed": config.SEED,
    }
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
            "reproduction_threshold": config.REPRODUCTION_THRESHOLD,
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
    return render_template("debug.html", model=model, grid=grid, config=config)


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
