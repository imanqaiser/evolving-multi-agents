# Flask serves the page. Mesa handles the simulation in model.py.
# Run this file with: python app.py
from flask import Flask, redirect, render_template, request, url_for

import config
from model import SocietyModel

app = Flask(__name__)

# One simulation for this local demo. All browser tabs see this same world.
model = SocietyModel()


# A route connects a web address to a Python function.
# Opening http://127.0.0.1:5000/ calls this function.
@app.route("/")
def show_simulation():
    # Build three rows of cells for the HTML page.
    # Each cell gets its position, food amount, and list of living agents.
    grid = []
    for row in range(config.GRID_SIZE):
        cells = []
        for column in range(config.GRID_SIZE):
            occupants = []
            for agent in model.agents:
                if agent.position == (row, column):
                    occupants.append(agent)
            cells.append({
                "position": (row, column),
                "food": model.food[(row, column)],
                "capacity": model.capacity[(row, column)],
                "agents": occupants,
            })
        grid.append(cells)

    # render_template fills in the values and loops in templates/index.html.
    return render_template("index.html", model=model, grid=grid, config=config)


# POST means the button asks the server to change something.
@app.route("/step", methods=["POST"])
def next_timestep():
    # Do nothing if all agents are dead.
    if model.running:
        model.step()

    # Return to the main page. Refreshing it won't repeat the button press.
    return redirect(url_for("show_simulation"))


@app.route("/reset", methods=["POST"])
def reset_simulation():
    # Replace the old simulation with a fresh starting world.
    global model
    environment_name = request.form.get("environment", model.environment_name)
    if environment_name not in config.ENVIRONMENTS:
        return "Unknown environment preset.", 400
    model = SocietyModel(environment_name)
    print(f"Simulation reset to timestep 0: {environment_name}.", flush=True)
    return redirect(url_for("show_simulation"))


# This runs only when you type 'python app.py', not when another file imports it.
if __name__ == "__main__":
    # Handle one request at a time so two button presses cannot overlap.
    app.run(host="127.0.0.1", port=5000, threaded=False)
