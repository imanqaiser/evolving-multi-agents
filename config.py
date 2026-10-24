# Coordinates are (row, column).
SEED = 42
DEFAULT_ENVIRONMENT = "ABUNDANT"
DEFAULT_WORLD = "SMALL"

# SMALL is the hand-placed 3 by 3 debug world: it uses STARTING_AGENTS below
# and each environment's "patches" list, so every agent can be followed by eye.
# LARGE is for experiments: random starting agents (A in the top half, B in the
# bottom half) and random patches, where each cell becomes a patch with
# probability "patch_density".
WORLDS = {
    "SMALL": {"grid_size": 3, "print_turns": True},
    "LARGE": {"grid_size": 20, "agents_per_population": 15, "print_turns": False},
}

# Reproduction transfers these resources from the parents to their child.
# Keep CHILD_STARTING_RESOURCES equal to twice REPRODUCTION_COST_PER_PARENT.
REPRODUCTION_THRESHOLD = 5.0
REPRODUCTION_COST_PER_PARENT = 2.0
CHILD_STARTING_RESOURCES = 4.0
REPRODUCTION_COOLDOWN = 5
MIN_REPRODUCTION_AGE = 3
MAX_AGE = 50
MUTATION_RATE = 0.10
MUTATION_STD = 0.05

# Experimental defaults. Change the numbers or patch positions here.
# Only patch cells produce food; all other cells stay at zero.
ENVIRONMENTS = {
    "ABUNDANT": {
        "starting_resources": 5.0,
        "patches": [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
        "patch_density": 0.55,
        "food_per_patch": 3.0,
        "regeneration": 0.4,
        "survival_cost": 0.5,
        "movement_cost": 0.25,
        "maximum_age": MAX_AGE,
    },
    "SCARCE": {
        "starting_resources": 5.0,
        "patches": [(1, 0), (1, 1), (1, 2)],
        "patch_density": 0.4,
        "food_per_patch": 2.0,
        # 0.1 made extinction certain in every world; 0.25 is harsh but survivable.
        "regeneration": 0.25,
        "survival_cost": 0.5,
        "movement_cost": 0.25,
        "maximum_age": MAX_AGE,
    },
}

# SMALL world only: name, starting position, exploratory tendency.
STARTING_AGENTS = [
    ("A1", (0, 0), 0.1),
    ("A2", (0, 2), 0.3),
    ("B1", (2, 0), 0.7),
    ("B2", (2, 2), 0.9),
]
