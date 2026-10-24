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
    "LARGE": {"grid_size": 30, "agents_per_population": 30, "print_turns": False},
}

# Reproduction transfers these resources from the parents to their child.
# Keep CHILD_STARTING_RESOURCES equal to twice REPRODUCTION_COST_PER_PARENT.
REPRODUCTION_THRESHOLD = 5.0
REPRODUCTION_COST_PER_PARENT = 2.0
CHILD_STARTING_RESOURCES = 4.0
REPRODUCTION_COOLDOWN = 5
MIN_REPRODUCTION_AGE = 3
MATING_RADIUS = 1  # Parents must be within this many cells of each other.
MAX_AGE = 50
MUTATION_RATE = 0.10
MUTATION_STD = 0.05

# Confrontations between the two populations (see confront() in model.py).
# A robbery takes STEAL_FRACTION of the victim's resources. Every agent that
# keeps fighting pays FIGHT_COST, win or lose. Keeping FIGHT_COST above a
# typical robbery (STEAL_FRACTION × about 5 resources) makes fighting risky,
# so cooperating can pay off; with cheap fights, everyone evolves to fight.
FIGHT_COST = 1.5
STEAL_FRACTION = 0.25
# The responses to a rival. Avoid/ignore and raid/respect happen while moving;
# cooperate/compete happen when two agents from different populations meet.
ACTIONS = ["avoid", "ignore", "raid", "respect", "cooperate", "compete"]

# Farms (see SocietyAgent.farm in model.py). A farm belongs to the population
# of the agent that planted it, and everyone in that population may eat from it.
# Farms grow "farm_growth" food per timestep (set per environment below).
PLANT_COST = 1.0           # Resources spent to plant a farm.
PLANT_MIN_RESOURCES = 3.0  # Agents with less than this never plant.
FARM_CAPACITY = 4.0        # Most food a farm can hold.
FARM_ABANDON_STEPS = 10    # A farm nobody from its population works for this long disappears.

# Experimental defaults. Change the numbers or patch positions here.
# Wild patches are scattered and regrow slowly: enough to survive on, not to
# thrive. Bare land produces nothing until somebody plants a farm on it.
ENVIRONMENTS = {
    "ABUNDANT": {
        "starting_resources": 5.0,
        "patches": [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
        "patch_density": 0.25,
        "food_per_patch": 2.0,
        "regeneration": 0.15,
        "farm_growth": 0.7,
        "survival_cost": 0.5,
        "movement_cost": 0.25,
        "maximum_age": MAX_AGE,
    },
    "SCARCE": {
        "starting_resources": 5.0,
        "patches": [(1, 0), (1, 1), (1, 2)],
        "patch_density": 0.2,
        "food_per_patch": 2.0,
        "regeneration": 0.1,
        "farm_growth": 0.6,
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
