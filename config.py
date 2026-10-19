# Small, fixed starting world. Coordinates are (row, column).
GRID_SIZE = 3
SEED = 42
DEFAULT_ENVIRONMENT = "ABUNDANT"

# Experimental defaults. Change the numbers or patch positions here.
# Only the listed cells produce food; all other cells stay at zero.
ENVIRONMENTS = {
    "ABUNDANT": {
        "starting_resources": 5.0,
        "patches": [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
        "food_per_patch": 3.0,
        "regeneration": 0.4,
        "survival_cost": 0.5,
        "movement_cost": 0.25,
        "maximum_age": 30,
    },
    "SCARCE": {
        "starting_resources": 5.0,
        "patches": [(1, 0), (1, 1), (1, 2)],
        "food_per_patch": 2.0,
        "regeneration": 0.1,
        "survival_cost": 0.5,
        "movement_cost": 0.25,
        "maximum_age": 30,
    },
}

# Name, starting position, exploratory tendency.
STARTING_AGENTS = [
    ("A1", (0, 0), 0.1),
    ("A2", (0, 2), 0.3),
    ("B1", (2, 0), 0.7),
    ("B2", (2, 2), 0.9),
]
