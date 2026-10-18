# Small, fixed starting world. Coordinates are (row, column).
GRID_SIZE = 3
STARTING_RESOURCES = 12.0
MOVEMENT_COST = 0.25
METABOLIC_COST = 0.5
MAXIMUM_AGE = 30
REGENERATION_RATE = 0.5
SEED = 42

# Zero means no food grows there. Other cells start full and grow back.
RESOURCE_CAPACITY = [
    [0.0, 2.0, 0.0],
    [2.0, 5.0, 2.0],
    [0.0, 2.0, 0.0],
]

# Name, starting position, exploratory tendency.
STARTING_AGENTS = [
    ("A1", (0, 0), 0.1),
    ("A2", (0, 2), 0.3),
    ("B1", (2, 0), 0.7),
    ("B2", (2, 2), 0.9),
]
