# Coordinates are (row, column).
SEED = 42
DEFAULT_ENVIRONMENT = "ABUNDANT"
DEFAULT_WORLD = "SMALL"

# SMALL is the hand-placed 3 by 3 debug world: it uses STARTING_AGENTS below
# and each environment's "patches" list, so every agent can be followed by eye.
# LARGE is for experiments: random starting agents (A in the top half, B in the
# bottom half) and random patches, where each cell becomes a patch with
# probability "patch_density".
# EXPERIMENT is the world every experiment in experiments/ uses: big enough
# that even the scarcest abundance level holds 100+ agents, so selection is
# not swamped by random drift (see DESIGN.md, "Calibration").
WORLDS = {
    "SMALL": {"grid_size": 3, "print_turns": True},
    "LARGE": {"grid_size": 30, "agents_per_population": 30, "print_turns": False},
    "EXPERIMENT": {"grid_size": 50, "agents_per_population": 60, "print_turns": False},
}

# The genome: five tendencies between 0 and 1 (see SocietyAgent in model.py).
GENES = [
    "exploratory_tendency",
    "risk_tolerance",
    "sociability",
    "cooperative_tendency",
    "farming_tendency",
]

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
FARMS_ENABLED = True       # False (as a run override) turns planting off: wild food only.

# ---------------------------------------------------------------------------
# Environments: one scarcity gradient.
# The geography is the same at every level (same share of fertile cells, same
# patch size). Only productivity changes: wild regrowth and farm growth are
# both the base value times one multiplier. Level 1 is the scarcest, level 5
# the most plentiful. The multipliers were calibrated once (see DESIGN.md,
# "Calibration") and are frozen: do not tune them to change a result.
# ---------------------------------------------------------------------------
BASE_REGENERATION = 0.15  # Wild food regrown per patch per timestep at multiplier 1.
BASE_FARM_GROWTH = 0.7    # Farm food grown per farm per timestep at multiplier 1.
ABUNDANCE_MULTIPLIERS = {1: 0.6, 2: 1.0, 3: 1.6, 4: 2.5, 5: 4.0}
MIDDLE_LEVEL = 3


# The settings for one abundance level (1 to 5).
def make_environment(level):
    multiplier = ABUNDANCE_MULTIPLIERS[level]
    return {
        "level": level,
        "multiplier": multiplier,
        "starting_resources": 5.0,
        # SMALL world only: hand-placed patches (fewer in the scarcer levels).
        "patches": [(1, 0), (1, 1), (1, 2)] if level <= 2 else [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
        "patch_density": 0.25,
        "food_per_patch": 2.0,
        "regeneration": BASE_REGENERATION * multiplier,
        "farm_growth": BASE_FARM_GROWTH * multiplier,
        "survival_cost": 0.5,
        "movement_cost": 0.25,
        "maximum_age": MAX_AGE,
    }


# ABUNDANT and SCARCE are aliases for the two ends of the gradient.
ENVIRONMENTS = {"ABUNDANT": make_environment(5), "SCARCE": make_environment(1)}
for _level in ABUNDANCE_MULTIPLIERS:
    ENVIRONMENTS[f"LEVEL_{_level}"] = make_environment(_level)


# Accept "ABUNDANT", "LEVEL_3", or just the number 3.
def environment_name(name_or_level):
    if isinstance(name_or_level, int):
        return f"LEVEL_{name_or_level}"
    return name_or_level


# SMALL world only: name, starting position, exploratory tendency.
STARTING_AGENTS = [
    ("A1", (0, 0), 0.1),
    ("A2", (0, 2), 0.3),
    ("B1", (2, 0), 0.7),
    ("B2", (2, 2), 0.9),
]


# ---------------------------------------------------------------------------
# Model 2 (see DESIGN.md, "Model 2"). Every new mechanic sits behind a switch.
# With the defaults below every switch is off and the model is exactly Model 1.
# SocietyModel(..., version=2) turns them all on by applying MODEL_2 (at the
# end of this section) as overrides.
# ---------------------------------------------------------------------------
# 1. Two body genes with built-in costs.
BODY_GENES = False
BODY_GENE_NAMES = ["vision", "body_size"]
VISION_MIN_RADIUS = 1   # vision gene 0 sees 1 cell in every direction (as in Model 1) ...
VISION_MAX_RADIUS = 4   # ... and vision gene 1 sees 4 cells.
VISION_COST = 0.0       # Extra living cost per timestep for each cell of sight radius above 1.
BODY_COST = 0.0         # Extra living cost per timestep for body_size 1 (in proportion below that).
BODY_BASE = 0.5         # Strength = BODY_BASE + body_size: the largest agent is 3 times as strong as the smallest.

# 2. Explicit fitness and fitness-based selection.
#   "endogenous"  Model 1: partners pair up in random order
#   "fitness"     Model 2: an agent picks its partner by a tournament on fitness
#   "neutral"     control: behaviour ignores the genes and partners are picked at random
SELECTION_MODE = "endogenous"
TOURNAMENT_SIZE = 3

# 3. Soil that wears out. Fertility (0 to 1) multiplies a farm's growth.
SOIL_ENABLED = False
SOIL_DEPLETION = 0.0    # Fertility lost each timestep a farm is harvested.
SOIL_RECOVERY = 0.0     # Fertility regained each timestep the cell is not harvested.

# 4. Cooperation that can pay: when both agents cooperate, each gains TRADE_BONUS.
TRADE_BONUS = 0.0
TRADE_COOLDOWN = 5      # An agent can collect a trade bonus at most once in this many timesteps.

# Documented fallback for Gate 1 (not used unless the gate needs it): start
# both populations mixed over the whole grid instead of in separate halves.
INTERLEAVED_START = False

# PROPOSAL under test, off by default (None): homelands with a shared border
# zone. See SocietyModel.may_enter(). A number is the width of the zone, in
# rows, that each population may enter on the other side of the middle line.
HOMELAND_BAND = None

# 5. Model 2's own abundance scale: seven levels on a ratio scale.
# PLACEHOLDER values until calibration (experiments/m_calibrate.py) freezes them.
M2_MULTIPLIERS = {1: 1.0, 2: 1.4, 3: 2.0, 4: 2.8, 5: 4.0, 6: 5.6, 7: 8.0}
M2_NAMED_LEVELS = {"SCARCE": 1, "MIDDLE": 4, "PLENTIFUL": 7}

for _level, _multiplier in M2_MULTIPLIERS.items():
    ENVIRONMENTS[f"M2_LEVEL_{_level}"] = make_environment(MIDDLE_LEVEL)
    ENVIRONMENTS[f"M2_LEVEL_{_level}"].update(
        level=_level, multiplier=_multiplier, regeneration=BASE_REGENERATION * _multiplier,
        farm_growth=BASE_FARM_GROWTH * _multiplier,
        patches=[(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)])
for _name, _level in M2_NAMED_LEVELS.items():
    ENVIRONMENTS[f"M2_{_name}"] = ENVIRONMENTS[f"M2_LEVEL_{_level}"]


# Model 2 version of environment_name(): a level number 1-7, "SCARCE",
# "MIDDLE", "PLENTIFUL", or a full name such as "M2_LEVEL_3".
def model2_environment_name(name_or_level):
    if isinstance(name_or_level, int):
        return f"M2_LEVEL_{name_or_level}"
    if name_or_level in M2_NAMED_LEVELS:
        return f"M2_LEVEL_{M2_NAMED_LEVELS[name_or_level]}"
    if name_or_level in ("ABUNDANT", "LEVEL_5"):   # The dashboard's default name.
        return f"M2_LEVEL_{M2_NAMED_LEVELS['MIDDLE']}"
    return name_or_level


# Everything version=2 switches on. PLACEHOLDER numbers until calibration.
MODEL_2 = {
    "BODY_GENES": True,
    "VISION_COST": 0.05,
    "BODY_COST": 0.2,
    "SELECTION_MODE": "fitness",
    "MATING_RADIUS": 3,
    "REPRODUCTION_THRESHOLD": 6.0,
    "REPRODUCTION_COST_PER_PARENT": 3.0,
    "CHILD_STARTING_RESOURCES": 6.0,
    "SOIL_ENABLED": True,
    "SOIL_DEPLETION": 0.02,
    "SOIL_RECOVERY": 0.01,
    "TRADE_BONUS": 0.5,
}
