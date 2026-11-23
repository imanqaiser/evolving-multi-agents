# Mesa provides the Model and Agent base classes and records experiment data.
import mesa
from mesa.datacollection import DataCollector

import config


class SocietyAgent(mesa.Agent):
    # __init__ sets up an agent when we create it.
    def __init__(self, model, name, position, exploration=None, genome=None,
                 generation=0, parent_ids=None):
        # genome=None gives a starting agent random genes. A starting agent can
        # also be handed a genome from a saved gene pool. Children (parent_ids
        # given) always receive the genome made by make_child_genome().
        # super() calls Mesa's setup code. Mesa assigns an ID and registers the agent.
        super().__init__(model)
        self.name = name
        self.population = name[0]  # A1 and A2 belong to A; B1 and B2 belong to B.
        self.alive = True  # Set before position, because the position setter checks it.
        self.position = position
        self.resources = model.environment["starting_resources"]
        self.age = 0
        self.generation = generation
        self.parent_ids = parent_ids
        # A timestep number avoids having to subtract from a counter each turn.
        self.next_reproduction_step = 0
        self.status = "Alive"
        self.last_action = "Starting position; no action yet."
        # What happens during a turn is collected in turn_... values (see
        # start_turn) and applied to resources in finish_turn.
        self.start_turn()
        # Lifetime totals, written to model.lifetime_records when the agent dies.
        self.birth_step = model.steps
        self.total_food = 0.0    # Food eaten.
        self.total_stolen = 0.0  # Taken from the other population (robberies and farm raids).
        self.total_lost = 0.0    # Robbed from this agent.
        self.total_robbed = 0.0  # Robbed by this agent (part of total_stolen; farm raids are not).
        self.total_trade = 0.0   # Trade bonuses collected (Model 2).
        self.next_trade_step = 0
        self.offspring = 0       # Children produced: the agent's realized fitness.
        # A genome is a dictionary of five tendencies between 0 and 1 (config.GENES):
        #   exploratory_tendency  chance of exploring when nearby food is poor (step)
        #   risk_tolerance        chance of fighting on instead of backing down (confront)
        #   sociability           chance of ignoring rivals instead of avoiding them (step)
        #   cooperative_tendency  chance of leaving a rival alone instead of robbing it
        #                         (confront) and of respecting rival farms (step)
        #   farming_tendency      chance of staying to tend its population's farm (step)
        #                         and of planting a new farm (farm)
        if genome is None:
            self.genome = {
                "exploratory_tendency": exploration,
                "risk_tolerance": self.random.random(),
                "sociability": self.random.random(),
                "cooperative_tendency": self.random.random(),
                "farming_tendency": self.random.random(),
            }
        else:
            # Use the given dictionary, not new random genes.
            self.genome = genome.copy()
        # Model 2's body genes (vision, body_size). A genome from a Model 1
        # gene pool does not have them, so they are drawn at random.
        for trait in model.genes:
            if trait not in self.genome:
                self.genome[trait] = self.random.random()
        if parent_ids is not None:
            self.resources = model.param("CHILD_STARTING_RESOURCES")
            self.last_action = f"Born at {position}; first normal turn is next timestep."
        # self.traits is what the agent's decisions actually read. Normally it
        # IS the genome. In the neutral control it is a separate random draw, so
        # genes are inherited and mutated but never affect behaviour: any change
        # in gene averages is then pure random drift, with no selection.
        if model.neutral:
            self.traits = {}
            for trait in model.genes:
                self.traits[trait] = self.random.random()
        else:
            self.traits = self.genome
        # Model 2 body: what the two body genes cost and give. Both are fixed
        # at birth. In Model 1 the extra cost is zero and strength is not used.
        #   body_cost  added to the living cost every timestep (finish_turn)
        #   strength   weight in fights (confront) and in sharing a cell's food (share_food)
        self.body_cost = 0.0
        self.strength = 1.0
        if model.body_genes:
            extra_radius = (config.VISION_MAX_RADIUS - config.VISION_MIN_RADIUS) * self.traits["vision"]
            self.body_cost = (model.param("VISION_COST") * extra_radius
                              + model.param("BODY_COST") * self.traits["body_size"])
            self.strength = model.param("BODY_BASE") + self.traits["body_size"]
        self.sight_radius = config.VISION_MIN_RADIUS  # Radius used on the latest turn.
        # Three-zone map: how many rows into the rival homeland this agent is
        # willing to go, set at birth from its expansion gene. A value between
        # two whole rows is rounded up or down at random, in proportion.
        self.reach = 0
        if model.zones:
            exact = self.traits["expansion"] * model.param("EXPANSION_MAX_ROWS")
            self.reach = int(exact)
            if self.random.random() < exact - self.reach:
                self.reach += 1

    # model.cell_agents is a lookup table: position → list of living agents there.
    # It lets an agent check only the nine nearby cells instead of every agent.
    # @property makes "agent.position = (2, 3)" run the setter below, so the
    # table is always updated whenever any code moves an agent.
    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, new_position):
        # Dead agents keep their last position but are no longer in the table.
        if self.alive:
            if hasattr(self, "_position"):
                self.model.cell_agents[self._position].remove(self)
            self.model.cell_agents[new_position].append(self)
        self._position = new_position

    # Removing an agent also takes it out of the lookup table, then lets Mesa
    # remove it from the list of living agents.
    def remove(self):
        self.model.cell_agents[self._position].remove(self)
        super().remove()

    # THE FITNESS FUNCTION (Model 2; used by SocietyModel.choose_partner).
    # Everything the agent has gathered so far in its life: food eaten, plus
    # robbery gains, plus trade bonuses, minus what was robbed from it. It is
    # divided by what one agent needs to stay alive for a full lifetime, so a
    # fitness of 1 means "has gathered one lifetime's basic need". The total
    # only grows while the agent is alive, so gathering fast and surviving
    # long both raise it. No behaviour is rewarded directly.
    def fitness(self):
        gathered = self.total_food + self.total_robbed + self.total_trade - self.total_lost
        environment = self.model.environment
        return gathered / (environment["maximum_age"] * environment["survival_cost"])

    # How far the agent sees this turn (Model 2). The vision gene maps to a
    # radius between 1 and 4 cells. A gene value that falls between two whole
    # radii is rounded up or down at random, in proportion: 2.3 becomes 3 on
    # 30% of turns and 2 on 70%. So every small change in the gene matters.
    def look_radius(self):
        exact = config.VISION_MIN_RADIUS + (
            (config.VISION_MAX_RADIUS - config.VISION_MIN_RADIUS) * self.traits["vision"])
        radius = int(exact)
        if self.random.random() < exact - radius:
            radius += 1
        return radius

    # A parent must be alive, old enough, well supplied, and off cooldown.
    def can_reproduce(self):
        return (
            self.alive
            and self.age >= self.model.param("MIN_REPRODUCTION_AGE")
            and self.resources >= self.model.param("REPRODUCTION_THRESHOLD")
            and self.model.steps >= self.next_reproduction_step
            and self.unique_id not in self.model.parents_used_this_step
        )

    # Plain-language version of can_reproduce() for the dashboard's inspector.
    # It describes the next timestep, which is when reproduction is checked again.
    def reproduction_status(self):
        if not self.alive:
            return "Dead"
        if self.age < self.model.param("MIN_REPRODUCTION_AGE"):
            return f"Too young (reproduces from age {self.model.param('MIN_REPRODUCTION_AGE')})"
        if self.resources < self.model.param("REPRODUCTION_THRESHOLD"):
            return f"Needs {self.model.param('REPRODUCTION_THRESHOLD'):.1f} resources"
        if self.model.steps + 1 < self.next_reproduction_step:
            return f"Cooldown until timestep {self.next_reproduction_step}"
        return "Ready: needs an eligible partner from its population nearby"

    # A turn has three phases (see SocietyModel.step):
    #   1. step()            every agent looks around and moves, in random order.
    #                        Meeting a rival on the way leads to a confrontation.
    #   2. share_food()      the agents in each cell split that cell's food
    #   3. finish_turn()     every agent eats, pays its costs, ages, may die,
    #                        and works or plants farms

    # Phase 1: look around, then either stay to tend a farm or choose where to move.
    def step(self):
        # model.all_agents keeps dead agents, so make sure a dead one never acts.
        if not self.alive:
            return
        seen = []
        rival_positions = []  # Where the rivals in view are standing.
        # Model 1: every agent sees one cell in each direction. Model 2: the vision gene decides.
        radius = 1
        if self.model.body_genes:
            radius = self.look_radius()
            self.sight_radius = radius
        for agent in self.model.nearby_agents(self.position, radius):
            if agent is not self:
                seen.append(agent.name)
                if agent.population != self.population:
                    rival_positions.append(agent.position)

        # TEND or MOVE ON. On its own population's farm, the agent stays to work
        # it with chance farming_tendency: farmers settle down, the others roam
        # in search of wild food, rival farms to raid, and fights.
        # Agents only stay while saving up: once they have enough to reproduce,
        # they move on, which is how settled farmers meet partners.
        environment = self.model.environment
        on_own_farm = self.model.farms.get(self.position) == self.population
        saving_up = self.resources < self.model.param("REPRODUCTION_THRESHOLD")
        # On the three-zone map there is no forced staying: soil wears out, so
        # the agent simply compares its farm with the other cells it can see,
        # and the farming gene only decides whether it PLANTS (see farm()).
        if (on_own_farm and saving_up and not self.model.zones
                and self.random.random() < self.traits["farming_tendency"]):
            destination = self.position
            reason = "stayed to tend its population's farm"
            rival_note = ""
        elif self.model.body_genes:
            destination, reason, rival_note = self.choose_destination_wide(radius, rival_positions)
        else:
            destination, reason, rival_note = self.choose_destination(rival_positions)

        old_position = self.position
        self.position = destination
        # Chebyshev distance: moving one cell diagonally counts as one move.
        distance = max(abs(destination[0] - old_position[0]), abs(destination[1] - old_position[1]))
        self.turn_movement_cost = distance * environment["movement_cost"]
        if distance > 0:
            self.model.moves[self.population] += 1
        # The explanation text is only for the dashboard; batch runs skip it.
        if not self.model.headless:
            if destination != old_position:
                movement = f"moved {old_position} → {destination}"
            else:
                movement = f"stayed at {destination}"
            self.turn_movement_note = f"{movement}: {reason}. Saw agents: {', '.join(seen) or 'none'}.{rival_note}"
        # Meeting rivals leads to a confrontation; see SocietyModel.confront().
        self.model.detect_encounters(self)

    # Where to move this turn: avoid or ignore rivals, raid or respect rival
    # farms, then go for the most food in view, or explore if it is too little.
    # Returns the destination and two pieces of explanation text.
    def choose_destination(self, rival_positions):
        # The agent sees the 3 by 3 area around it, including its current cell.
        visible = self.model.cells_around(self.position)

        # AVOID or IGNORE, decided only when a rival is in view.
        # sociability is the chance of ignoring rivals. Otherwise the agent
        # avoids them: it keeps out of every cell on or next to a rival it can
        # see, because meeting a rival there would lead to a confrontation.
        options = visible
        rival_note = ""
        if rival_positions:
            if self.random.random() < self.traits["sociability"]:
                self.model.count_action(self.population, "ignore")
                rival_note = " Ignored the rivals in view."
            else:
                self.model.count_action(self.population, "avoid")
                rival_note = " Kept its distance from rivals."
                options = []
                for position in visible:
                    close_to_rival = False
                    for rival_position in rival_positions:
                        if max(abs(position[0] - rival_position[0]), abs(position[1] - rival_position[1])) <= 1:
                            close_to_rival = True
                    if not close_to_rival:
                        options.append(position)
                # Surrounded: there is nowhere safe to go.
                if not options:
                    options = visible

        # RAID or RESPECT, decided only when a rival population's farm with food
        # is among the options. The chance of raiding is 1 − cooperative_tendency.
        # Respecting agents leave rival farms out of their options.
        rival_farms = []
        for position in options:
            if self.model.is_rival_farm(position, self.population) and self.model.food[position] > 0:
                rival_farms.append(position)
        if rival_farms:
            if self.random.random() < 1 - self.traits["cooperative_tendency"]:
                self.model.count_action(self.population, "raid")
                rival_note += " Willing to raid a rival farm."
            else:
                self.model.count_action(self.population, "respect")
                rival_note += " Respected the rival farms in view."
                allowed = []
                for position in options:
                    if position not in rival_farms:
                        allowed.append(position)
                if allowed:
                    options = allowed

        # Find the largest food amount among the cells the agent is willing to enter.
        most_food = 0.0
        for position in options:
            if self.model.food[position] > most_food:
                most_food = self.model.food[position]

        # EXPLORE or EXPLOIT. Exploring is only considered when the best food in
        # view would not even pay for this turn (survival + one move).
        # exploratory_tendency is then the chance of jumping two cells into
        # unseen territory, at twice the movement cost, hoping for better food.
        environment = self.model.environment
        poor = most_food < environment["survival_cost"] + environment["movement_cost"]
        far_cells = self.cells_two_away()
        if poor and far_cells and self.random.random() < self.traits["exploratory_tendency"]:
            destination = self.random.choice(far_cells)
            reason = f"explored: the most food in view ({most_food:.2f}) was too little"
        elif most_food > 0:
            # Several cells may tie. Make a list and pick one randomly.
            choices = []
            for position in options:
                if self.model.food[position] == most_food:
                    choices.append(position)
            destination = self.random.choice(choices)
            reason = f"chose the most visible food ({most_food:.2f}); ties are random"
        else:
            neighbors = []
            for position in options:
                if position != self.position:
                    neighbors.append(position)
            if neighbors:
                destination = self.random.choice(neighbors)
                reason = "saw no food, so searched a neighboring cell"
            else:
                destination = self.position
                reason = "had no neighboring cell to move to"

        return destination, reason, rival_note

    # Model 2 version of choose_destination(): the same four decisions, but
    # the agent sees "radius" cells in every direction and may walk to any
    # cell it sees, paying the movement cost for every cell of distance. So it
    # compares cells by what they are worth after the walk: food − walking cost.
    def choose_destination_wide(self, radius, rival_positions):
        model = self.model
        environment = model.environment
        movement_cost = environment["movement_cost"]
        row, column = self.position

        # AVOID or IGNORE, exactly as in Model 1.
        avoiding = False
        rival_note = ""
        if rival_positions:
            if self.random.random() < self.traits["sociability"]:
                model.count_action(self.population, "ignore")
                rival_note = " Ignored the rivals in view."
            else:
                model.count_action(self.population, "avoid")
                rival_note = " Kept its distance from rivals."
                avoiding = True

        # Only fertile cells can hold food, so only those need checking.
        options = []       # (cell, distance, expected food) for every cell with food the agent may enter.
        rival_farms = False
        for cell in model.fertile_cells(self.position, radius):
            food = model.food[cell]
            if food <= 0 or not model.may_enter(self, cell):
                continue
            if avoiding and self.close_to_rival(cell, rival_positions):
                continue
            distance = max(abs(cell[0] - row), abs(cell[1] - column))
            # The agent expects to share the cell equally with whoever is
            # already standing there (agents move one at a time, so it sees
            # where the earlier movers went).
            others = len(model.cell_agents[cell])
            if distance > 0:
                food = food / (others + 1)
            elif others > 1:
                food = food / others
            options.append((cell, distance, food))
            if model.is_rival_farm(cell, self.population):
                rival_farms = True

        # RAID or RESPECT, exactly as in Model 1.
        if rival_farms:
            if self.random.random() < 1 - self.traits["cooperative_tendency"]:
                model.count_action(self.population, "raid")
                rival_note += " Willing to raid a rival farm."
            else:
                model.count_action(self.population, "respect")
                rival_note += " Respected the rival farms in view."
                allowed = []
                for option in options:
                    if not model.is_rival_farm(option[0], self.population):
                        allowed.append(option)
                options = allowed

        # The best cell is the one with the most food left after walking there.
        best_value = 0.0
        choices = []
        for cell, distance, food in options:
            value = food - movement_cost * distance
            if value > best_value:
                best_value = value
                choices = [cell]
            elif value == best_value and choices:
                choices.append(cell)

        # EXPLORE or EXPLOIT. Poor means the best cell would not pay for this
        # turn's basic survival cost. Exploring jumps to just beyond the edge
        # of sight (radius + 1 cells away), paying for every cell.
        poor = best_value < environment["survival_cost"]
        far_cells = [cell for cell in self.cells_at_distance(radius + 1) if model.may_enter(self, cell)]
        if poor and far_cells and self.random.random() < self.traits["exploratory_tendency"]:
            destination = self.random.choice(far_cells)
            reason = f"explored: the best food in view was worth only {best_value:.2f}"
        elif choices:
            destination = self.random.choice(choices)
            reason = f"chose the best food in view (worth {best_value:.2f} after walking); ties are random"
        else:
            neighbors = []
            for position in model.cells_around(self.position):
                if (position != self.position and model.may_enter(self, position)
                        and not (avoiding and self.close_to_rival(position, rival_positions))):
                    neighbors.append(position)
            if neighbors:
                destination = self.random.choice(neighbors)
                reason = "saw no food worth the walk, so searched a neighboring cell"
            else:
                destination = self.position
                reason = "had no neighboring cell to move to"
        return destination, reason, rival_note

    # True if the cell is on or next to a rival the agent can see.
    def close_to_rival(self, cell, rival_positions):
        for rival_position in rival_positions:
            if max(abs(cell[0] - rival_position[0]), abs(cell[1] - rival_position[1])) <= 1:
                return True
        return False

    # Clear what happened last turn. Called for every agent at the start of a
    # timestep, before anyone moves, because an agent can be robbed before its
    # own move.
    def start_turn(self):
        self.turn_start_resources = self.resources
        self.turn_food = 0.0
        self.turn_movement_cost = 0.0
        self.turn_fight_cost = 0.0
        self.turn_stolen = 0.0  # Resources robbed from (−) or by (+) this agent.
        self.turn_trade = 0.0   # Trade bonus collected this turn (Model 2).
        self.turn_movement_note = ""
        self.turn_food_note = ""
        self.turn_meeting_note = ""

    # What the agent holds right now, before this turn's food and costs.
    def current_resources(self):
        return self.resources + self.turn_stolen - self.turn_fight_cost + self.turn_trade

    # Cells exactly two steps away in one of the eight directions: outside the
    # agent's view, so jumping there is a gamble.
    def cells_two_away(self):
        return self.cells_at_distance(2)

    # Cells exactly "distance" steps away in one of the eight directions.
    def cells_at_distance(self, distance):
        row, column = self.position
        cells = []
        for row_step in [-1, 0, 1]:
            for column_step in [-1, 0, 1]:
                if row_step == 0 and column_step == 0:
                    continue
                target_row = row + distance * row_step
                target_column = column + distance * column_step
                if 0 <= target_row < self.model.grid_size and 0 <= target_column < self.model.grid_size:
                    cells.append((target_row, target_column))
        return cells

    # Phase 3: eat the food received in share_food(), pay costs, age, maybe die,
    # and finally work on or plant a farm.
    def finish_turn(self):
        # The living cost: the same basic survival cost for everyone, plus (in
        # Model 2) what this agent's eyesight and body size cost to maintain.
        survival_cost = self.model.environment["survival_cost"] + self.body_cost
        self.resources += (self.turn_food + self.turn_stolen - self.turn_movement_cost
                           - self.turn_fight_cost - survival_cost) + self.turn_trade
        self.total_food += self.turn_food
        self.age += 1

        # Death removes an agent from Mesa's living-agent list.
        # model.all_agents still keeps it so its final state stays on the page.
        if self.resources <= 0:
            self.alive = False
            self.status = "Dead: resources ran out"
            self.model.starvation_deaths += 1
            self.model.deaths[self.population]["starvation"] += 1
        elif self.age >= self.model.environment["maximum_age"]:
            self.alive = False
            self.status = f"Dead: reached age {self.model.environment['maximum_age']}"
            self.model.old_age_deaths += 1
            self.model.deaths[self.population]["old_age"] += 1
        if not self.alive:
            self.remove()
            self.model.record_lifetime(self, "starvation" if self.resources <= 0 else "old_age")
            if self.resources <= 0:
                self.model.add_event("death", f"{self.name} starved at age {self.age}",
                                     [self.unique_id], self.position)
            else:
                self.model.add_event("death", f"{self.name} died of old age",
                                     [self.unique_id], self.position)

        farm_note = ""
        planting_cost = 0.0
        if self.alive:
            farm_note, planting_cost = self.farm()

        # The rest only writes the explanation for the dashboard.
        if self.model.headless:
            return
        # :.2f means show a number with two digits after the decimal point.
        extra_text = ""
        if self.turn_stolen > 0:
            extra_text += f" + {self.turn_stolen:.2f} stolen"
        elif self.turn_stolen < 0:
            extra_text += f" − {-self.turn_stolen:.2f} stolen from it"
        if self.turn_fight_cost > 0:
            extra_text += f" − {self.turn_fight_cost:.2f} fighting"
        if self.turn_trade > 0:
            extra_text += f" + {self.turn_trade:.2f} trade"
        if planting_cost > 0:
            extra_text += f" − {planting_cost:.2f} planting"
        self.last_action = (
            f"{self.turn_movement_note}{self.turn_meeting_note}{self.turn_food_note}{farm_note} "
            f"Resources: {self.turn_start_resources:.2f} + {self.turn_food:.2f} food "
            f"− {self.turn_movement_cost:.2f} movement{extra_text} − {survival_cost:.2f} survival "
            f"= {self.resources:.2f}. Age: {self.age}. {self.status}."
        )
        self.model.messages.append(f"{self.name}: {self.last_action}")

    # Farming, at the end of a living agent's turn. Returns (note, cost paid).
    # Standing on its own population's farm counts as working it, which stops
    # the farm from being abandoned. An agent that can afford it may also plant
    # a new farm, with chance farming_tendency (investing in a safe food
    # supply). Only fertile land can be farmed: a wild patch that is
    # not a farm yet. The agent plants on its own cell, or, when working a farm,
    # on a fertile neighbouring cell. That is how farmland spreads into villages.
    def farm(self):
        model = self.model
        note = ""
        site = None
        if not model.param("FARMS_ENABLED"):
            return note, 0.0
        if model.farms.get(self.position) == self.population:
            model.farm_last_worked[self.position] = model.steps
            note = " Worked its population's farm."
            free_cells = []
            for position in model.cells_around(self.position):
                if model.can_plant(position, self.population):
                    free_cells.append(position)
            if free_cells:
                site = self.random.choice(free_cells)
        elif model.can_plant(self.position, self.population):
            site = self.position

        if site is None or self.resources < model.param("PLANT_MIN_RESOURCES"):
            return note, 0.0
        if self.random.random() < self.traits["farming_tendency"]:
            cost = model.param("PLANT_COST")
            self.resources -= cost
            model.plant_farm(site, self.population, self)
            return note + f" Planted a farm at {site} for population {self.population}.", cost
        return note, 0.0


class SocietyModel(mesa.Model):
    # Create the world and its starting agents.
    #   environment_name  "ABUNDANT", "SCARCE", "LEVEL_1" ... "LEVEL_5", or a level number 1-5.
    #   world_name        "SMALL", "LARGE" or "EXPERIMENT" (see config.WORLDS).
    #   seed              the same seed always gives exactly the same run.
    #   headless          True for batch runs: skips everything only the dashboard
    #                     needs (explanation text, event log, encounter list, dead
    #                     agents). The simulation itself is identical either way.
    #   neutral           True for the neutral control: genes are inherited but
    #                     behaviour is random, so there is no selection (see SocietyAgent).
    #   overrides         {name: value} replacing environment settings (e.g.
    #                     "farm_growth") or config constants (e.g. "FIGHT_COST") for this run.
    #   gene_pools        {"A": [genome, ...], "B": [...]}: starting agents of that
    #                     population draw their genome from the list instead of at random.
    #   frozen_genomes    True stops evolution: every child gets a genome copied
    #                     from its population's founders instead of inheriting.
    #   record_every      headless runs save a data row every this many timesteps.
    def __init__(self, environment_name=config.DEFAULT_ENVIRONMENT, world_name=config.DEFAULT_WORLD,
                 seed=None, headless=False, neutral=False, overrides=None, gene_pools=None,
                 frozen_genomes=False, record_every=1, version=1, zones=False):
        # A seed makes Reset repeat the same sequence of random choices.
        if seed is None:
            seed = config.SEED
        super().__init__(rng=seed)
        self.seed = seed
        self.headless = headless
        self.neutral = neutral
        self.frozen_genomes = frozen_genomes
        self.record_every = record_every
        # version=2 switches on every Model 2 mechanic (config.MODEL_2) and
        # uses Model 2's seven abundance levels. Explicit overrides still win.
        self.version = version
        if version == 2:
            overrides = {**config.MODEL_2, **(config.MODEL_2_ZONES if zones else {}), **(overrides or {})}
            environment_name = config.model2_environment_name(environment_name)
        environment_name = config.environment_name(environment_name)
        self.environment_name = environment_name
        self.world_name = world_name
        # Copy the presets so this run keeps its own settings.
        self.environment = config.ENVIRONMENTS[environment_name].copy()
        self.world = config.WORLDS[world_name].copy()
        # Overrides: lower-case names are environment settings, upper-case
        # names are constants from config.py (read through self.param()).
        self.overrides = {}
        for name, value in (overrides or {}).items():
            if name in self.environment:
                self.environment[name] = value
            elif name.isupper() and hasattr(config, name):
                self.overrides[name] = value
            else:
                raise ValueError(f"Unknown parameter override: {name}")
        # How partners are chosen and whether genes steer behaviour (config.SELECTION_MODE).
        # neutral=True is the neutral control of whichever model is running.
        self.selection_mode = self.param("SELECTION_MODE")
        if self.selection_mode not in ("endogenous", "fitness", "neutral"):
            raise ValueError(f"Unknown selection mode: {self.selection_mode}")
        if self.selection_mode == "neutral":
            self.neutral = True
        elif neutral and self.selection_mode == "fitness":
            self.selection_mode = "neutral"
        self.body_genes = self.param("BODY_GENES")
        self.homeland_band = self.param("HOMELAND_BAND")
        self.soil = self.param("SOIL_ENABLED")
        # The three-zone map needs room for two homelands and the shared land,
        # so the 3 by 3 debug world never uses it.
        self.zones = bool(self.param("ZONES")) and self.world["grid_size"] >= 20
        self.genes = list(config.GENES)  # The genome: five genes, or seven with the body genes.
        if self.body_genes:
            self.genes += config.BODY_GENE_NAMES
        if self.zones:
            self.genes += config.ZONE_GENE_NAMES
        if self.param("CHILD_STARTING_RESOURCES") != 2 * self.param("REPRODUCTION_COST_PER_PARENT"):
            raise ValueError("Child starting resources must equal both parents' combined contributions.")
        self.grid_size = self.world["grid_size"]
        self.around_cache = {}  # See cells_around().
        self.food = {}
        self.capacity = {}
        self.cell_agents = {}  # Position → living agents there. See SocietyAgent.position.
        self.farms = {}  # Position → population that owns the farm there.
        self.farm_last_worked = {}  # Position → last timestep an owner worked the farm.
        for row in range(self.grid_size):
            for column in range(self.grid_size):
                self.food[(row, column)] = 0.0
                self.capacity[(row, column)] = 0.0
                self.cell_agents[(row, column)] = []

        # The SMALL world uses hand-placed patches. Other worlds place them randomly.
        if world_name == "SMALL":
            patches = self.environment["patches"]
        else:
            patches = []
            for position in self.food:
                if self.random.random() < self.environment["patch_density"]:
                    patches.append(position)
        for position in patches:
            self.food[position] = self.environment["food_per_patch"]
            self.capacity[position] = self.environment["food_per_patch"]
        # Remember the wild capacity, so an abandoned farm can return to it.
        # wild_capacity > 0 also marks a cell as fertile (plantable) land.
        self.wild_capacity = self.capacity.copy()
        # Three zones: rows 0 .. shared_first-1 are homeland A, the next
        # SHARED_ROWS rows are the shared land, the rest are homeland B.
        # row_zone[row] is "A", "S" (shared) or "B".
        self.row_zone = None
        self.unplanted_capacity = self.wild_capacity
        if self.zones:
            shared_rows = self.param("SHARED_ROWS")
            self.shared_first = (self.grid_size - shared_rows) // 2
            self.shared_last = self.shared_first + shared_rows - 1
            self.row_zone = []
            for row in range(self.grid_size):
                if row < self.shared_first:
                    self.row_zone.append("A")
                elif row <= self.shared_last:
                    self.row_zone.append("S")
                else:
                    self.row_zone.append("B")
            # Homelands have no wild food: fertile cells there stay empty until
            # somebody plants a farm, and go back to empty when it is abandoned.
            self.unplanted_capacity = self.wild_capacity.copy()
            for position in self.capacity:
                if self.row_zone[position[0]] != "S" and self.wild_capacity[position] > 0:
                    self.food[position] = 0.0
                    self.capacity[position] = 0.0
                    self.unplanted_capacity[position] = 0.0
            # Wild regrowth in the shared land is the same at every level.
            if "regeneration" not in (overrides or {}):
                self.environment["regeneration"] = self.param("SHARED_REGENERATION")
        # Farm growth per population (the same for both unless HOMELAND_LEVELS is set).
        self.set_farm_growth()
        # Counters for the three-zone analysis (all cumulative).
        self.food_from = {p: {"farm": 0.0, "wild": 0.0, "raid": 0.0} for p in "AB"}
        self.zone_steps = {p: {"home": 0, "shared": 0, "rival": 0} for p in "AB"}
        self.encounters_in = {"A": 0, "S": 0, "B": 0}
        self.fights_in = {"A": 0, "S": 0, "B": 0}
        self.encounter_rows = [0] * self.grid_size
        # Soil (Model 2): every fertile cell has a fertility between 0 and 1
        # that multiplies farm growth there. It belongs to the cell, so it
        # stays when a farm is abandoned and planted again. See update_soil().
        self.fertile = [position for position in self.capacity if self.wild_capacity[position] > 0]
        self.fertility = dict.fromkeys(self.fertile, 1.0)
        self.harvested = set()        # Farm cells eaten from this timestep.
        self.fertile_cache = {}       # See fertile_cells().
        self.moves = {"A": 0, "B": 0}       # Turns on which an agent changed cell.
        self.trades = {"A": 0, "B": 0}      # Trade bonuses collected.
        self.trade_created = 0.0            # Resources created by trade bonuses.

        # The SMALL world uses the fixed list in config. Other worlds start A in
        # the top half and B in the bottom half, with fully random genomes.
        if world_name == "SMALL":
            starting_agents = config.STARTING_AGENTS
        else:
            starting_agents = []
            half = self.grid_size // 2
            halves = [("A", 0, half), ("B", half, self.grid_size)]
            if self.zones:  # Everyone starts in its own homeland.
                halves = [("A", 0, self.shared_first), ("B", self.shared_last + 1, self.grid_size)]
            if self.param("INTERLEAVED_START"):
                halves = [("A", 0, self.grid_size), ("B", 0, self.grid_size)]
            for population, first_row, last_row in halves:
                for number in range(1, self.world["agents_per_population"] + 1):
                    position = (
                        self.random.randrange(first_row, last_row),
                        self.random.randrange(self.grid_size),
                    )
                    starting_agents.append((f"{population}{number}", position, self.random.random()))

        # Every agent ever created, living or dead, for the dashboard's inspector.
        # Headless runs leave it empty and keep model.lifetime_records instead.
        self.all_agents = []
        self.lifetime_records = []  # One row per dead agent. See record_lifetime().
        self.deaths = {"A": {"starvation": 0, "old_age": 0}, "B": {"starvation": 0, "old_age": 0}}
        self.agent_steps = {"A": 0, "B": 0}  # Turns lived, summed over a population's agents.
        self.founding_pools = {"A": [], "B": []}  # The starting agents' genomes.
        self.next_agent_number = {"A": 1, "B": 1}
        for name, position, exploration in starting_agents:
            genome = None
            if gene_pools and gene_pools.get(name[0]):
                genome = self.random.choice(gene_pools[name[0]])
            agent = SocietyAgent(self, name, position, exploration, genome=genome)
            if not headless:
                self.all_agents.append(agent)
            self.founding_pools[agent.population].append(agent.genome.copy())
            self.next_agent_number[agent.population] = max(
                self.next_agent_number[agent.population], int(name[1:]) + 1
            )

        self.messages = [f"Starting {world_name} {environment_name} world. Press Next timestep."]
        self.events = []  # Short readable events for the dashboard. See add_event().
        self.extinct = set()  # Populations already reported as extinct.
        self.last_mutations = []  # Traits that mutated in the most recent child.
        self.encounters_this_step = []
        self.encounters = []  # Permanent snapshots from every completed timestep.
        self.encounter_pairs = set()  # A set remembers pairs already seen this step.
        self.encounters_per_timestep = 0
        self.cumulative_encounters = 0
        # Death counts are cumulative for the run, not just the latest timestep.
        self.starvation_deaths = 0
        self.old_age_deaths = 0
        self.parents_used_this_step = set()
        self.births_this_step = 0
        self.cumulative_births = 0
        self.population_a_births = 0
        self.population_b_births = 0
        self.highest_generation = 0  # Highest ever born, including agents that later die.
        # How often each population chose each response, this timestep and in total.
        self.actions_this_step = {}
        self.cumulative_actions = {}
        for population in ["A", "B"]:
            self.actions_this_step[population] = {}
            self.cumulative_actions[population] = {}
            for action in config.ACTIONS:
                self.actions_this_step[population][action] = 0
                self.cumulative_actions[population][action] = 0
        self.fights_this_step = 0
        self.cumulative_fights = 0
        # Food and resources each population took from the other, by raids or fights.
        self.stolen_this_step = {"A": 0.0, "B": 0.0}
        self.cumulative_stolen = {"A": 0.0, "B": 0.0}
        self.farms_planted = {"A": 0, "B": 0}
        # DataCollector saves one row of numbers per timestep for later analysis.
        # These strings name attributes updated in collect_data() below.
        reporters = {
            "Timestep": "steps",
            "Population A": "population_a",
            "Population B": "population_b",
            "Environmental resources": "environmental_resources",
            "Average agent resources": "average_resources",
            "Average age": "average_age",
            "Encounters per timestep": "encounters_per_timestep",
            "Cumulative encounters": "cumulative_encounters",
            "Deaths from starvation": "starvation_deaths",
            "Deaths from old age": "old_age_deaths",
            "Births per timestep": "births_this_step",
            "Cumulative births": "cumulative_births",
            "Population A births": "population_a_births",
            "Population B births": "population_b_births",
            "Highest generation reached": "highest_generation",
            "Average generation": "average_generation",
            "Fights per timestep": "fights_this_step",
            "Cumulative fights": "cumulative_fights",
        }
        # One column per population and response, counted per timestep.
        for population in ["A", "B"]:
            for action in config.ACTIONS:
                reporters[f"Population {population} {action}"] = f"{population}_{action}"
            reporters[f"Population {population} farms"] = f"{population}_farms"
            reporters[f"Population {population} stolen"] = f"{population}_stolen"
        # Running totals. Analysis uses these, because they stay correct when
        # a row is saved only every few timesteps (record_every).
        for population in ["A", "B"]:
            for action in config.ACTIONS:
                reporters[f"Population {population} cumulative {action}"] = f"{population}_cumulative_{action}"
            for column in ["cumulative stolen", "starvation deaths", "old age deaths", "agent steps",
                           "moves", "trades"]:
                reporters[f"Population {population} {column}"] = f"{population}_{column.replace(' ', '_')}"
        # Add one data column for each trait in each population.
        # A reporter string tells Mesa where to read that value on this model.
        for population in ["A", "B"]:
            for trait in self.genes:
                column = f"Population {population} average {trait}"
                reporters[column] = f"{population}_average_{trait}"
        reporters["Average fertility"] = "average_fertility"
        for population in ["A", "B"]:
            for source in ["farm", "wild", "raid"]:
                reporters[f"Population {population} food {source}"] = f"{population}_food_{source}"
            for zone in ["home", "shared", "rival"]:
                reporters[f"Population {population} steps {zone}"] = f"{population}_steps_{zone}"
        for zone, label in [("A", "homeland A"), ("S", "shared land"), ("B", "homeland B")]:
            reporters[f"Encounters in {label}"] = f"encounters_in_{zone}"
            reporters[f"Fights in {label}"] = f"fights_in_{zone}"
        self.datacollector = DataCollector(model_reporters=reporters)
        self.collect_data()

    # Count living agents with the requested population label.
    def count_population(self, population):
        count = 0
        for agent in self.agents:
            if agent.population == population:
                count += 1
        return count

    # A constant from config.py, unless this run overrides it.
    def param(self, name):
        if name in self.overrides:
            return self.overrides[name]
        return getattr(config, name)

    # Switch to another abundance level in the middle of a run (the shock
    # experiment). The land stays the same; only the growth rates change.
    def set_environment(self, name_or_level):
        if self.version == 2:
            name_or_level = config.model2_environment_name(name_or_level)
        name = config.environment_name(name_or_level)
        self.environment_name = name
        for setting in ["level", "multiplier", "regeneration", "farm_growth"]:
            if setting == "regeneration" and self.zones:
                continue  # The shared land does not change with the level.
            self.environment[setting] = config.ENVIRONMENTS[name][setting]
        self.set_farm_growth()

    # Farm growth per timestep for each population's farms. Normally both use
    # the environment's value. HOMELAND_LEVELS (experiment M7) gives each
    # population's homeland its own Model 2 abundance level.
    def set_farm_growth(self):
        growth = self.environment["farm_growth"]
        self.farm_growth = {"A": growth, "B": growth}
        levels = self.param("HOMELAND_LEVELS")
        if levels:
            for population, level in levels.items():
                self.farm_growth[population] = config.BASE_FARM_GROWTH * config.M2_MULTIPLIERS[level]

    # Save one row describing an agent's whole life. Called when it dies.
    def record_lifetime(self, agent, cause):
        record = {
            "id": agent.unique_id,
            "population": agent.population,
            "generation": agent.generation,
            "birth_step": agent.birth_step,
            "lifespan": agent.age,
            "food_eaten": agent.total_food,
            "stolen": agent.total_stolen,
            "lost": agent.total_lost,
            "offspring": agent.offspring,
            "cause": cause,
            "trade": agent.total_trade,
            "fitness": agent.fitness(),
        }
        record.update(agent.genome)
        self.lifetime_records.append(record)

    # The genomes of a population's living agents: its gene pool.
    def living_genomes(self, population):
        genomes = []
        for agent in self.agents:
            if agent.population == population:
                genomes.append(agent.genome.copy())
        return genomes

    # The cells within "radius" steps of a position, including that cell.
    # Radius 1 is the 3 by 3 area an agent can see.
    # max() and min() keep the row and column inside the grid edges.
    def cells_around(self, position, radius=1):
        # The grid never changes, so each answer is worked out once and remembered.
        key = (position, radius)
        if key in self.around_cache:
            return self.around_cache[key]
        row, column = position
        cells = []
        for nearby_row in range(max(0, row - radius), min(self.grid_size, row + radius + 1)):
            for nearby_column in range(max(0, column - radius), min(self.grid_size, column + radius + 1)):
                cells.append((nearby_row, nearby_column))
        self.around_cache[key] = cells
        return cells

    # Homelands (an option, off by default; see config.HOMELAND_BAND). Population
    # A lives in the top half and B in the bottom half. Each may also enter a
    # band of HOMELAND_BAND rows on the other side of the middle line, so the
    # two populations share a border zone but neither can be pushed off its land.
    #
    # Three-zone map (config.ZONES): an agent may always enter its own homeland
    # and the shared land. It may enter the rival homeland only as deep as its
    # expansion gene allows (agent.reach rows past the shared land).
    def may_enter(self, agent, cell):
        population = agent.population
        if self.zones:
            zone = self.row_zone[cell[0]]
            if zone == "S" or zone == population:
                return True
            if population == "A":
                return cell[0] - self.shared_last <= agent.reach
            return self.shared_first - cell[0] <= agent.reach
        band = self.homeland_band
        if band is None:
            return True
        half = self.grid_size // 2
        if population == "A":
            return cell[0] < half + band
        return cell[0] >= half - band

    # The fertile cells within "radius" steps of a position (Model 2 sight).
    # The land never changes, so each answer is worked out once and remembered.
    def fertile_cells(self, position, radius):
        key = (position, radius)
        if key not in self.fertile_cache:
            cells = []
            for cell in self.cells_around(position, radius):
                if self.wild_capacity[cell] > 0:
                    cells.append(cell)
            self.fertile_cache[key] = cells
        return self.fertile_cache[key]

    # Soil (Model 2), once per timestep: a farm that was eaten from loses
    # SOIL_DEPLETION fertility; every other fertile cell regains SOIL_RECOVERY.
    def update_soil(self):
        depletion = self.param("SOIL_DEPLETION")
        recovery = self.param("SOIL_RECOVERY")
        fertility = self.fertility
        for position in self.fertile:
            if position in self.harvested:
                fertility[position] = max(0.0, fertility[position] - depletion)
            elif fertility[position] < 1.0:
                fertility[position] = min(1.0, fertility[position] + recovery)
        self.harvested = set()

    # True if the cell is a farm belonging to the other population.
    def is_rival_farm(self, position, population):
        owner = self.farms.get(position)
        return owner is not None and owner != population

    # Fertile land that nobody farms yet: a wild patch without an owner.
    # On the three-zone map a population may only plant in its own homeland.
    def can_plant(self, position, population=None):
        if self.zones and population is not None and self.row_zone[position[0]] != population:
            return False
        return self.wild_capacity[position] > 0 and position not in self.farms

    # Turn a wild patch into a farm. It keeps its food and grows faster from now on.
    def plant_farm(self, position, population, planter):
        self.farms[position] = population
        self.farm_last_worked[position] = self.steps
        self.capacity[position] = self.param("FARM_CAPACITY")
        self.farms_planted[population] += 1
        self.add_event("farm", f"{planter.name} planted a farm for population {population}",
                       [planter.unique_id], position)

    # A farm nobody from its population has worked for a while goes back to
    # what it was before: bare land or a wild patch.
    def abandon_farms(self):
        for position in list(self.farms):
            if self.steps - self.farm_last_worked[position] > self.param("FARM_ABANDON_STEPS"):
                population = self.farms[position]
                del self.farms[position]
                del self.farm_last_worked[position]
                self.capacity[position] = self.unplanted_capacity[position]
                self.food[position] = min(self.food[position], self.capacity[position])
                self.add_event("farm", f"Population {population}'s farm was abandoned", [], position)

    # Record food or resources one population took from the other.
    def add_stolen(self, population, amount):
        self.stolen_this_step[population] += amount
        self.cumulative_stolen[population] += amount

    # Add one to the counters for a population's response (avoid, ignore, ...).
    def count_action(self, population, action):
        self.actions_this_step[population][action] += 1
        self.cumulative_actions[population][action] += 1

    # Phase 2 of a timestep, after every agent has moved: the agents in each
    # cell split its food equally. Food eaten from the other population's farm
    # counts as stolen.
    def share_food(self):
        for position, agents in self.cell_agents.items():
            if not agents:
                continue
            food = self.food[position]
            self.food[position] = 0.0
            # Model 2: a bigger agent takes a bigger share (in proportion to strength).
            total_strength = 0.0
            if self.body_genes:
                for agent in agents:
                    total_strength += agent.strength
            for agent in agents:
                if self.body_genes:
                    agent.turn_food = food * agent.strength / total_strength
                else:
                    agent.turn_food = food / len(agents)
                if len(agents) > 1 and not self.headless:
                    agent.turn_food_note += f" Split {food:.2f} food with {len(agents) - 1} other(s)."

            owner = self.farms.get(position)
            # Where each population's food comes from: its own farms, wild
            # food, or the other population's farms (raids).
            for agent in agents:
                if owner is None:
                    self.food_from[agent.population]["wild"] += agent.turn_food
                elif owner == agent.population:
                    self.food_from[agent.population]["farm"] += agent.turn_food
                else:
                    self.food_from[agent.population]["raid"] += agent.turn_food
            if owner is None or food == 0:
                continue
            self.harvested.add(position)  # Wears out the soil (Model 2). See update_soil().
            raiders = []
            defended = False
            for agent in agents:
                if agent.population == owner:
                    defended = True
                else:
                    raiders.append(agent)
                    self.add_stolen(agent.population, agent.turn_food)
                    agent.total_stolen += agent.turn_food
            if raiders and not defended and not self.headless:
                names = " and ".join(agent.name for agent in raiders)
                ids = []
                for agent in raiders:
                    ids.append(agent.unique_id)
                    agent.turn_food_note += f" Raided population {owner}'s farm."
                self.add_event("theft", f"{names} raided population {owner}'s farm and took {food:.2f} food",
                               ids, position)

    # Hawk–Dove confrontation between two agents from different populations
    # that have just met (same or neighbouring cell). At stake are the
    # resources they carry. Each one COOPERATES or COMPETES:
    #   both cooperate   → they leave each other alone
    #   one competes     → it robs the other of STEAL_FRACTION of its resources
    #   both compete     → each fights on (risk_tolerance) or backs down.
    #       One fights on: it robs the one that backed down.
    #       Both fight on: both pay FIGHT_COST and the winner robs the loser.
    #       Richer fighters are more likely to win. Both back down: nothing.
    def confront(self, first, second):
        competing = []
        for agent in [first, second]:
            # Hungry agents share less: with no resources the chance of
            # cooperating is halved; at the reproduction threshold it is unchanged.
            wealth = min(1.0, max(0.0, agent.current_resources()) / self.param("REPRODUCTION_THRESHOLD"))
            chance = agent.traits["cooperative_tendency"] * (0.5 + 0.5 * wealth)
            if self.random.random() < chance:
                self.count_action(agent.population, "cooperate")
            else:
                self.count_action(agent.population, "compete")
                competing.append(agent)

        if not competing:
            # Model 2: both cooperated, so they trade and each gains TRADE_BONUS.
            # This is the only place resources appear from nowhere, so it is
            # limited: an agent collects at most one bonus per TRADE_COOLDOWN timesteps.
            bonus = self.param("TRADE_BONUS")
            for agent in [first, second]:
                if bonus > 0 and self.steps >= agent.next_trade_step:
                    agent.turn_trade += bonus
                    agent.total_trade += bonus
                    agent.next_trade_step = self.steps + self.param("TRADE_COOLDOWN")
                    self.trades[agent.population] += 1
                    self.trade_created += bonus
                    if not self.headless:
                        agent.turn_meeting_note += f" Traded and gained {bonus:.2f}."
            for agent in ([] if self.headless else [first, second]):
                agent.turn_meeting_note += f" Met {first.name if agent is second else second.name} peacefully."
            return
        fought = False
        if len(competing) == 1:
            robber = competing[0]
        else:
            fighting = []
            for agent in competing:
                if self.random.random() < agent.traits["risk_tolerance"]:
                    fighting.append(agent)
            if not fighting:
                for agent in ([] if self.headless else [first, second]):
                    agent.turn_meeting_note += " Both sides threatened, then backed down."
                return
            if len(fighting) == 1:
                robber = fighting[0]
            else:
                fought = True
                weights = []
                for agent in fighting:
                    agent.turn_fight_cost += self.param("FIGHT_COST")
                    if self.body_genes:
                        # Model 2: body size decides fights; this replaces Model 1's wealth weighting.
                        weight = agent.strength
                        # Defender's advantage: stronger on its own homeland.
                        if self.zones and self.row_zone[agent.position[0]] == agent.population:
                            weight *= self.param("DEFENDER_ADVANTAGE")
                        weights.append(weight)
                    else:
                        weights.append(max(agent.current_resources(), 0.1))
                robber = self.random.choices(fighting, weights=weights)[0]
                self.fights_this_step += 1
                self.cumulative_fights += 1
                if self.zones:
                    self.fights_in[self.row_zone[second.position[0]]] += 1
        victim = second if robber is first else first
        self.rob(robber, victim, fought)

    # Move STEAL_FRACTION of the victim's resources to the robber.
    def rob(self, robber, victim, fought):
        amount = self.param("STEAL_FRACTION") * max(victim.current_resources(), 0.0)
        victim.turn_stolen -= amount
        robber.turn_stolen += amount
        victim.total_lost += amount
        robber.total_stolen += amount
        robber.total_robbed += amount
        self.add_stolen(robber.population, amount)
        if self.headless:
            return
        if fought:
            text = f"{robber.name} beat {victim.name} in a fight and robbed {amount:.2f}"
        else:
            text = f"{robber.name} robbed {victim.name} of {amount:.2f} without a fight"
        robber.turn_meeting_note += f" {text}."
        victim.turn_meeting_note += f" {text}."
        self.add_event("compete", text, [robber.unique_id, victim.unique_id], victim.position)
        self.messages.append(f"Confrontation: {text}.")

    # Living agents within "radius" steps of a position, including that cell.
    def nearby_agents(self, position, radius=1):
        found = []
        for cell in self.cells_around(position, radius):
            found.extend(self.cell_agents[cell])
        return found

    # Inherit each gene separately. All randomness comes from Mesa's seeded RNG.
    def make_child_genome(self, first, second):
        genome = {}
        self.last_mutations = []
        # Frozen genomes (experiment E3's "no further evolution" runs): the
        # child gets a copy of a random founder's genome, so the population's
        # gene pool never changes however long the run lasts.
        if self.frozen_genomes:
            return self.random.choice(self.founding_pools[first.population]).copy()
        for trait in first.genome:
            value = self.random.choice([first.genome[trait], second.genome[trait]])
            if self.random.random() < self.param("MUTATION_RATE"):
                # gauss(0, std) samples a small change centered on zero.
                value += self.random.gauss(0, self.param("MUTATION_STD"))
                self.last_mutations.append(trait)
            # Clamp: stop values going below zero or above one.
            genome[trait] = max(0.0, min(1.0, value))
        return genome

    # SELECTION (Model 2): an eligible agent picks its partner by tournament.
    # TOURNAMENT_SIZE eligible agents of its own population within
    # MATING_RADIUS are drawn at random, and the one with the highest fitness
    # (SocietyAgent.fitness) wins. In the neutral control the tournament has
    # size 1, so the partner is simply a random eligible neighbour.
    def choose_partner(self, first):
        eligible = []
        for other in self.nearby_agents(first.position, self.param("MATING_RADIUS")):
            if other is not first and other.population == first.population and other.can_reproduce():
                eligible.append(other)
        if not eligible:
            return None
        size = 1 if self.selection_mode == "neutral" else self.param("TOURNAMENT_SIZE")
        entrants = self.random.sample(eligible, min(size, len(eligible)))
        winner = entrants[0]
        for entrant in entrants[1:]:
            if entrant.fitness() > winner.fitness():
                winner = entrant
        return winner

    # Pair nearby eligible parents once, after normal turns have finished.
    def reproduce(self):
        # Copy before adding children. Newborns cannot join this round of pairing.
        candidates = []
        for agent in self.agents:
            if agent.can_reproduce():
                candidates.append(agent)
        self.random.shuffle(candidates)
        if self.selection_mode != "endogenous":
            for first in candidates:
                if first.can_reproduce():
                    second = self.choose_partner(first)
                    if second is not None:
                        self.make_child(first, second)
            return
        # Remember each candidate's place in the shuffled list.
        order = {}
        for index in range(len(candidates)):
            order[candidates[index].unique_id] = index

        for first in candidates:
            if not first.can_reproduce():
                continue
            # The partner is the earliest later candidate in the shuffled list
            # that is within MATING_RADIUS, same population, and still able.
            second = None
            for other in self.nearby_agents(first.position, self.param("MATING_RADIUS")):
                if other.unique_id not in order or order[other.unique_id] <= order[first.unique_id]:
                    continue
                if other.population != first.population or not other.can_reproduce():
                    continue
                if second is None or order[other.unique_id] < order[second.unique_id]:
                    second = other
            if second is not None:
                self.make_child(first, second)

    # Create the child of two parents and charge both parents for it.
    def make_child(self, first, second):
        genome = self.make_child_genome(first, second)
        position = self.random.choice([first.position, second.position])
        population = first.population
        name = f"{population}{self.next_agent_number[population]}"
        self.next_agent_number[population] += 1
        generation = max(first.generation, second.generation) + 1
        child = SocietyAgent(
            self, name, position, genome=genome, generation=generation,
            parent_ids=(first.unique_id, second.unique_id),
        )
        if not self.headless:
            self.all_agents.append(child)

        # Combined parent contributions become the child's holdings.
        for parent in [first, second]:
            parent.resources -= self.param("REPRODUCTION_COST_PER_PARENT")
            parent.next_reproduction_step = self.steps + self.param("REPRODUCTION_COOLDOWN")
            parent.offspring += 1
            self.parents_used_this_step.add(parent.unique_id)

        self.births_this_step += 1
        self.cumulative_births += 1
        if population == "A":
            self.population_a_births += 1
        else:
            self.population_b_births += 1
        self.highest_generation = max(self.highest_generation, generation)
        if self.headless:
            return
        event_text = f"{child.name} born to {first.name} + {second.name} (generation {generation})"
        if self.last_mutations:
            event_text += f"; mutated: {', '.join(self.last_mutations)}"
        self.add_event("birth", event_text, [child.unique_id, first.unique_id, second.unique_id], position)
        # Round only the explanation; the stored genes keep full precision.
        genome_description = []
        for trait, value in child.genome.items():
            genome_description.append(f"{trait}: {value:.3f}")
        genome_text = ", ".join(genome_description)
        self.messages.append(
            f"Reproduction: {first.name} + {second.name} → {child.name} at {position}. "
            f"Parents contributed {self.param('REPRODUCTION_COST_PER_PARENT'):.2f} each; "
            f"child starts with {child.resources:.2f}. Child generation: {generation}. "
            f"Child genome: {genome_text}. Parents next eligible at timestep "
            f"{first.next_reproduction_step}."
        )

    # Called after each turn with the agent that just moved. Only pairs including
    # that agent need checking, because nobody else changed position, and only
    # agents in the nine nearby cells can be close enough.
    # Called with no agent, it checks every living pair.
    # A pair may be detected repeatedly, but is recorded only on its first sighting.
    def detect_encounters(self, mover=None):
        pairs = []
        if mover is None:
            living_agents = list(self.agents)
            for first_index in range(len(living_agents)):
                # Start after first_index so we never check an agent against itself.
                for second_index in range(first_index + 1, len(living_agents)):
                    pairs.append((living_agents[first_index], living_agents[second_index]))
        elif mover.alive:
            for other in self.nearby_agents(mover.position):
                if other is not mover:
                    pairs.append((mover, other))

        for first, second in pairs:
            if first.population == second.population:
                continue

            # Chebyshev distance: the larger row/column difference.
            # 0 = same cell, 1 = adjacent (including diagonals).
            row_distance = abs(first.position[0] - second.position[0])
            column_distance = abs(first.position[1] - second.position[1])
            distance = max(row_distance, column_distance)
            pair = tuple(sorted((first.unique_id, second.unique_id)))
            if distance > 1 or pair in self.encounter_pairs:
                continue

            self.encounter_pairs.add(pair)
            self.encounters_per_timestep += 1
            self.cumulative_encounters += 1
            # Where it happened: "second" is the agent that was already there.
            self.encounter_rows[second.position[0]] += 1
            if self.zones:
                self.encounters_in[self.row_zone[second.position[0]]] += 1
            if self.headless:
                self.confront(first, second)
                continue
            record = {
                "timestep": self.steps,
                "agent_1_id": first.unique_id,
                "agent_2_id": second.unique_id,
                "agent_1_name": first.name,
                "agent_2_name": second.name,
                "population_1": first.population,
                "population_2": second.population,
                "position_1": first.position,
                "position_2": second.position,
                "resources_1": first.resources,
                "resources_2": second.resources,
                "distance": distance,
            }
            self.encounters_this_step.append(record)
            self.encounters.append(record)
            self.messages.append(
                f"Encounter: {first.name} (ID {first.unique_id}, {first.population}) "
                f"at {first.position}, resources {first.resources:.2f}; "
                f"{second.name} (ID {second.unique_id}, {second.population}) "
                f"at {second.position}, resources {second.resources:.2f}. "
                f"Distance {distance}."
            )
            self.confront(first, second)

    # Save one short event for the dashboard's event log. The detailed
    # per-turn explanations stay in self.messages for the debug page.
    def add_event(self, kind, text, agent_ids, position):
        if self.headless:
            return
        self.events.append({
            "timestep": self.steps,
            "kind": kind,
            "text": text,
            "agent_ids": agent_ids,
            "position": position,
        })
        # Keep only recent events so a long run cannot fill up the memory.
        if len(self.events) > 2000:
            del self.events[:1000]

    # Calculate the summary values, then let Mesa save them as one data row.
    def collect_data(self):
        self.environmental_resources = sum(self.food.values())
        # One pass over the living agents, adding up everything that is averaged.
        counts = {"A": 0, "B": 0}
        gene_totals = {"A": dict.fromkeys(self.genes, 0.0), "B": dict.fromkeys(self.genes, 0.0)}
        total_resources = 0.0
        total_age = 0
        total_generation = 0
        for agent in self.agents:
            total_resources += agent.resources
            total_age += agent.age
            total_generation += agent.generation
            counts[agent.population] += 1
            totals = gene_totals[agent.population]
            for trait in self.genes:
                totals[trait] += agent.genome[trait]
        self.population_a = counts["A"]
        self.population_b = counts["B"]
        self.average_resources = 0.0
        self.average_age = 0.0
        self.average_generation = 0.0
        if len(self.agents) > 0:
            self.average_resources = total_resources / len(self.agents)
            self.average_age = total_age / len(self.agents)
            self.average_generation = total_generation / len(self.agents)
        for zone in ["A", "S", "B"]:
            setattr(self, f"encounters_in_{zone}", self.encounters_in[zone])
            setattr(self, f"fights_in_{zone}", self.fights_in[zone])
        self.average_fertility = 1.0
        if self.soil and self.fertile:
            self.average_fertility = sum(self.fertility.values()) / len(self.fertile)
        for population in ["A", "B"]:
            for source, amount in self.food_from[population].items():
                setattr(self, f"{population}_food_{source}", amount)
            for zone, count in self.zone_steps[population].items():
                setattr(self, f"{population}_steps_{zone}", count)
            setattr(self, f"{population}_moves", self.moves[population])
            setattr(self, f"{population}_trades", self.trades[population])
            for trait in self.genes:
                average = 0.0
                if counts[population] > 0:
                    average = gene_totals[population][trait] / counts[population]
                # setattr stores a named value for DataCollector to read.
                setattr(self, f"{population}_average_{trait}", average)
            for action in config.ACTIONS:
                setattr(self, f"{population}_{action}", self.actions_this_step[population][action])
                setattr(self, f"{population}_cumulative_{action}", self.cumulative_actions[population][action])
            farm_count = 0
            for owner in self.farms.values():
                if owner == population:
                    farm_count += 1
            setattr(self, f"{population}_farms", farm_count)
            setattr(self, f"{population}_stolen", self.stolen_this_step[population])
            setattr(self, f"{population}_cumulative_stolen", self.cumulative_stolen[population])
            setattr(self, f"{population}_starvation_deaths", self.deaths[population]["starvation"])
            setattr(self, f"{population}_old_age_deaths", self.deaths[population]["old_age"])
            setattr(self, f"{population}_agent_steps", self.agent_steps[population])
        self.datacollector.collect(self)

    # Advance the whole world by one timestep.
    def step(self):
        self.messages = []
        self.encounters_this_step = []
        self.encounter_pairs = set()
        self.encounters_per_timestep = 0
        self.births_this_step = 0
        self.parents_used_this_step = set()
        self.fights_this_step = 0
        for population in ["A", "B"]:
            self.stolen_this_step[population] = 0.0
            for action in config.ACTIONS:
                self.actions_this_step[population][action] = 0
        # Mesa increases self.steps automatically before this function runs.
        for agent in self.agents:
            agent.start_turn()
            self.agent_steps[agent.population] += 1
            if self.zones:
                zone = self.row_zone[agent.position[0]]
                if zone == "S":
                    self.zone_steps[agent.population]["shared"] += 1
                elif zone == agent.population:
                    self.zone_steps[agent.population]["home"] += 1
                else:
                    self.zone_steps[agent.population]["rival"] += 1
        # Phase 1: shuffle_do calls each living agent's step() once, in a random order.
        self.agents.shuffle_do("step")
        # Phase 2: divide the food in every occupied cell.
        self.share_food()
        # Phase 3: eat, pay costs, age, and check for death.
        # list() copies the agents first, because dying agents leave self.agents.
        for agent in list(self.agents):
            agent.finish_turn()
        # All normal turns are over before any children are created.
        self.reproduce()
        for population in ["A", "B"]:
            if population not in self.extinct and self.count_population(population) == 0:
                self.extinct.add(population)
                self.add_event("extinction", f"Population {population} has died out", [], None)

        self.abandon_farms()

        # Refill each food cell, without going above its capacity.
        # Farms grow at farm_growth; wild patches regrow at regeneration.
        restored = 0.0
        regeneration = self.environment["regeneration"]
        farm_growth = self.environment["farm_growth"]
        for position, capacity in self.capacity.items():
            before = self.food[position]
            if capacity == 0 and before == 0:
                continue  # Barren land: nothing to grow.
            growth = regeneration
            if position in self.farms:
                growth = self.farm_growth[self.farms[position]]
                if self.soil:
                    growth *= self.fertility[position]
            self.food[position] = min(capacity, before + growth)
            restored += self.food[position] - before
        if self.soil:
            self.update_soil()
        else:
            self.harvested = set()
        self.running = len(self.agents) > 0
        if self.headless:
            # Batch runs save a row every record_every timesteps and nothing else.
            if self.steps % self.record_every == 0:
                self.collect_data()
            return
        self.messages.append(f"Food regenerated by {restored:.2f} units after all turns.")
        self.messages.append(
            f"Cross-population encounters this timestep: {self.encounters_per_timestep}; "
            f"cumulative: {self.cumulative_encounters}. "
            f"Total deaths: {self.starvation_deaths} starvation, {self.old_age_deaths} old age."
        )
        self.messages.append(
            f"Births this timestep: {self.births_this_step}; cumulative: {self.cumulative_births}. "
            f"Highest generation reached: {self.highest_generation}."
        )
        self.collect_data()

        # Show the same explanation in Terminal as on the web page.
        # Large worlds skip this: hundreds of lines per timestep are unreadable.
        if self.world["print_turns"]:
            print(f"\n--- Timestep {self.steps} ---", flush=True)
            for message in self.messages:
                print(message, flush=True)
