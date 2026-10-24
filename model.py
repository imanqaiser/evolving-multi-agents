# Mesa provides the Model and Agent base classes and records experiment data.
import mesa
from mesa.datacollection import DataCollector

import config


class SocietyAgent(mesa.Agent):
    # __init__ sets up an agent when we create it.
    def __init__(self, model, name, position, exploration=None, genome=None,
                 generation=0, parent_ids=None):
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
        # A genome is a dictionary of four tendencies between 0 and 1:
        #   exploratory_tendency  chance of exploring when nearby food is poor (step)
        #   sociability           chance of ignoring rivals instead of avoiding them (step)
        #   cooperative_tendency  chance of leaving a rival alone instead of robbing it
        #                         (confront) and of respecting rival farms (step)
        #   risk_tolerance        chance of fighting on instead of backing down (confront);
        #                         cautious agents (low risk) tend and plant farms (step, farm)
        if genome is None:
            self.genome = {
                "exploratory_tendency": exploration,
                "risk_tolerance": self.random.random(),
                "sociability": self.random.random(),
                "cooperative_tendency": self.random.random(),
            }
        else:
            # Children receive the inherited dictionary, not new random genes.
            self.genome = genome.copy()
            self.resources = config.CHILD_STARTING_RESOURCES
            self.last_action = f"Born at {position}; first normal turn is next timestep."

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

    # A parent must be alive, old enough, well supplied, and off cooldown.
    def can_reproduce(self):
        return (
            self.alive
            and self.age >= config.MIN_REPRODUCTION_AGE
            and self.resources >= config.REPRODUCTION_THRESHOLD
            and self.model.steps >= self.next_reproduction_step
            and self.unique_id not in self.model.parents_used_this_step
        )

    # Plain-language version of can_reproduce() for the dashboard's inspector.
    # It describes the next timestep, which is when reproduction is checked again.
    def reproduction_status(self):
        if not self.alive:
            return "Dead"
        if self.age < config.MIN_REPRODUCTION_AGE:
            return f"Too young (reproduces from age {config.MIN_REPRODUCTION_AGE})"
        if self.resources < config.REPRODUCTION_THRESHOLD:
            return f"Needs {config.REPRODUCTION_THRESHOLD:.1f} resources"
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
        for agent in self.model.nearby_agents(self.position):
            if agent is not self:
                seen.append(agent.name)
                if agent.population != self.population:
                    rival_positions.append(agent.position)

        # TEND or MOVE ON. On its own population's farm, the agent stays to work
        # it with chance 1 − risk_tolerance: cautious agents settle down, bold
        # ones roam in search of wild food, rival farms to raid, and fights.
        # Agents only stay while saving up: once they have enough to reproduce,
        # they move on, which is how settled farmers meet partners.
        environment = self.model.environment
        on_own_farm = self.model.farms.get(self.position) == self.population
        saving_up = self.resources < config.REPRODUCTION_THRESHOLD
        if on_own_farm and saving_up and self.random.random() < 1 - self.genome["risk_tolerance"]:
            destination = self.position
            reason = "stayed to tend its population's farm"
            rival_note = ""
        else:
            destination, reason, rival_note = self.choose_destination(rival_positions)

        old_position = self.position
        self.position = destination
        # Chebyshev distance: moving one cell diagonally counts as one move.
        distance = max(abs(destination[0] - old_position[0]), abs(destination[1] - old_position[1]))
        self.turn_movement_cost = distance * environment["movement_cost"]
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
            if self.random.random() < self.genome["sociability"]:
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
            if self.random.random() < 1 - self.genome["cooperative_tendency"]:
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
        if poor and far_cells and self.random.random() < self.genome["exploratory_tendency"]:
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

    # Clear what happened last turn. Called for every agent at the start of a
    # timestep, before anyone moves, because an agent can be robbed before its
    # own move.
    def start_turn(self):
        self.turn_start_resources = self.resources
        self.turn_food = 0.0
        self.turn_movement_cost = 0.0
        self.turn_fight_cost = 0.0
        self.turn_stolen = 0.0  # Resources robbed from (−) or by (+) this agent.
        self.turn_movement_note = ""
        self.turn_food_note = ""
        self.turn_meeting_note = ""

    # What the agent holds right now, before this turn's food and costs.
    def current_resources(self):
        return self.resources + self.turn_stolen - self.turn_fight_cost

    # Cells exactly two steps away in one of the eight directions: outside the
    # agent's view, so jumping there is a gamble.
    def cells_two_away(self):
        row, column = self.position
        cells = []
        for row_step in [-1, 0, 1]:
            for column_step in [-1, 0, 1]:
                if row_step == 0 and column_step == 0:
                    continue
                target_row = row + 2 * row_step
                target_column = column + 2 * column_step
                if 0 <= target_row < self.model.grid_size and 0 <= target_column < self.model.grid_size:
                    cells.append((target_row, target_column))
        return cells

    # Phase 3: eat the food received in share_food(), pay costs, age, maybe die,
    # and finally work on or plant a farm.
    def finish_turn(self):
        survival_cost = self.model.environment["survival_cost"]
        self.resources += (self.turn_food + self.turn_stolen - self.turn_movement_cost
                           - self.turn_fight_cost - survival_cost)
        self.age += 1

        # Death removes an agent from Mesa's living-agent list.
        # model.all_agents still keeps it so its final state stays on the page.
        if self.resources <= 0:
            self.alive = False
            self.status = "Dead: resources ran out"
            self.model.starvation_deaths += 1
        elif self.age >= self.model.environment["maximum_age"]:
            self.alive = False
            self.status = f"Dead: reached age {self.model.environment['maximum_age']}"
            self.model.old_age_deaths += 1
        if not self.alive:
            self.remove()
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

        # :.2f means show a number with two digits after the decimal point.
        extra_text = ""
        if self.turn_stolen > 0:
            extra_text += f" + {self.turn_stolen:.2f} stolen"
        elif self.turn_stolen < 0:
            extra_text += f" − {-self.turn_stolen:.2f} stolen from it"
        if self.turn_fight_cost > 0:
            extra_text += f" − {self.turn_fight_cost:.2f} fighting"
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
    # a new farm, with chance 1 − risk_tolerance (cautious agents invest in a
    # safe food supply). Only fertile land can be farmed: a wild patch that is
    # not a farm yet. The agent plants on its own cell, or, when working a farm,
    # on a fertile neighbouring cell. That is how farmland spreads into villages.
    def farm(self):
        model = self.model
        note = ""
        site = None
        if model.farms.get(self.position) == self.population:
            model.farm_last_worked[self.position] = model.steps
            note = " Worked its population's farm."
            free_cells = []
            for position in model.cells_around(self.position):
                if model.can_plant(position):
                    free_cells.append(position)
            if free_cells:
                site = self.random.choice(free_cells)
        elif model.can_plant(self.position):
            site = self.position

        if site is None or self.resources < config.PLANT_MIN_RESOURCES:
            return note, 0.0
        if self.random.random() < 1 - self.genome["risk_tolerance"]:
            self.resources -= config.PLANT_COST
            model.plant_farm(site, self.population, self)
            return note + f" Planted a farm at {site} for population {self.population}.", config.PLANT_COST
        return note, 0.0


class SocietyModel(mesa.Model):
    # Create the world and its starting agents.
    def __init__(self, environment_name=config.DEFAULT_ENVIRONMENT, world_name=config.DEFAULT_WORLD):
        # A seed makes Reset repeat the same sequence of random choices.
        super().__init__(rng=config.SEED)
        if config.CHILD_STARTING_RESOURCES != 2 * config.REPRODUCTION_COST_PER_PARENT:
            raise ValueError("Child starting resources must equal both parents' combined contributions.")
        self.environment_name = environment_name
        self.world_name = world_name
        # Copy the presets so this run keeps its own settings.
        self.environment = config.ENVIRONMENTS[environment_name].copy()
        self.world = config.WORLDS[world_name].copy()
        self.grid_size = self.world["grid_size"]
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
        self.wild_capacity = self.capacity.copy()

        # The SMALL world uses the fixed list in config. Other worlds start A in
        # the top half and B in the bottom half, with fully random genomes.
        if world_name == "SMALL":
            starting_agents = config.STARTING_AGENTS
        else:
            starting_agents = []
            half = self.grid_size // 2
            for population, first_row, last_row in [("A", 0, half), ("B", half, self.grid_size)]:
                for number in range(1, self.world["agents_per_population"] + 1):
                    position = (
                        self.random.randrange(first_row, last_row),
                        self.random.randrange(self.grid_size),
                    )
                    starting_agents.append((f"{population}{number}", position, self.random.random()))

        self.all_agents = []
        self.next_agent_number = {"A": 1, "B": 1}
        for name, position, exploration in starting_agents:
            agent = SocietyAgent(self, name, position, exploration)
            self.all_agents.append(agent)
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
        }
        # One column per population and response, counted per timestep.
        for population in ["A", "B"]:
            for action in config.ACTIONS:
                reporters[f"Population {population} {action}"] = f"{population}_{action}"
            reporters[f"Population {population} farms"] = f"{population}_farms"
            reporters[f"Population {population} stolen"] = f"{population}_stolen"
        # Add one data column for each trait in each population.
        # A reporter string tells Mesa where to read that value on this model.
        for population in ["A", "B"]:
            for trait in self.all_agents[0].genome:
                column = f"Population {population} average {trait}"
                reporters[column] = f"{population}_average_{trait}"
        self.datacollector = DataCollector(model_reporters=reporters)
        self.collect_data()

    # Count living agents with the requested population label.
    def count_population(self, population):
        count = 0
        for agent in self.agents:
            if agent.population == population:
                count += 1
        return count

    # The cells within "radius" steps of a position, including that cell.
    # Radius 1 is the 3 by 3 area an agent can see.
    # max() and min() keep the row and column inside the grid edges.
    def cells_around(self, position, radius=1):
        row, column = position
        cells = []
        for nearby_row in range(max(0, row - radius), min(self.grid_size, row + radius + 1)):
            for nearby_column in range(max(0, column - radius), min(self.grid_size, column + radius + 1)):
                cells.append((nearby_row, nearby_column))
        return cells

    # True if the cell is a farm belonging to the other population.
    def is_rival_farm(self, position, population):
        owner = self.farms.get(position)
        return owner is not None and owner != population

    # Fertile land that nobody farms yet: a wild patch without an owner.
    def can_plant(self, position):
        return self.wild_capacity[position] > 0 and position not in self.farms

    # Turn a wild patch into a farm. It keeps its food and grows faster from now on.
    def plant_farm(self, position, population, planter):
        self.farms[position] = population
        self.farm_last_worked[position] = self.steps
        self.capacity[position] = config.FARM_CAPACITY
        self.farms_planted[population] += 1
        self.add_event("farm", f"{planter.name} planted a farm for population {population}",
                       [planter.unique_id], position)

    # A farm nobody from its population has worked for a while goes back to
    # what it was before: bare land or a wild patch.
    def abandon_farms(self):
        for position in list(self.farms):
            if self.steps - self.farm_last_worked[position] > config.FARM_ABANDON_STEPS:
                population = self.farms[position]
                del self.farms[position]
                del self.farm_last_worked[position]
                self.capacity[position] = self.wild_capacity[position]
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
            for agent in agents:
                agent.turn_food = food / len(agents)
                if len(agents) > 1:
                    agent.turn_food_note += f" Split {food:.2f} food with {len(agents) - 1} other(s)."

            owner = self.farms.get(position)
            if owner is None or food == 0:
                continue
            raiders = []
            defended = False
            for agent in agents:
                if agent.population == owner:
                    defended = True
                else:
                    raiders.append(agent)
                    self.add_stolen(agent.population, agent.turn_food)
            if raiders and not defended:
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
            wealth = min(1.0, max(0.0, agent.current_resources()) / config.REPRODUCTION_THRESHOLD)
            chance = agent.genome["cooperative_tendency"] * (0.5 + 0.5 * wealth)
            if self.random.random() < chance:
                self.count_action(agent.population, "cooperate")
            else:
                self.count_action(agent.population, "compete")
                competing.append(agent)

        if not competing:
            for agent in [first, second]:
                agent.turn_meeting_note += f" Met {first.name if agent is second else second.name} peacefully."
            return
        fought = False
        if len(competing) == 1:
            robber = competing[0]
        else:
            fighting = []
            for agent in competing:
                if self.random.random() < agent.genome["risk_tolerance"]:
                    fighting.append(agent)
            if not fighting:
                for agent in [first, second]:
                    agent.turn_meeting_note += " Both sides threatened, then backed down."
                return
            if len(fighting) == 1:
                robber = fighting[0]
            else:
                fought = True
                weights = []
                for agent in fighting:
                    agent.turn_fight_cost += config.FIGHT_COST
                    weights.append(max(agent.current_resources(), 0.1))
                robber = self.random.choices(fighting, weights=weights)[0]
                self.fights_this_step += 1
                self.cumulative_fights += 1
        victim = second if robber is first else first
        self.rob(robber, victim, fought)

    # Move STEAL_FRACTION of the victim's resources to the robber.
    def rob(self, robber, victim, fought):
        amount = config.STEAL_FRACTION * max(victim.current_resources(), 0.0)
        victim.turn_stolen -= amount
        robber.turn_stolen += amount
        self.add_stolen(robber.population, amount)
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
        for trait in first.genome:
            value = self.random.choice([first.genome[trait], second.genome[trait]])
            if self.random.random() < config.MUTATION_RATE:
                # gauss(0, std) samples a small change centered on zero.
                value += self.random.gauss(0, config.MUTATION_STD)
                self.last_mutations.append(trait)
            # Clamp: stop values going below zero or above one.
            genome[trait] = max(0.0, min(1.0, value))
        return genome

    # Pair nearby eligible parents once, after normal turns have finished.
    def reproduce(self):
        # Copy before adding children. Newborns cannot join this round of pairing.
        candidates = []
        for agent in self.agents:
            if agent.can_reproduce():
                candidates.append(agent)
        self.random.shuffle(candidates)
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
            for other in self.nearby_agents(first.position, config.MATING_RADIUS):
                if other.unique_id not in order or order[other.unique_id] <= order[first.unique_id]:
                    continue
                if other.population != first.population or not other.can_reproduce():
                    continue
                if second is None or order[other.unique_id] < order[second.unique_id]:
                    second = other
            if second is None:
                continue

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
            self.all_agents.append(child)

            # Combined parent contributions become the child's holdings.
            for parent in [first, second]:
                parent.resources -= config.REPRODUCTION_COST_PER_PARENT
                parent.next_reproduction_step = self.steps + config.REPRODUCTION_COOLDOWN
                self.parents_used_this_step.add(parent.unique_id)

            self.births_this_step += 1
            self.cumulative_births += 1
            if population == "A":
                self.population_a_births += 1
            else:
                self.population_b_births += 1
            self.highest_generation = max(self.highest_generation, generation)
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
                f"Parents contributed {config.REPRODUCTION_COST_PER_PARENT:.2f} each; "
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
        living_agents = list(self.agents)
        pairs = []
        if mover is None:
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
            self.encounters_per_timestep += 1
            self.cumulative_encounters += 1
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
        self.population_a = self.count_population("A")
        self.population_b = self.count_population("B")
        self.environmental_resources = sum(self.food.values())
        total_resources = 0.0
        total_age = 0
        total_generation = 0
        for agent in self.agents:
            total_resources += agent.resources
            total_age += agent.age
            total_generation += agent.generation
        self.average_resources = 0.0
        self.average_age = 0.0
        self.average_generation = 0.0
        if len(self.agents) > 0:
            self.average_resources = total_resources / len(self.agents)
            self.average_age = total_age / len(self.agents)
            self.average_generation = total_generation / len(self.agents)
        for population in ["A", "B"]:
            for trait in self.all_agents[0].genome:
                total = 0.0
                count = 0
                for agent in self.agents:
                    if agent.population == population:
                        total += agent.genome[trait]
                        count += 1
                average = 0.0
                if count > 0:
                    average = total / count
                # setattr stores a named value for DataCollector to read.
                setattr(self, f"{population}_average_{trait}", average)
            for action in config.ACTIONS:
                setattr(self, f"{population}_{action}", self.actions_this_step[population][action])
            farm_count = 0
            for owner in self.farms.values():
                if owner == population:
                    farm_count += 1
            setattr(self, f"{population}_farms", farm_count)
            setattr(self, f"{population}_stolen", self.stolen_this_step[population])
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
        for position in self.food:
            before = self.food[position]
            growth = self.environment["regeneration"]
            if position in self.farms:
                growth = self.environment["farm_growth"]
            self.food[position] = min(self.capacity[position], before + growth)
            restored += self.food[position] - before
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
        self.running = len(self.agents) > 0
        self.collect_data()

        # Show the same explanation in Terminal as on the web page.
        # Large worlds skip this: hundreds of lines per timestep are unreadable.
        if self.world["print_turns"]:
            print(f"\n--- Timestep {self.steps} ---", flush=True)
            for message in self.messages:
                print(message, flush=True)
