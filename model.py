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
        # A genome is just a dictionary of traits for now.
        # Only exploratory_tendency affects behavior in this version.
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

    # One agent's turn: look around, choose a cell, collect food, pay costs, age.
    def step(self):
        # model.all_agents keeps dead agents, so make sure a dead one never acts.
        if not self.alive:
            return
        row, column = self.position
        visible = []
        neighbors = []

        # Visit the 3 by 3 area around this agent, including its current cell.
        # max() and min() keep the row and column inside the grid edges.
        size = self.model.grid_size
        for nearby_row in range(max(0, row - 1), min(size, row + 2)):
            for nearby_column in range(max(0, column - 1), min(size, column + 2)):
                position = (nearby_row, nearby_column)
                visible.append(position)
                if position != self.position:
                    neighbors.append(position)

        # Record nearby agents for the explanation, without reacting to them yet.
        seen = []
        for agent in self.model.nearby_agents(self.position):
            if agent is not self:
                seen.append(agent.name)

        exploration = self.genome["exploratory_tendency"]
        # random() gives a number from 0 to 1. Comparing it to a probability
        # makes the agent sometimes explore instead of taking the best food.
        if neighbors and self.random.random() < 0.3 * exploration:
            destination = self.random.choice(neighbors)
            reason = "explored instead of choosing the most food"
        else:
            # First find the largest food amount the agent can actually see.
            most_food = 0.0
            for position in visible:
                if self.model.food[position] > most_food:
                    most_food = self.model.food[position]

            if most_food > 0:
                # Several cells may tie. Make a list and pick one randomly.
                choices = []
                for position in visible:
                    if self.model.food[position] == most_food:
                        choices.append(position)
                destination = self.random.choice(choices)
                reason = f"chose the most visible food ({most_food:.2f}); ties are random"
            elif neighbors:
                destination = self.random.choice(neighbors)
                reason = "saw no food, so searched a neighboring cell"
            else:
                destination = self.position
                reason = "had no neighboring cell to move to"

        # Save the old values so we can explain the changes afterward.
        before = self.resources
        old_position = self.position
        self.position = destination
        cellmates = []
        for agent in self.model.cell_agents[destination]:
            if agent is not self:
                cellmates.append(agent.name)

        # Take all the food from the chosen cell. Later agents see it empty.
        collected = self.model.food[destination]
        self.model.food[destination] = 0.0
        movement_cost = 0.0
        if destination != old_position:
            movement_cost = self.model.environment["movement_cost"]
        self.resources += collected - movement_cost - self.model.environment["survival_cost"]
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
                                     [self.unique_id], destination)
            else:
                self.model.add_event("death", f"{self.name} died of old age",
                                     [self.unique_id], destination)

        if destination != old_position:
            movement = f"moved {old_position} → {destination}"
        else:
            movement = f"stayed at {destination}"

        # :.2f means show a number with two digits after the decimal point.
        self.last_action = (
            f"{movement}: {reason}. Saw agents: {', '.join(seen) or 'none'}. "
            f"Others at destination: {', '.join(cellmates) or 'none'}. "
            f"Resources: {before:.2f} + {collected:.2f} food "
            f"− {movement_cost:.2f} movement − {self.model.environment['survival_cost']:.2f} survival "
            f"= {self.resources:.2f}. Age: {self.age}. {self.status}."
        )
        self.model.messages.append(f"{self.name}: {self.last_action}")
        # Observe encounters only. This never changes any agent's decisions.
        self.model.detect_encounters(self)


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
        }
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

    # Living agents in the 3 by 3 area around a position, including that cell.
    def nearby_agents(self, position):
        row, column = position
        found = []
        for nearby_row in range(max(0, row - 1), min(self.grid_size, row + 2)):
            for nearby_column in range(max(0, column - 1), min(self.grid_size, column + 2)):
                found.extend(self.cell_agents[(nearby_row, nearby_column)])
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
            # that is in the nine nearby cells, same population, and still able.
            second = None
            for other in self.nearby_agents(first.position):
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
                f"Distance {distance}. Recorded only; no interaction."
            )
            self.add_event("encounter", f"{first.name} met {second.name}",
                           [first.unique_id, second.unique_id], first.position)

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
        self.datacollector.collect(self)

    # Advance the whole world by one timestep.
    def step(self):
        self.messages = []
        self.encounters_this_step = []
        self.encounter_pairs = set()
        self.encounters_per_timestep = 0
        self.births_this_step = 0
        self.parents_used_this_step = set()
        # Mesa increases self.steps automatically before this function runs.
        # shuffle_do calls each living agent's step() once, in a random order.
        self.agents.shuffle_do("step")
        # All normal turns are over before any children are created.
        self.reproduce()
        for population in ["A", "B"]:
            if population not in self.extinct and self.count_population(population) == 0:
                self.extinct.add(population)
                self.add_event("extinction", f"Population {population} has died out", [], None)

        # Refill each food cell, without going above its original capacity.
        restored = 0.0
        for position in self.food:
            before = self.food[position]
            self.food[position] = min(
                self.capacity[position],
                before + self.environment["regeneration"],
            )
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
