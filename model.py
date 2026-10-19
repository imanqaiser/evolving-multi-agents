# Mesa provides the Model and Agent base classes and records experiment data.
import mesa
from mesa.datacollection import DataCollector

import config


class SocietyAgent(mesa.Agent):
    # __init__ sets up an agent when we create it.
    def __init__(self, model, name, position, exploration):
        # super() calls Mesa's setup code. Mesa assigns an ID and registers the agent.
        super().__init__(model)
        self.name = name
        self.population = name[0]  # A1 and A2 belong to A; B1 and B2 belong to B.
        self.position = position
        self.resources = model.environment["starting_resources"]
        self.age = 0
        self.alive = True
        self.status = "Alive"
        self.last_action = "Starting position; no action yet."
        # A genome is just a dictionary of traits for now.
        # Only exploratory_tendency affects behavior in this version.
        self.genome = {
            "exploratory_tendency": exploration,
            "risk_tolerance": self.random.random(),
            "sociability": self.random.random(),
            "cooperative_tendency": self.random.random(),
        }

    # One agent's turn: look around, choose a cell, collect food, pay costs, age.
    def step(self):
        row, column = self.position
        visible = []
        neighbors = []

        # Visit the 3 by 3 area around this agent, including its current cell.
        # max() and min() keep the row and column inside the grid edges.
        for nearby_row in range(max(0, row - 1), min(config.GRID_SIZE, row + 2)):
            for nearby_column in range(max(0, column - 1), min(config.GRID_SIZE, column + 2)):
                position = (nearby_row, nearby_column)
                visible.append(position)
                if position != self.position:
                    neighbors.append(position)

        # Record nearby agents for the explanation, without reacting to them yet.
        seen = []
        for agent in self.model.agents:
            if agent is not self and agent.position in visible:
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
        for agent in self.model.agents:
            if agent is not self and agent.position == destination:
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
        self.model.detect_encounters()


class SocietyModel(mesa.Model):
    # Create the world and its four agents.
    def __init__(self, environment_name=config.DEFAULT_ENVIRONMENT):
        # A seed makes Reset repeat the same sequence of random choices.
        super().__init__(rng=config.SEED)
        self.environment_name = environment_name
        # Copy the preset so this run keeps its own settings.
        self.environment = config.ENVIRONMENTS[environment_name].copy()
        self.food = {}
        self.capacity = {}
        for row in range(config.GRID_SIZE):
            for column in range(config.GRID_SIZE):
                self.food[(row, column)] = 0.0
                self.capacity[(row, column)] = 0.0
        for position in self.environment["patches"]:
            self.food[position] = self.environment["food_per_patch"]
            self.capacity[position] = self.environment["food_per_patch"]

        self.all_agents = []
        for name, position, exploration in config.STARTING_AGENTS:
            agent = SocietyAgent(self, name, position, exploration)
            self.all_agents.append(agent)

        self.messages = [f"Starting {environment_name} world. Press Next timestep."]
        self.encounters_this_step = []
        self.encounters = []  # Permanent snapshots from every completed timestep.
        self.encounter_pairs = set()  # A set remembers pairs already seen this step.
        self.encounters_per_timestep = 0
        self.cumulative_encounters = 0
        # Death counts are cumulative for the run, not just the latest timestep.
        self.starvation_deaths = 0
        self.old_age_deaths = 0
        # DataCollector saves one row of numbers per timestep for later analysis.
        # These strings name attributes updated in collect_data() below.
        self.datacollector = DataCollector(model_reporters={
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
        })
        self.collect_data()

    # Count living agents with the requested population label.
    def count_population(self, population):
        count = 0
        for agent in self.agents:
            if agent.population == population:
                count += 1
        return count

    # Check each living pair after a turn, including pairs not involving the mover.
    # A pair may be detected repeatedly, but is recorded only on its first sighting.
    def detect_encounters(self):
        living_agents = list(self.agents)
        for first_index in range(len(living_agents)):
            first = living_agents[first_index]
            # Start after first_index so we never check an agent against itself.
            for second_index in range(first_index + 1, len(living_agents)):
                second = living_agents[second_index]
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

    # Calculate the summary values, then let Mesa save them as one data row.
    def collect_data(self):
        self.population_a = self.count_population("A")
        self.population_b = self.count_population("B")
        self.environmental_resources = sum(self.food.values())
        total_resources = 0.0
        total_age = 0
        for agent in self.agents:
            total_resources += agent.resources
            total_age += agent.age
        self.average_resources = 0.0
        self.average_age = 0.0
        if len(self.agents) > 0:
            self.average_resources = total_resources / len(self.agents)
            self.average_age = total_age / len(self.agents)
        self.datacollector.collect(self)

    # Advance the whole world by one timestep.
    def step(self):
        self.messages = []
        self.encounters_this_step = []
        self.encounter_pairs = set()
        self.encounters_per_timestep = 0
        # Mesa increases self.steps automatically before this function runs.
        # shuffle_do calls each living agent's step() once, in a random order.
        self.agents.shuffle_do("step")

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
        self.running = len(self.agents) > 0
        self.collect_data()

        # Show the same explanation in Terminal as on the web page.
        print(f"\n--- Timestep {self.steps} ---", flush=True)
        for message in self.messages:
            print(message, flush=True)
