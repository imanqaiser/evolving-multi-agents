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
        self.resources = config.STARTING_RESOURCES
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
            movement_cost = config.MOVEMENT_COST
        self.resources += collected - movement_cost - config.METABOLIC_COST
        self.age += 1

        # Death removes an agent from Mesa's living-agent list.
        # model.all_agents still keeps it so its final state stays on the page.
        if self.resources <= 0:
            self.alive = False
            self.status = "Dead: resources ran out"
        elif self.age >= config.MAXIMUM_AGE:
            self.alive = False
            self.status = f"Dead: reached age {config.MAXIMUM_AGE}"
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
            f"− {movement_cost:.2f} movement − {config.METABOLIC_COST:.2f} survival "
            f"= {self.resources:.2f}. Age: {self.age}. {self.status}."
        )
        self.model.messages.append(f"{self.name}: {self.last_action}")


class SocietyModel(mesa.Model):
    # Create the world and its four agents.
    def __init__(self):
        # A seed makes Reset repeat the same sequence of random choices.
        super().__init__(rng=config.SEED)
        self.food = {}
        for row in range(config.GRID_SIZE):
            for column in range(config.GRID_SIZE):
                self.food[(row, column)] = config.RESOURCE_CAPACITY[row][column]

        self.all_agents = []
        for name, position, exploration in config.STARTING_AGENTS:
            agent = SocietyAgent(self, name, position, exploration)
            self.all_agents.append(agent)

        self.messages = ["Starting world. Press Next timestep to see four actions."]
        # DataCollector saves one row of numbers per timestep for later analysis.
        # These strings name attributes updated in collect_data() below.
        self.datacollector = DataCollector(model_reporters={
            "Timestep": "steps",
            "Population A": "population_a",
            "Population B": "population_b",
            "Environmental resources": "environmental_resources",
            "Average agent resources": "average_resources",
            "Average age": "average_age",
        })
        self.collect_data()

    # Count living agents with the requested population label.
    def count_population(self, population):
        count = 0
        for agent in self.agents:
            if agent.population == population:
                count += 1
        return count

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
        # Mesa increases self.steps automatically before this function runs.
        # shuffle_do calls each living agent's step() once, in a random order.
        self.agents.shuffle_do("step")

        # Refill each food cell, without going above its original capacity.
        restored = 0.0
        for position in self.food:
            row, column = position
            before = self.food[position]
            self.food[position] = min(
                config.RESOURCE_CAPACITY[row][column],
                before + config.REGENERATION_RATE,
            )
            restored += self.food[position] - before
        self.messages.append(f"Food regenerated by {restored:.2f} units after all turns.")
        self.running = len(self.agents) > 0
        self.collect_data()

        # Show the same explanation in Terminal as on the web page.
        print(f"\n--- Timestep {self.steps} ---", flush=True)
        for message in self.messages:
            print(message, flush=True)
