# Run with: python -m pytest -q
# Tests for the experiment features: seeds, headless mode, the neutral control,
# lifetime records, gene pools, overrides, and the farming gene.
import os
import tempfile
import unittest
from unittest.mock import patch

import config
from experiments.common import load_gene_pool, save_gene_pool
from model import SocietyModel


# Run a model for some timesteps and return it.
def run(steps=150, level=3, world="LARGE", **options):
    model = SocietyModel(level, world, **options)
    for _ in range(steps):
        model.step()
    return model


def table(model):
    return model.datacollector.get_model_vars_dataframe()


class SeedTests(unittest.TestCase):
    def test_same_seed_gives_identical_results(self):
        first = run(seed=7, headless=True)
        second = run(seed=7, headless=True)
        self.assertTrue(table(first).equals(table(second)))
        self.assertEqual(first.lifetime_records, second.lifetime_records)
        self.assertEqual(first.food, second.food)

    def test_different_seeds_differ(self):
        first = run(seed=7, headless=True)
        second = run(seed=8, headless=True)
        self.assertFalse(table(first).equals(table(second)))

    def test_seed_does_not_depend_on_config(self):
        first = run(steps=50, seed=7, headless=True)
        with patch.object(config, "SEED", 999):
            second = run(steps=50, seed=7, headless=True)
        self.assertTrue(table(first).equals(table(second)))

    def test_headless_and_dashboard_modes_have_identical_dynamics(self):
        headless = run(seed=3, headless=True)
        dashboard = run(seed=3, headless=False)
        self.assertTrue(table(headless).equals(table(dashboard)))
        self.assertEqual(headless.food, dashboard.food)
        self.assertEqual(headless.lifetime_records, dashboard.lifetime_records)
        # Headless keeps none of the dashboard-only records.
        self.assertEqual(headless.all_agents, [])
        self.assertEqual(headless.events, [])
        self.assertEqual(headless.encounters, [])
        self.assertGreater(len(dashboard.all_agents), 0)

    def test_record_every_saves_fewer_rows_with_the_same_values(self):
        every = run(steps=100, seed=3, headless=True)
        tenth = run(steps=100, seed=3, headless=True, record_every=10)
        self.assertEqual(len(table(tenth)), 11)
        self.assertTrue(table(every).iloc[::10].reset_index(drop=True).equals(table(tenth)))


class EnvironmentTests(unittest.TestCase):
    def test_levels_share_geography_and_scale_productivity_together(self):
        base = config.ENVIRONMENTS["LEVEL_1"]
        previous = 0
        for level in range(1, 6):
            environment = config.ENVIRONMENTS[f"LEVEL_{level}"]
            self.assertEqual(environment["patch_density"], base["patch_density"])
            self.assertEqual(environment["food_per_patch"], base["food_per_patch"])
            self.assertGreater(environment["regeneration"], previous)
            previous = environment["regeneration"]
            self.assertAlmostEqual(environment["regeneration"] / config.BASE_REGENERATION,
                                   environment["farm_growth"] / config.BASE_FARM_GROWTH)
        # The same seed gives the same land at every level.
        self.assertEqual(SocietyModel(1, "LARGE", seed=5).wild_capacity,
                         SocietyModel(5, "LARGE", seed=5).wild_capacity)

    def test_aliases_point_at_the_ends_of_the_gradient(self):
        self.assertEqual(config.ENVIRONMENTS["SCARCE"], config.ENVIRONMENTS["LEVEL_1"])
        self.assertEqual(config.ENVIRONMENTS["ABUNDANT"], config.ENVIRONMENTS["LEVEL_5"])

    def test_overrides_and_shock(self):
        model = SocietyModel(3, "LARGE", seed=1, overrides={"FIGHT_COST": 9.0, "regeneration": 0.5})
        self.assertEqual(model.param("FIGHT_COST"), 9.0)
        self.assertEqual(model.param("STEAL_FRACTION"), config.STEAL_FRACTION)
        self.assertEqual(model.environment["regeneration"], 0.5)
        self.assertEqual(config.FIGHT_COST, 1.5)  # The override is for this run only.
        with self.assertRaises(ValueError):
            SocietyModel(3, "LARGE", overrides={"NO_SUCH_PARAMETER": 1})
        land = model.wild_capacity.copy()
        model.set_environment(1)
        self.assertEqual(model.environment["farm_growth"], config.ENVIRONMENTS["LEVEL_1"]["farm_growth"])
        self.assertEqual(model.wild_capacity, land)

    def test_farms_can_be_disabled(self):
        model = run(steps=100, seed=1, headless=True, overrides={"FARMS_ENABLED": False})
        self.assertEqual(model.farms_planted, {"A": 0, "B": 0})


class ConservationTests(unittest.TestCase):
    # Robberies only move resources: within a timestep the amounts taken and
    # lost cancel out exactly, apart from floating-point rounding.
    def test_robberies_conserve_resources_during_a_run(self):
        model = SocietyModel(3, "LARGE", seed=11, headless=True)
        original = model.share_food
        totals = []

        def checked_share_food():
            # Called after all moves and confrontations, before anyone dies.
            totals.append(sum(agent.turn_stolen for agent in model.agents))
            original()

        model.share_food = checked_share_food
        for _ in range(150):
            model.step()
        self.assertGreater(model.cumulative_stolen["A"] + model.cumulative_stolen["B"], 0)
        for total in totals:
            self.assertAlmostEqual(total, 0.0, places=9)

    def test_a_fight_destroys_exactly_two_fight_costs(self):
        model = SocietyModel(seed=1)
        first, second = model.all_agents[0], model.all_agents[2]
        for agent in [first, second]:
            agent.resources = 10
            agent.genome.update({"cooperative_tendency": 0, "risk_tolerance": 1})
        model.confront(first, second)
        change = sum(a.turn_stolen - a.turn_fight_cost for a in [first, second])
        self.assertAlmostEqual(change, -2 * config.FIGHT_COST)
        self.assertAlmostEqual(first.total_stolen + second.total_stolen,
                               first.total_lost + second.total_lost)

    def test_reproduction_conserves_resources_during_a_run(self):
        model = SocietyModel(3, "LARGE", seed=11, headless=True)
        original = model.reproduce
        differences = []

        def checked_reproduce():
            before = sum(agent.resources for agent in model.agents)
            original()
            differences.append(sum(agent.resources for agent in model.agents) - before)

        model.reproduce = checked_reproduce
        for _ in range(150):
            model.step()
        self.assertGreater(model.cumulative_births, 0)
        for difference in differences:
            self.assertAlmostEqual(difference, 0.0, places=9)


class GenomeTests(unittest.TestCase):
    def test_genes_stay_between_zero_and_one(self):
        # A huge mutation size pushes genes against the limits all the time.
        model = run(steps=200, seed=2, headless=True, overrides={"MUTATION_STD": 0.8, "MUTATION_RATE": 1.0})
        self.assertGreater(len(model.lifetime_records), 100)
        genomes = [[record[gene] for gene in config.GENES] for record in model.lifetime_records]
        genomes += [list(agent.genome.values()) for agent in model.agents]
        for genome in genomes:
            for value in genome:
                self.assertTrue(0.0 <= value <= 1.0)

    def test_every_agent_has_the_five_genes(self):
        model = SocietyModel(3, "LARGE", seed=2)
        for agent in model.agents:
            self.assertEqual(list(agent.genome), config.GENES)

    def test_farming_depends_on_farming_tendency_not_risk_tolerance(self):
        for risk in [0.0, 1.0]:
            model = SocietyModel(seed=1)
            agent = model.all_agents[0]
            agent.resources = 10
            agent.position = (1, 1)  # A wild patch in the SMALL world.
            agent.genome.update({"risk_tolerance": risk, "farming_tendency": 0})
            agent.farm()
            self.assertEqual(model.farms, {})
            agent.genome["farming_tendency"] = 1
            agent.farm()
            self.assertEqual(model.farms, {(1, 1): "A"})

    def test_lifetime_records(self):
        model = run(steps=200, seed=4, headless=True)
        records = model.lifetime_records
        self.assertEqual(len(records), model.starvation_deaths + model.old_age_deaths)
        self.assertEqual(sum(r["cause"] == "starvation" for r in records), model.starvation_deaths)
        for population in ["A", "B"]:
            deaths = model.deaths[population]
            self.assertEqual(sum(r["population"] == population for r in records),
                             deaths["starvation"] + deaths["old_age"])
        # Every birth gives each of two parents one more offspring.
        offspring = sum(r["offspring"] for r in records) + sum(a.offspring for a in model.agents)
        self.assertEqual(offspring, 2 * model.cumulative_births)
        for record in records:
            self.assertLessEqual(record["lifespan"], config.MAX_AGE)
            self.assertGreaterEqual(record["food_eaten"], 0)
            for gene in config.GENES:
                self.assertIn(gene, record)
        # Robbery gains and losses cancel; farm raids add to "stolen" only.
        stolen = sum(r["stolen"] for r in records) + sum(a.total_stolen for a in model.agents)
        self.assertAlmostEqual(stolen, model.cumulative_stolen["A"] + model.cumulative_stolen["B"], places=6)


class NeutralControlTests(unittest.TestCase):
    def test_neutral_behaviour_is_independent_of_genes(self):
        model = run(steps=50, seed=1, neutral=True)
        for agent in model.agents:
            self.assertIsNot(agent.traits, agent.genome)
            self.assertNotEqual(agent.traits, agent.genome)
        normal = SocietyModel(3, "LARGE", seed=1)
        for agent in normal.agents:
            self.assertIs(agent.traits, agent.genome)

    # With no selection, gene averages only wander: across 10 seeds the mean
    # change of every gene must be within 3 standard errors of zero.
    def test_neutral_control_has_no_directional_trend(self):
        changes = {gene: [] for gene in config.GENES}
        for seed in range(10):
            model = run(steps=500, seed=100 + seed, headless=True, neutral=True, record_every=10)
            data = table(model)
            for population in ["A", "B"]:
                if data[f"Population {population}"].iloc[-1] == 0:
                    continue
                for gene in config.GENES:
                    column = data[f"Population {population} average {gene}"]
                    changes[gene].append(column.iloc[-1] - column.iloc[0])
        for gene, values in changes.items():
            self.assertGreaterEqual(len(values), 10)
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            standard_error = (variance / len(values)) ** 0.5
            self.assertLess(abs(mean), 3 * standard_error, f"{gene} drifted by {mean:+.3f}")


class GenePoolTests(unittest.TestCase):
    def test_save_and_load(self):
        model = run(steps=50, seed=6, headless=True)
        pool = model.living_genomes("A")
        self.assertEqual(len(pool), model.population_a)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "pool.json")
            save_gene_pool(path, pool)
            loaded = load_gene_pool(path)
        self.assertEqual(len(loaded), len(pool))
        for saved, original in zip(loaded, pool):
            for gene in config.GENES:
                self.assertAlmostEqual(saved[gene], original[gene], places=4)

    def test_starting_population_is_drawn_from_the_pool(self):
        special = {gene: 0.123 for gene in config.GENES}
        model = SocietyModel(3, "LARGE", seed=6, gene_pools={"B": [special]})
        for agent in model.agents:
            if agent.population == "B":
                self.assertEqual(agent.genome, special)
                self.assertEqual(agent.resources, model.environment["starting_resources"])
            else:
                self.assertNotEqual(agent.genome, special)

    def test_frozen_genomes_never_change(self):
        model = run(steps=200, seed=6, headless=True, frozen_genomes=True)
        self.assertGreater(model.cumulative_births, 50)
        for population in ["A", "B"]:
            founders = [tuple(genome.values()) for genome in model.founding_pools[population]]
            for genome in model.living_genomes(population):
                self.assertIn(tuple(genome.values()), founders)


if __name__ == "__main__":
    unittest.main()
