# Run with: python -m pytest -q
# Tests for Model 2: every new mechanic sits behind a switch, and with all the
# switches off the model is still exactly Model 1.
import os
import unittest

import pandas as pd

import config
from model import SocietyModel

ROOT = os.path.dirname(os.path.abspath(__file__))
ALL_GENES = config.GENES + config.BODY_GENE_NAMES


def run(steps=150, level=4, world="LARGE", **options):
    model = SocietyModel(level, world, **options)
    for _ in range(steps):
        model.step()
    return model


def table(model):
    return model.datacollector.get_model_vars_dataframe()


class Model1StillReproducibleTests(unittest.TestCase):
    # With every switch off, one full Model 1 experiment run (E1, selection,
    # seed 0) must give the numbers saved in results/ before Model 2 existed.
    def test_e1_seed_0_reproduces_the_saved_result(self):
        saved = pd.read_csv(os.path.join(ROOT, "results", "e1_timeseries.csv.gz"))
        saved = saved[(saved["condition"] == "selection") & (saved["seed"] == 0)].reset_index(drop=True)
        model = SocietyModel(3, "EXPERIMENT", seed=0, headless=True, record_every=10)
        for _ in range(2000):
            model.step()
        fresh = table(model).reset_index(drop=True)
        self.assertEqual(len(fresh), len(saved))
        compared = 0
        for column in saved.columns:
            if column in ("experiment", "condition", "level", "seed"):
                continue
            # The saved file keeps five significant digits of every decimal number.
            allowed = saved[column].abs() * 1e-4 + 1e-9
            self.assertTrue(((fresh[column] - saved[column]).abs() <= allowed).all(), column)
            compared += 1
        self.assertGreater(compared, 60)

    def test_defaults_are_model_1(self):
        model = SocietyModel(3, "LARGE", seed=1)
        self.assertEqual(model.genes, config.GENES)
        self.assertEqual(model.selection_mode, "endogenous")
        self.assertFalse(model.body_genes)
        self.assertFalse(model.soil)
        self.assertEqual(model.param("TRADE_BONUS"), 0.0)
        for agent in model.agents:
            self.assertEqual(agent.body_cost, 0.0)

    def test_switched_off_mechanics_leave_no_trace(self):
        model = run(level=3, seed=5, headless=True)
        self.assertEqual(model.trade_created, 0.0)
        self.assertEqual(model.trades, {"A": 0, "B": 0})
        self.assertEqual(set(model.fertility.values()), {1.0})


class SwitchTests(unittest.TestCase):
    def test_version_2_turns_everything_on(self):
        model = SocietyModel("MIDDLE", "LARGE", seed=1, version=2)
        self.assertEqual(model.genes, ALL_GENES)
        self.assertEqual(model.selection_mode, "fitness")
        self.assertTrue(model.body_genes and model.soil)
        self.assertEqual(model.environment_name, f"M2_LEVEL_{config.M2_NAMED_LEVELS['MIDDLE']}")
        for agent in model.agents:
            self.assertEqual(list(agent.genome), ALL_GENES)

    # Each mechanic can be switched on alone, on top of Model 1.
    def test_each_switch_works_alone(self):
        base = table(run(level=3, seed=5, headless=True))
        for overrides in [{"BODY_GENES": True, "VISION_COST": 0.02, "BODY_COST": 0.2},
                          {"SELECTION_MODE": "fitness"},
                          {"SOIL_ENABLED": True, "SOIL_DEPLETION": 0.02, "SOIL_RECOVERY": 0.02},
                          {"TRADE_BONUS": 0.5}]:
            model = run(level=3, seed=5, headless=True, overrides=overrides)
            changed = table(model)
            shared = [column for column in base.columns if column in changed.columns]
            self.assertFalse(base[shared].equals(changed[shared]), overrides)
            self.assertGreater(len(model.agents), 0, overrides)

    def test_unknown_selection_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            SocietyModel(3, "LARGE", seed=1, overrides={"SELECTION_MODE": "random"})

    def test_neutral_flag_selects_the_neutral_mode(self):
        self.assertEqual(SocietyModel(4, "LARGE", seed=1, version=2, neutral=True).selection_mode, "neutral")
        legacy = SocietyModel(3, "LARGE", seed=1, neutral=True)
        self.assertEqual(legacy.selection_mode, "endogenous")  # Model 1's neutral control is unchanged.
        self.assertTrue(legacy.neutral)

    def test_interleaved_start_mixes_the_populations(self):
        model = SocietyModel(4, "EXPERIMENT", seed=1, version=2, overrides={"INTERLEAVED_START": True})
        rows = [agent.position[0] for agent in model.agents if agent.population == "A"]
        self.assertGreater(max(rows), model.grid_size // 2)


class BodyGeneTests(unittest.TestCase):
    def test_sight_radius_follows_the_vision_gene(self):
        model = SocietyModel(4, "LARGE", seed=3, version=2)
        agent = next(iter(model.agents))
        for gene, low, high in [(0.0, 1, 1), (1.0, 4, 4), (1 / 3, 2, 2), (0.5, 2, 3)]:
            agent.traits["vision"] = gene
            radii = [agent.look_radius() for _ in range(400)]
            self.assertEqual((min(radii), max(radii)), (low, high))
        # Halfway between two radii, the average lands halfway too (no plateaus).
        self.assertAlmostEqual(sum(radii) / len(radii), 2.5, delta=0.1)

    def test_agents_never_move_further_than_they_see(self):
        model = SocietyModel(4, "LARGE", seed=3, version=2)
        for _ in range(60):
            before = {agent.unique_id: agent.position for agent in model.agents}
            model.step()
            for agent in model.agents:
                if agent.unique_id not in before:
                    continue
                old = before[agent.unique_id]
                distance = max(abs(agent.position[0] - old[0]), abs(agent.position[1] - old[1]))
                # An explorer jumps to one cell beyond what it saw.
                self.assertLessEqual(distance, agent.sight_radius + 1)

    def test_body_genes_cost_resources_every_step(self):
        model = SocietyModel(4, "LARGE", seed=3, version=2)
        for agent in model.agents:
            expected = (model.param("VISION_COST") * 3 * agent.genome["vision"]
                        + model.param("BODY_COST") * agent.genome["body_size"])
            self.assertAlmostEqual(agent.body_cost, expected)
            self.assertAlmostEqual(agent.strength, config.BODY_BASE + agent.genome["body_size"])

    def test_bigger_agents_win_more_fights_and_more_food(self):
        model = SocietyModel(4, "SMALL", seed=1, version=2)
        big, small = model.all_agents[0], model.all_agents[2]
        for agent, size in [(big, 1.0), (small, 0.0)]:
            agent.genome.update({"cooperative_tendency": 0, "risk_tolerance": 1, "body_size": size})
            agent.strength = config.BODY_BASE + size
        wins = 0
        for _ in range(400):
            for agent in [big, small]:
                agent.resources = 10
                agent.start_turn()
            model.confront(big, small)
            wins += big.turn_stolen > 0
        self.assertAlmostEqual(wins / 400, 0.75, delta=0.07)  # Strength 1.5 against 0.5.
        # Sharing one cell's food: in proportion to strength.
        big.position = small.position = (1, 1)
        model.food[(1, 1)] = 2.0
        model.share_food()
        self.assertAlmostEqual(big.turn_food, 1.5)
        self.assertAlmostEqual(small.turn_food, 0.5)


class SoilTests(unittest.TestCase):
    def test_fertility_stays_between_zero_and_one(self):
        model = SocietyModel(7, "LARGE", seed=2, version=2, headless=True,
                             overrides={"SOIL_DEPLETION": 0.3, "SOIL_RECOVERY": 0.4})
        lowest = 1.0
        for _ in range(200):
            model.step()
            for value in model.fertility.values():
                self.assertTrue(0.0 <= value <= 1.0)
            lowest = min(lowest, min(model.fertility.values()))
        self.assertLess(lowest, 0.5)

    def test_harvested_farm_wears_out_and_recovers_when_left_alone(self):
        model = SocietyModel(4, "SMALL", seed=1, version=2,
                             overrides={"SOIL_DEPLETION": 0.1, "SOIL_RECOVERY": 0.05})
        agent = model.all_agents[0]
        cell = (1, 1)
        agent.position = cell
        model.plant_farm(cell, "A", agent)
        model.food[cell] = 2.0
        model.share_food()
        model.update_soil()
        self.assertAlmostEqual(model.fertility[cell], 0.9)
        model.update_soil()  # Nobody ate from it this time.
        self.assertAlmostEqual(model.fertility[cell], 0.95)
        # Fertility belongs to the cell: it survives the farm being abandoned.
        del model.farms[cell]
        self.assertAlmostEqual(model.fertility[cell], 0.95)

    def test_farm_growth_is_multiplied_by_fertility(self):
        model = SocietyModel(4, "SMALL", seed=1, version=2)
        for agent in list(model.agents):
            agent.alive = False
            agent.remove()
        cell = (1, 1)
        model.farms[cell] = "A"
        model.farm_last_worked[cell] = 10 ** 9
        model.capacity[cell] = config.FARM_CAPACITY
        model.food[cell] = 0.0
        model.fertility[cell] = 0.5
        model.step()
        self.assertAlmostEqual(model.food[cell], 0.5 * model.environment["farm_growth"])


class TradeTests(unittest.TestCase):
    def make_pair(self, bonus):
        model = SocietyModel(4, "SMALL", seed=1, version=2, overrides={"TRADE_BONUS": bonus})
        first, second = model.all_agents[0], model.all_agents[2]
        for agent in [first, second]:
            agent.resources = 10
            agent.genome["cooperative_tendency"] = 1
        return model, first, second

    def test_mutual_cooperation_pays_the_bonus(self):
        model, first, second = self.make_pair(0.5)
        model.confront(first, second)
        self.assertEqual((first.turn_trade, second.turn_trade), (0.5, 0.5))
        self.assertEqual(model.trade_created, 1.0)
        # Not again until the cooldown has passed.
        model.confront(first, second)
        self.assertEqual(model.trade_created, 1.0)

    def test_zero_bonus_is_model_1(self):
        model, first, second = self.make_pair(0.0)
        model.confront(first, second)
        self.assertEqual((first.turn_trade, second.turn_trade), (0.0, 0.0))
        self.assertEqual(model.trade_created, 0.0)

    # Over a whole run, robberies and births only move resources around; the
    # trade bonus is the only thing that creates them from nothing.
    def test_trade_is_the_only_source_of_resources(self):
        model = SocietyModel(4, "LARGE", seed=11, version=2, headless=True, overrides={"TRADE_BONUS": 1.0})
        original_share, original_reproduce = model.share_food, model.reproduce
        checks = []

        def checked_share_food():
            robbed = sum(agent.turn_stolen for agent in model.agents)
            traded = sum(agent.turn_trade for agent in model.agents)
            checks.append((robbed, traded))
            original_share()

        def checked_reproduce():
            before = sum(agent.resources for agent in model.agents)
            original_reproduce()
            checks.append((sum(agent.resources for agent in model.agents) - before, 0.0))

        model.share_food, model.reproduce = checked_share_food, checked_reproduce
        created = 0.0
        for _ in range(150):
            model.step()
        for moved, traded in checks:
            self.assertAlmostEqual(moved, 0.0, places=9)
            created += traded
        self.assertGreater(model.trade_created, 0)
        self.assertAlmostEqual(created, model.trade_created, places=6)


class SelectionTests(unittest.TestCase):
    def test_fitness_counts_only_what_was_gathered(self):
        model = SocietyModel(4, "SMALL", seed=1, version=2)
        agent = model.all_agents[0]
        agent.total_food, agent.total_robbed, agent.total_trade, agent.total_lost = 10.0, 2.0, 1.0, 3.0
        lifetime_need = model.environment["maximum_age"] * model.environment["survival_cost"]
        self.assertAlmostEqual(agent.fitness(), 10.0 / lifetime_need)

    # With three candidates and a tournament of size 3 the fittest always wins;
    # in the neutral control each of the three is picked about equally often.
    def test_tournament_picks_the_fitter_partner_more_often_than_chance(self):
        for neutral, expected in [(False, 1.0), (True, 1 / 3)]:
            model = SocietyModel(4, "LARGE", seed=4, version=2, neutral=neutral)
            agents = [agent for agent in model.agents if agent.population == "A"][:4]
            for agent in list(model.agents):
                if agent not in agents:
                    agent.alive = False
                    agent.remove()
            for index, agent in enumerate(agents):
                agent.position = (5, 5)
                agent.age = 10
                agent.resources = 20
                agent.total_food = float(index)  # The last one is the fittest.
            chooser, best = agents[0], agents[3]
            picks = sum(model.choose_partner(chooser) is best for _ in range(600))
            self.assertAlmostEqual(picks / 600, expected, delta=0.06)

    def test_partner_is_within_the_mating_radius_and_same_population(self):
        model = SocietyModel(4, "LARGE", seed=4, version=2)
        chooser = next(agent for agent in model.agents if agent.population == "A")
        for agent in model.agents:
            agent.age, agent.resources = 10, 20
        for _ in range(50):
            partner = model.choose_partner(chooser)
            if partner is None:
                continue
            self.assertEqual(partner.population, "A")
            distance = max(abs(partner.position[0] - chooser.position[0]),
                           abs(partner.position[1] - chooser.position[1]))
            self.assertLessEqual(distance, model.param("MATING_RADIUS"))

    # No selection: across 10 seeds the mean change of each of the seven genes
    # must be within 3 standard errors of zero.
    def test_neutral_mode_has_no_directional_trend_in_any_gene(self):
        changes = {gene: [] for gene in ALL_GENES}
        for seed in range(10):
            model = run(steps=400, seed=200 + seed, headless=True, neutral=True, record_every=10, version=2)
            data = table(model)
            for population in ["A", "B"]:
                if data[f"Population {population}"].iloc[-1] == 0:
                    continue
                for gene in ALL_GENES:
                    column = data[f"Population {population} average {gene}"]
                    changes[gene].append(column.iloc[-1] - column.iloc[0])
        for gene, values in changes.items():
            self.assertGreaterEqual(len(values), 10)
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            self.assertLess(abs(mean), 3 * (variance / len(values)) ** 0.5, f"{gene} drifted by {mean:+.3f}")

    def test_genes_stay_between_zero_and_one(self):
        model = run(steps=150, seed=2, headless=True, version=2,
                    overrides={"MUTATION_STD": 0.8, "MUTATION_RATE": 1.0})
        for record in model.lifetime_records:
            for gene in ALL_GENES:
                self.assertTrue(0.0 <= record[gene] <= 1.0)


if __name__ == "__main__":
    unittest.main()
