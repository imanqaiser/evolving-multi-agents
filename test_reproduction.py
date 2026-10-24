# Run with: python -m unittest -v test_reproduction
# These checks use small, controlled worlds. They do not tune the real presets.
import contextlib
import io
import unittest
from unittest.mock import patch

import config
from model import SocietyModel


class ReproductionTests(unittest.TestCase):
    # Keep one eligible same-population pair, with everyone else removed.
    def make_pair(self):
        model = SocietyModel()
        for agent in model.all_agents[2:]:
            agent.alive = False
            agent.remove()
        first, second = model.all_agents[:2]
        for agent in [first, second]:
            agent.position = (1, 1)
            agent.age = config.MIN_REPRODUCTION_AGE
            agent.resources = config.REPRODUCTION_THRESHOLD
        return model, first, second

    def test_cross_population_cannot_reproduce(self):
        model, first, second = self.make_pair()
        second.population = "B"
        model.reproduce()
        self.assertEqual(model.cumulative_births, 0)

    def test_below_threshold_cannot_reproduce(self):
        model, first, second = self.make_pair()
        first.resources -= 0.01
        model.reproduce()
        self.assertEqual(model.cumulative_births, 0)

    def test_proximity_including_diagonals(self):
        for position, expected in [((1, 1), 1), ((0, 0), 1), ((0, 1), 1), ((2, 2), 0)]:
            model, first, second = self.make_pair()
            first.position = (0, 0)
            second.position = position
            model.reproduce()
            self.assertEqual(model.cumulative_births, expected)

    def test_conservation_counters_and_placement(self):
        model, first, second = self.make_pair()
        second.position = (0, 0)
        before = first.resources + second.resources + sum(model.food.values())
        model.reproduce()
        child = model.all_agents[-1]
        self.assertEqual(first.resources, 3.0)
        self.assertEqual(second.resources, 3.0)
        self.assertEqual(child.resources, 4.0)
        self.assertEqual(sum(a.resources for a in model.agents) + sum(model.food.values()), before)
        self.assertIn(child.position, [first.position, second.position])
        self.assertEqual(child.name, 'A3')
        self.assertEqual(set(child.parent_ids), {first.unique_id, second.unique_id})
        self.assertEqual(child.generation, 1)
        self.assertEqual(child.age, 0)
        self.assertEqual(child.next_reproduction_step, 0)
        self.assertEqual(model.births_this_step, 1)
        self.assertEqual(model.cumulative_births, 1)
        self.assertEqual(model.population_a_births, 1)
        self.assertEqual(model.population_b_births, 0)
        self.assertEqual(model.count_population('A'), 3)
        model.collect_data()
        row = model.datacollector.get_model_vars_dataframe().iloc[-1]
        self.assertEqual(row['Population A'], 3)
        self.assertEqual(row['Average generation'], 1 / 3)

    def test_generation_uses_older_generation(self):
        model, first, second = self.make_pair()
        first.generation = 2
        second.generation = 4
        model.reproduce()
        self.assertEqual(model.all_agents[-1].generation, 5)
        self.assertEqual(model.highest_generation, 5)

    def test_population_b_births_and_empty_population_averages(self):
        model, first, second = self.make_pair()
        first.population = second.population = 'B'
        model.reproduce()
        model.collect_data()
        self.assertEqual(model.population_b_births, 1)
        row = model.datacollector.get_model_vars_dataframe().iloc[-1]
        self.assertEqual(row['Population B births'], 1)
        for trait in first.genome:
            self.assertEqual(row[f'Population A average {trait}'], 0)
            expected = sum(a.genome[trait] for a in model.agents) / 3
            self.assertAlmostEqual(row[f'Population B average {trait}'], expected)

    def test_gene_by_gene_crossover(self):
        model, first, second = self.make_pair()
        for trait in first.genome:
            first.genome[trait] = 0.1
            second.genome[trait] = 0.9
        with patch.object(config, 'MUTATION_RATE', 0):
            with patch.object(model.random, 'choice', side_effect=[0.1, 0.9, 0.1, 0.9]):
                genome = model.make_child_genome(first, second)
        self.assertEqual(list(genome.values()), [0.1, 0.9, 0.1, 0.9])
        self.assertEqual(set(first.genome.values()), {0.1})
        self.assertEqual(set(second.genome.values()), {0.9})

    def test_independent_mutation_and_clamping(self):
        model, first, second = self.make_pair()
        for trait in first.genome:
            first.genome[trait] = second.genome[trait] = 0.5
        with patch.object(model.random, 'random', side_effect=[0.01, 0.9, 0.01, 0.9]):
            with patch.object(model.random, 'gauss', side_effect=[2.0, -2.0]) as normal:
                genome = model.make_child_genome(first, second)
        self.assertEqual(list(genome.values()), [1.0, 0.5, 0.0, 0.5])
        self.assertEqual(normal.call_count, 2)
        normal.assert_called_with(0, config.MUTATION_STD)
        for _ in range(1000):
            self.assertTrue(all(0 <= value <= 1 for value in model.make_child_genome(first, second).values()))

    def test_cooldown_and_parent_once_per_step(self):
        model, first, second = self.make_pair()
        model.reproduce()
        for agent in [first, second]:
            agent.resources = 10
        model.reproduce()
        self.assertEqual(model.cumulative_births, 1)
        model.parents_used_this_step = set()
        model.steps = 4
        model.reproduce()
        self.assertEqual(model.cumulative_births, 1)
        model.steps = 5
        model.reproduce()
        self.assertEqual(model.cumulative_births, 2)
        self.assertEqual(model.all_agents[-1].name, 'A4')

    def test_once_per_step_even_with_zero_cooldown_and_extra_partner(self):
        model = SocietyModel()
        for agent in model.agents:
            agent.population = 'A'
            agent.position = (1, 1)
            agent.age = 3
            agent.resources = 20
        with patch.object(config, 'REPRODUCTION_COOLDOWN', 0):
            model.reproduce()
            model.reproduce()
        self.assertEqual(model.cumulative_births, 2)
        self.assertEqual(len(model.parents_used_this_step), 4)

    def test_minimum_age(self):
        model, first, second = self.make_pair()
        first.age = config.MIN_REPRODUCTION_AGE - 1
        model.reproduce()
        self.assertEqual(model.cumulative_births, 0)
        first.age += 1
        model.reproduce()
        self.assertEqual(model.cumulative_births, 1)

    def test_newborn_does_not_take_birth_turn(self):
        model, first, second = self.make_pair()
        for agent in [first, second]:
            agent.resources = 20
        with contextlib.redirect_stdout(io.StringIO()):
            model.step()
        child = model.all_agents[-1]
        self.assertEqual(child.age, 0)
        self.assertEqual(child.resources, config.CHILD_STARTING_RESOURCES)
        self.assertEqual(model.births_this_step, 1)
        with contextlib.redirect_stdout(io.StringIO()):
            model.step()
        self.assertEqual(child.age, 1)
        self.assertEqual(model.births_this_step, 0)

    def test_starvation_and_old_age(self):
        for cause in ['starvation', 'old_age']:
            model = SocietyModel()
            for position in model.food:
                model.food[position] = 0
            for agent in model.agents:
                if cause == 'starvation':
                    agent.resources = 0.1
                else:
                    agent.resources = 100
                    agent.age = config.MAX_AGE - 1
            with contextlib.redirect_stdout(io.StringIO()):
                model.step()
            self.assertEqual(len(model.agents), 0)
            self.assertEqual(model.starvation_deaths, 4 if cause == 'starvation' else 0)
            self.assertEqual(model.old_age_deaths, 4 if cause == 'old_age' else 0)

    def test_encounters_unchanged_and_exclude_same_population(self):
        model = SocietyModel()
        for agent in model.agents:
            agent.position = (1, 1)
        model.detect_encounters()
        model.detect_encounters()
        self.assertEqual(model.encounters_per_timestep, 4)
        self.assertTrue(all(e['population_1'] != e['population_2'] for e in model.encounters))
        model, first, second = self.make_pair()
        model.reproduce()
        model.detect_encounters()
        self.assertEqual(model.encounters_per_timestep, 0)

    def test_both_presets_and_reproducibility(self):
        for name in ['ABUNDANT', 'SCARCE']:
            first, second = SocietyModel(name), SocietyModel(name)
            with contextlib.redirect_stdout(io.StringIO()):
                for _ in range(100):
                    if not first.running:
                        break
                    first.step()
                    second.step()
                    self.assertEqual(first.food, second.food)
                    self.assertTrue(all(0 <= food <= first.capacity[p] for p, food in first.food.items()))
            self.assertTrue(first.datacollector.get_model_vars_dataframe().equals(
                second.datacollector.get_model_vars_dataframe()))
            self.assertEqual([a.genome for a in first.all_agents], [a.genome for a in second.all_agents])
            self.assertEqual(first.cumulative_births, first.population_a_births + first.population_b_births)
            self.assertEqual(len(first.all_agents), 4 + first.cumulative_births)
            for trait in first.all_agents[0].genome:
                for population in ['A', 'B']:
                    column = f'Population {population} average {trait}'
                    value = first.datacollector.get_model_vars_dataframe()[column].iloc[-1]
                    self.assertTrue(0 <= value <= 1)

    def test_cell_index_matches_positions_in_large_world(self):
        model = SocietyModel('ABUNDANT', 'LARGE')
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(60):
                model.step()
        listed = []
        for position, agents in model.cell_agents.items():
            for agent in agents:
                self.assertEqual(agent.position, position)
                listed.append(agent)
        self.assertEqual(len(listed), len(model.agents))
        self.assertTrue(all(agent.alive for agent in listed))

    def test_events_record_births_and_deaths(self):
        model, first, second = self.make_pair()
        model.reproduce()
        birth = model.events[-1]
        self.assertEqual(birth['kind'], 'birth')
        self.assertEqual(birth['agent_ids'][0], model.all_agents[-1].unique_id)
        for agent in model.agents:
            agent.resources = 0.1
        for position in model.food:
            model.food[position] = 0
        with contextlib.redirect_stdout(io.StringIO()):
            model.step()
        kinds = [event['kind'] for event in model.events]
        self.assertEqual(kinds.count('death'), 3)
        self.assertIn('extinction', kinds)

    def test_invalid_resource_transfer_is_reported(self):
        with patch.object(config, 'CHILD_STARTING_RESOURCES', 5):
            with self.assertRaises(ValueError):
                SocietyModel()


if __name__ == '__main__':
    unittest.main()
