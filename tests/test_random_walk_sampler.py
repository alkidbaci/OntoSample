import unittest


class TestRandomWalks(unittest.TestCase):

    def test_get_subgraph(self):
        with open('../examples/synthetic_problems.json') as json_file:
            import json
            settings = json.load(json_file)

        from ontolearn.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(path=settings['data_path'])

        from Sampling.Samplers.RandomWalkSampler import RandomWalkSampler
        sampler = RandomWalkSampler(kb)

        sampled_kb = sampler.sample(5)

        removed_individuals = sampler.get_removed_individuals()
        self.assertTrue(set(removed_individuals).isdisjoint(set(sampled_kb.all_individuals_set())))
