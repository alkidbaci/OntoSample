import unittest
from ontosample.classic_samplers import *
from ontosample.lpc_samplers import *
from ontosample.lpf_samplers import *
from ontolearn_light.knowledge_base import KnowledgeBase


class TestSamplers(unittest.TestCase):
    kb = KnowledgeBase(path="KGs/Hepatitis/hepatitis.owl")
    pos = {OWLNamedIndividual("http://dl-learner.org/res/patient346"),
           OWLNamedIndividual("http://dl-learner.org/res/patient345")}
    neg = {OWLNamedIndividual("http://dl-learner.org/res/patient344"),
           OWLNamedIndividual("http://dl-learner.org/res/patient34")}
    lp = pos.union(neg)

    def test_rn(self):
        sampler = RandomNodeSampler(self.kb)
        sampled_kb = sampler.sample(50)
        self.assertEqual(len(list(sampled_kb.ontology.individuals_in_signature())), 50)

    def test_re(self):
        sampler = RandomEdgeSampler(self.kb)
        sampled_kb = sampler.sample(50)
        self.assertIn(len(list(sampled_kb.ontology.individuals_in_signature())), [50, 51])

    def test_rw(self):
        sampler = RandomWalkSampler(self.kb)
        sampled_kb = sampler.sample(2)
        self.assertEqual(len(list(sampled_kb.ontology.individuals_in_signature())), 2)

    def test_rwj(self):
        sampler = RandomWalkerJumpsSampler(self.kb)
        sampled_kb = sampler.sample(50)
        self.assertEqual(len(list(sampled_kb.ontology.individuals_in_signature())), 50)

    def test_rwp(self):
        sampler = RandomWalkerWithPrioritizationSampler(self.kb)
        sampled_kb = sampler.sample(2)
        self.assertEqual(len(list(sampled_kb.ontology.individuals_in_signature())), 2)

    def test_rwjp(self):
        sampler = RandomWalkerJumpsWithPrioritizationSampler(self.kb)
        sampled_kb = sampler.sample(50)
        self.assertEqual(len(list(sampled_kb.ontology.individuals_in_signature())), 50)

    def test_ff(self):
        sampler = ForestFireSampler(self.kb)
        sampled_kb = sampler.sample(50)
        self.assertEqual(len(list(sampled_kb.ontology.individuals_in_signature())), 50)

    # ====================================== LPC ======================================

    def test_rn_lpc(self):
        sampler = RandomNodeSamplerLPCentralized(self.kb, self.lp)
        sampled_kb = sampler.sample(50)
        individuals = list(sampled_kb.ontology.individuals_in_signature())
        self.assertEqual(len(individuals), 50)
        for i in self.lp:
            self.assertIn(i, individuals)

    def test_re_lpc(self):
        sampler = RandomEdgeSamplerLPCentralized(self.kb, self.lp)
        sampled_kb = sampler.sample(50)
        individuals = list(sampled_kb.ontology.individuals_in_signature())
        self.assertIn(len(individuals), [50, 51])
        for i in self.lp:
            self.assertIn(i, individuals)

    def test_rw_lpc(self):
        sampler = RandomWalkSamplerLPCentralized(self.kb, self.lp)
        sampled_kb = sampler.sample(4)
        individuals = list(sampled_kb.ontology.individuals_in_signature())
        self.assertEqual(len(individuals), 4)
        cnt = 0
        for i in self.lp:
            if i in individuals:
                cnt += 1
        self.assertGreaterEqual(cnt, 1)

    def test_rwj_lpc(self):
        sampler = RandomWalkerJumpsSamplerLPCentralized(self.kb, self.lp)
        sampled_kb = sampler.sample(50)
        individuals = list(sampled_kb.ontology.individuals_in_signature())
        self.assertEqual(len(individuals), 50)
        for i in self.lp:
            self.assertIn(i, individuals)

    def test_rwp_lpc(self):
        sampler = RandomWalkerWithPrioritizationSamplerLPCentralized(self.kb, self.lp)
        sampled_kb = sampler.sample(4)
        individuals = list(sampled_kb.ontology.individuals_in_signature())
        self.assertEqual(len(individuals), 4)
        cnt = 0
        for i in self.lp:
            if i in individuals:
                cnt += 1
        self.assertGreaterEqual(cnt, 1)

    def test_rwjp_lpc(self):
        sampler = RandomWalkerJumpsWithPrioritizationSamplerLPCentralized(self.kb, self.lp)
        sampled_kb = sampler.sample(50)
        individuals = list(sampled_kb.ontology.individuals_in_signature())
        self.assertEqual(len(individuals), 50)
        for i in self.lp:
            self.assertIn(i, individuals)

    def test_ff_lpc(self):
        sampler = ForestFireSamplerLPCentralized(self.kb, self.lp)
        sampled_kb = sampler.sample(50)
        individuals = list(sampled_kb.ontology.individuals_in_signature())
        self.assertEqual(len(individuals), 50)
        for i in self.lp:
            self.assertIn(i, individuals)

    # ====================================== LPF ======================================

    def test_rn_lpf(self):
        sampler = RandomNodeSamplerLPFirst(self.kb, self.lp)
        sampled_kb = sampler.sample(50)
        individuals = list(sampled_kb.ontology.individuals_in_signature())
        self.assertEqual(len(individuals), 50)
        for i in self.lp:
            self.assertIn(i, individuals)

    def test_re_lpf(self):
        sampler = RandomEdgeSamplerLPFirst(self.kb, self.lp)
        sampled_kb = sampler.sample(50)
        individuals = list(sampled_kb.ontology.individuals_in_signature())
        self.assertIn(len(individuals), [50, 51])
        for i in self.lp:
            self.assertIn(i, individuals)

    def test_rw_lpf(self):
        sampler = RandomWalkSamplerLPFirst(self.kb, self.lp)
        sampled_kb = sampler.sample(4)
        individuals = list(sampled_kb.ontology.individuals_in_signature())
        self.assertEqual(len(individuals), 4)
        cnt = 0
        for i in self.lp:
            if i in individuals:
                cnt += 1
        self.assertGreaterEqual(cnt, 1)

    def test_rwj_lpf(self):
        sampler = RandomWalkerJumpsSamplerLPFirst(self.kb, self.lp)
        sampled_kb = sampler.sample(50)
        individuals = list(sampled_kb.ontology.individuals_in_signature())
        self.assertEqual(len(individuals), 50)

    def test_rwp_lpf(self):
        sampler = RandomWalkerWithPrioritizationSamplerLPFirst(self.kb, self.lp)
        sampled_kb = sampler.sample(4)
        individuals = list(sampled_kb.ontology.individuals_in_signature())
        self.assertEqual(len(individuals), 4)
        cnt = 0
        for i in self.lp:
            if i in individuals:
                cnt += 1
        self.assertGreaterEqual(cnt, 1)

    def test_rwjp_lpf(self):
        sampler = RandomWalkerJumpsWithPrioritizationSamplerLPFirst(self.kb, self.lp)
        sampled_kb = sampler.sample(50)
        individuals = list(sampled_kb.ontology.individuals_in_signature())
        self.assertEqual(len(individuals), 50)
        for i in self.lp:
            self.assertIn(i, individuals)

    # ====================================== other tests ======================================

    def test_dp_sampling(self):

        def test_dp_count(smp, i):
            sampled_kb = smp.sample(5, 0.5)
            onto = sampled_kb.ontology
            reasoner = sampled_kb.reasoner
            cnt = 0
            for dp in onto.data_properties_in_signature():
                dp_of_node = reasoner.data_property_values(i, dp)
                if dp_of_node is not None:
                    cnt += len(list(dp_of_node))
            print(cnt)
            self.assertEqual(cnt, 5)

        target_ind = OWLNamedIndividual("http://dl-learner.org/res/screening3279")
        sampler = RandomNodeSamplerLPFirst(self.kb, {target_ind})
        sampler2 = RandomWalkSamplerLPFirst(self.kb, {target_ind})

        test_dp_count(sampler, target_ind)
        test_dp_count(sampler2, target_ind)

    def test_saving(self):
        sampler = RandomNodeSampler(self.kb)
        sampled_kg = sampler.sample(20)
        sampler.save_sample(sampled_kg, "sampled_kb")
        onto1 = sampled_kg.ontology
        kg2 = KnowledgeBase(path="sampled_kb.owl")
        onto2 = kg2.ontology

        self.assertEqual(list(onto1.individuals_in_signature()), list(onto2.individuals_in_signature()))
        self.assertEqual(list(onto1.classes_in_signature()), list(onto2.classes_in_signature()))
        self.assertEqual(list(onto1.object_properties_in_signature()), list(onto2.object_properties_in_signature()))
        self.assertEqual(list(onto1.data_properties_in_signature()), list(onto2.data_properties_in_signature()))
        self.assertEqual(list(onto1.properties_in_signature()), list(onto2.properties_in_signature()))



