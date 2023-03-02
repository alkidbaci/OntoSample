import json
import os
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import EvoLearner
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.model import OWLClass, OWLNamedIndividual, IRI
from ontolearn.utils import setup_logging
from ontolearn.metrics import Accuracy, F1
from Sampling.Samplers.LPFirst_Samplers.ForestFireSamplerLPFirst import ForestFireSamplerLPFirst
from Sampling.Samplers.LPFirst_Samplers.RandomWalkSamplerLPFirst import RandomWalkSamplerLPFirst
from Sampling.Samplers.LPFirst_Samplers.RandomWalkerWithPrioritizationSamplerLPFirst import \
    RandomWalkerWithPrioritizationSamplerLPFirst
from Sampling.Samplers.LPFirst_Samplers.RandomEdgeSamplerLPFirst import RandomEdgeSamplerLPFirst
from Sampling.Samplers.LPFirst_Samplers.RandomNodeSamplerLPFirst import RandomNodeSamplerLPFirst
from Sampling.Samplers.LPFirst_Samplers.RandomWalkerJumpsWithPrioritizationSamplerLPFirst import \
    RandomWalkerJumpsWithPrioritizationSamplerLPFirst
from Sampling.Samplers.LPFirst_Samplers.RandomWalkerJumpsSamplerLPFirst import RandomWalkerJumpsSamplerLPFirst
from Sampling.Samplers.RandomWalkSamplerLPCentralized import RandomWalkSamplerLPCentralized
from Sampling.Samplers.ForestFireSampler import ForestFireSampler
from Sampling.Samplers.RandomWalkSampler import RandomWalkSampler
from Sampling.Samplers.RandomWalkerJumpsSampler import RandomWalkerJumpsSampler
from Sampling.Samplers.RandomWalkerJumpsWithPrioritizationSampler import RandomWalkerJumpsWithPrioritizationSampler
from Sampling.Samplers.RandomWalkerWithPrioritizationSampler import RandomWalkerWithPrioritizationSampler
from Sampling.Samplers.NodeSampler import NodeSampler
from Sampling.Samplers.RandomEdgeSampler import RandomEdgeSampler
setup_logging()

try:
    os.chdir("examples")
except FileNotFoundError:
    pass
# example for family benchmark
with open('../examples/synthetic_problems.json') as json_file:
    settings = json.load(json_file)

#Performing sampling
kb = KnowledgeBase(path=settings['data_path'])
sampler = RandomWalkerJumpsSampler(kb)
sampled_kb = sampler.sample(50)

removed_individuals = sampler.get_removed_nodes()
hypo_dict = dict()

for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    for individual in removed_individuals:
        individual_as_str = individual.get_iri().as_str()
        if individual_as_str in p:
            p.remove(individual_as_str)
        if individual_as_str in n:
            n.remove(individual_as_str)
    print('Target concept: ', str_target_concept)
    # background info
    NS = 'http://www.benchmark.org/family#'
    if str_target_concept in ['Granddaughter', 'Aunt', 'Sister']:
        concepts_to_ignore = {
            OWLClass(IRI(NS, 'Brother')),
            OWLClass(IRI(NS, 'Sister')),
            OWLClass(IRI(NS, 'Daughter')),
            OWLClass(IRI(NS, 'Mother')),
            OWLClass(IRI(NS, 'Grandmother')),
            OWLClass(IRI(NS, 'Father')),
            OWLClass(IRI(NS, 'Grandparent')),
            OWLClass(IRI(NS, 'PersonWithASibling')),
            OWLClass(IRI(NS, 'Granddaughter')),
            OWLClass(IRI(NS, 'Son')),
            OWLClass(IRI(NS, 'Child')),
            OWLClass(IRI(NS, 'Grandson')),
            OWLClass(IRI(NS, 'Grandfather')),
            OWLClass(IRI(NS, 'Grandchild')),
            OWLClass(IRI(NS, 'Parent')),
        }
        target_kb = sampled_kb.ignore_and_copy(ignored_classes=concepts_to_ignore)
    elif str_target_concept in ['Brother']:
        concepts_to_ignore = {
            OWLClass(IRI(NS, 'Brother')),
            OWLClass(IRI(NS, 'Father')),
            OWLClass(IRI(NS, 'Grandparent')),
            OWLClass(IRI(NS, 'Son')),
            OWLClass(IRI(NS, 'Child')),
            OWLClass(IRI(NS, 'Grandson')),
            OWLClass(IRI(NS, 'Grandfather')),
            OWLClass(IRI(NS, 'Grandchild')),
            OWLClass(IRI(NS, 'Parent')),
        }
        target_kb = sampled_kb.ignore_and_copy(ignored_classes=concepts_to_ignore)
    else:
        target_kb = sampled_kb

    typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
    typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
    lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

    model = EvoLearner(knowledge_base=target_kb, max_runtime=600,quality_func=F1())
    model.fit(lp, verbose=False)

    model.save_best_hypothesis(n=3, path='Predictions_{0}'.format(str_target_concept))
    hypotheses = list(model.best_hypotheses(n=3))
    predictions = model.predict(individuals=list(typed_pos | typed_neg),
                                hypotheses=hypotheses)

    hypo_dict[str_target_concept] = hypotheses

    [print(_) for _ in hypotheses]

# Measuring F1-score and Accuracy for each targed concept using the hypotheses generated in the sampled graph.
kb = KnowledgeBase(path=settings['data_path'])

for str_target_concept,examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
    typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
    encoded_lp = kb.encode_learning_problem(PosNegLPStandard(pos=typed_pos, neg=typed_neg))

    print(f"{str_target_concept}:")
    hypotheses = hypo_dict[str_target_concept]
    for hypothesis in hypotheses:
        f1 = kb.evaluate_concept(hypothesis.concept, F1(), encoded_lp)
        accuracy = kb.evaluate_concept(hypothesis.concept, Accuracy(), encoded_lp)
        print(f'{hypotheses.index(hypothesis)+ 1}. F1: {f1.q} Accuracy: {accuracy.q}')





