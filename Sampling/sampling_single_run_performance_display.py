import json

from Sampling.Samplers.LPFirst_Samplers.ForestFireSamplerLPFirst import ForestFireSamplerLPFirst
from Sampling.Samplers.LPFirst_Samplers.RandomWalkSamplerLPFirst import RandomWalkSamplerLPFirst
from Sampling.Samplers.LPFirst_Samplers.RandomWalkerWithPrioritizationSamplerLPFirst import RandomWalkerWithPrioritizationSamplerLPFirst
from Sampling.Samplers.LPFirst_Samplers.RandomEdgeSamplerLPFirst import RandomEdgeSamplerLPFirst
from Sampling.Samplers.LPFirst_Samplers.RandomNodeSamplerLPFirst import RandomNodeSamplerLPFirst
from Sampling.Samplers.LPFirst_Samplers.RandomWalkerJumpsWithPrioritizationSamplerLPFirst import RandomWalkerJumpsWithPrioritizationSamplerLPFirst
from Sampling.Samplers.LPFirst_Samplers.RandomWalkerJumpsSamplerLPFirst import RandomWalkerJumpsSamplerLPFirst

from Sampling.Samplers.RandomWalkSamplerLPCentralized import RandomWalkSamplerLPCentralized
from Sampling.Samplers.RandomWalkerJumpsSamplerLPCentralized import RandomWalkerJumpsSamplerLPCentralized
from Sampling.Samplers.ForestFireSampler import ForestFireSampler
from Sampling.Samplers.RandomWalkSampler import RandomWalkSampler
from Sampling.Samplers.RandomWalkerJumpsSampler import RandomWalkerJumpsSampler
from Sampling.Samplers.RandomWalkerJumpsWithPrioritizationSampler import RandomWalkerJumpsWithPrioritizationSampler
from Sampling.Samplers.RandomWalkerWithPrioritizationSampler import RandomWalkerWithPrioritizationSampler
from Sampling.Samplers.NodeSampler import NodeSampler
from Sampling.Samplers.RandomEdgeSampler import RandomEdgeSampler
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import EvoLearner
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import F1, Accuracy
from owlapy.model import OWLNamedIndividual, IRI
from ontolearn.utils import setup_logging

setup_logging()

with open('premier-league_lp.json') as json_file:    # <--- set dataset you want to sample here (e.g. carcinogenesis)
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])

sampler = RandomWalkSamplerLPFirst(kb)    # <--- set the sampler here (e.g. RandomWalkSampler)

sampled_kb = sampler.sample(3000, "premier-league_lp.json")    # <--- set the number of nodes (edges if RandomEdgeSampler) and other hyperparameters here.
removed_individuals = sampler.get_removed_nodes()

for str_target_concept, examples in settings['lp1'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    for individual in removed_individuals:
        individual_as_str = individual.get_iri().as_str()
        if individual_as_str in p:
            p.remove(individual_as_str)
        if individual_as_str in n:
            n.remove(individual_as_str)
    typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
    typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
    lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

    model = EvoLearner(knowledge_base=sampled_kb, max_runtime=600, quality_func=F1())
    model.fit(lp, verbose=False)

    model.save_best_hypothesis(n=3, path='Predictions_{0}'.format(str_target_concept))
    # Get Top n hypotheses
    hypotheses = list(model.best_hypotheses(n=3))
    # Use hypotheses as binary function to label individuals.
    predictions = model.predict(individuals=list(typed_pos | typed_neg),
                                hypotheses=hypotheses)

    [print(_) for _ in hypotheses]

    # Measuring F1-score and Accuracy in the original graph using the hypotheses generated in the sampled graph.
    kb = KnowledgeBase(path=settings['data_path'])

    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
    typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
    lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
    encoded_lp = kb.encode_learning_problem(lp)

    for hypothesis in hypotheses:
        f1 = kb.evaluate_concept(hypothesis.concept, F1(), encoded_lp)
        accuracy = kb.evaluate_concept(hypothesis.concept, Accuracy(), encoded_lp)
        print(f'F1: {f1.q} Accuracy: {accuracy.q}')

    print("Done")

