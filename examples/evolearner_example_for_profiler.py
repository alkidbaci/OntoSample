import json

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import EvoLearner
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import F1, Accuracy
from owlapy.model import OWLNamedIndividual, IRI
from ontolearn.utils import setup_logging

setup_logging()

with open("carcinogenesis_lp.json") as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])

for str_target_concept, examples in settings['lp1'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
    typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
    lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

    model = EvoLearner(knowledge_base=kb, max_runtime=600, quality_func=F1())
    model.fit(lp, verbose=False)

    model.save_best_hypothesis(n=3, path='Predictions_{0}'.format(str_target_concept))
    # Get Top n hypotheses
    hypotheses = list(model.best_hypotheses(n=3))
    # Use hypotheses as binary function to label individuals.
    predictions = model.predict(individuals=list(typed_pos | typed_neg),
                                hypotheses=hypotheses)

    [print(_) for _ in hypotheses]
