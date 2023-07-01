import json
import math
import time


from Sampling.Samplers.LPFirst_Samplers.RandomWalkSamplerLPFirst import RandomWalkSamplerLPFirst
from Sampling.Samplers.LPFirst_Samplers.RandomWalkerWithPrioritizationSamplerLPFirst import \
    RandomWalkerWithPrioritizationSamplerLPFirst
from Sampling.Samplers.LPFirst_Samplers.RandomEdgeSamplerLPFirst import RandomEdgeSamplerLPFirst
from Sampling.Samplers.LPFirst_Samplers.RandomNodeSamplerLPFirst import RandomNodeSamplerLPFirst
from Sampling.Samplers.LPFirst_Samplers.RandomWalkerJumpsWithPrioritizationSamplerLPFirst import \
    RandomWalkerJumpsWithPrioritizationSamplerLPFirst
from Sampling.Samplers.LPFirst_Samplers.RandomWalkerJumpsSamplerLPFirst import RandomWalkerJumpsSamplerLPFirst

from Sampling.Samplers.LPCentralized_Samplers.RandomWalkerJumpsSamplerLPCentralized import \
    RandomWalkerJumpsSamplerLPCentralized
from Sampling.Samplers.LPCentralized_Samplers.RandomWalkerJumpsWithPrioritizationSamplerLPCentralized import \
    RandomWalkerJumpsWithPrioritizationSamplerLPCentralized
from Sampling.Samplers.LPCentralized_Samplers.RandomWalkSamplerLPCentralized import RandomWalkSamplerLPCentralized
from Sampling.Samplers.LPCentralized_Samplers.RandomWalkerWithPrioritizationSamplerLPCentralized import \
    RandomWalkerWithPrioritizationSamplerLPCentralized
from Sampling.Samplers.LPCentralized_Samplers.RandomNodeSamplerLPCentralized import RandomNodeSamplerLPCentralized
from Sampling.Samplers.LPCentralized_Samplers.RandomEdgeSamplerLPCentralized import RandomEdgeSamplerLPCentralized
from Sampling.Samplers.LPCentralized_Samplers.ForestFireSamplerLPCentralized import ForestFireSamplerLPCentralized

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


class EvaluationRow:
    def __init__(self, dataset, sampler, f1_score, f1_standard_deviation, accuracy, accuracy_standard_deviation,
                 average_runtime):
        self.dataset = dataset
        self.sampler = sampler
        self.f1_score = f1_score
        self.f1_standard_deviation = f1_standard_deviation
        self.accuracy = accuracy
        self.accuracy_standard_deviation = accuracy_standard_deviation
        self.average_runtime = average_runtime


# Datasets: "mutagenesis_lp.json","premier-league_lp.json","nctrer_lp.json","hepatitis_lp.json","carcinogenesis_lp.json"
datasets_path = {"mutagenesis_lp.json","premier-league_lp.json","nctrer_lp.json","hepatitis_lp.json","carcinogenesis_lp.json"}

# Samplers: "RNLPC","RWLPC","RWJLPC","RWPLPC","RWJPLPC","RELPC","FFLPC"
samplers = {"RNLPC", "RWLPC", "RWJLPC", "RWPLPC", "RWJPLPC", "RELPC", "FFLPC"}


evaluation_table = list()
sampling_percentage = 0.01  # <-- sampling faction
for path in datasets_path:
    with open(path) as json_file:
        settings = json.load(json_file)

    kb = KnowledgeBase(path=settings['data_path'])
    samples_nr = int(sampling_percentage * len(list(kb.ontology().individuals_in_signature())))
    for smp in samplers:
        if path == "hepatitis_lp.json" and (
                smp == "RW" or smp == "RWP" or smp == "RWLPF" or smp == "RWPLPF" or smp == "RWLPC" or smp == "RWPLPC"):
            continue
        if path == "mutagenesis_lp.json" and (smp == "RW" or smp == "RWP" or smp == "RWLPC" or smp == "RWPLPC"):
            continue
        iterations = 100  # <-- number of iterations
        f1_sum = 0
        accuracy_sum = 0
        QualityList = list()
        QualityList2 = list()
        average_runtime = 0
        for i in range(0, iterations):
            kb = KnowledgeBase(path=settings['data_path'])
            if smp == "RN":
                sampler = NodeSampler(kb)
            elif smp == "RW":
                sampler = RandomWalkSampler(kb)
            elif smp == "RWJ":
                sampler = RandomWalkerJumpsSampler(kb)
            elif smp == "RWP":
                sampler = RandomWalkerWithPrioritizationSampler(kb)
            elif smp == "RE":
                sampler = RandomEdgeSampler(kb)
            elif smp == "RWJP":
                sampler = RandomWalkerJumpsWithPrioritizationSampler(kb)
            elif smp == "FF":
                sampler = ForestFireSampler(kb)
            elif smp == "FFLPC":
                sampler = ForestFireSamplerLPCentralized(kb)
            elif smp == "RWLPF":
                sampler = RandomWalkSamplerLPFirst(kb)
            elif smp == "RWPLPF":
                sampler = RandomWalkerWithPrioritizationSamplerLPFirst(kb)
            elif smp == "RWJLPF":
                sampler = RandomWalkerJumpsSamplerLPFirst(kb)
            elif smp == "RWJPLPF":
                sampler = RandomWalkerJumpsWithPrioritizationSamplerLPFirst(kb)
            elif smp == "RNLPF":
                sampler = RandomNodeSamplerLPFirst(kb)
            elif smp == "RELPF":
                sampler = RandomEdgeSamplerLPFirst(kb)
            elif smp == "RWLPC":
                sampler = RandomWalkSamplerLPCentralized(kb)
            elif smp == "RWJLPC":
                sampler = RandomWalkerJumpsSamplerLPCentralized(kb)
            elif smp == "RWPLPC":
                sampler = RandomWalkerWithPrioritizationSamplerLPCentralized(kb)
            elif smp == "RWJPLPC":
                sampler = RandomWalkerJumpsWithPrioritizationSamplerLPCentralized(kb)
            elif smp == "RNLPC":
                sampler = RandomNodeSamplerLPCentralized(kb)
            elif smp == "RELPC":
                sampler = RandomEdgeSamplerLPCentralized(kb)

            print(f"---------Dataset:{path} Sampler: {smp}----------")
            sampled_kb = sampler.sample(samples_nr, lp_path=path)

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
                start_time = time.time()
                model.fit(lp, verbose=False)
                elapsed_time = round(time.time() - start_time, 4)
                average_runtime += elapsed_time

                # model = CELOE(knowledge_base=sampled_kb)
                # start_time = time.time()
                # model.fit(lp, verbose=False)
                # elapsed_time = round(time.time() - start_time, 4)
                # average_runtime += elapsed_time

                print(f'--Elapsed Time--:{elapsed_time}')
                model.save_best_hypothesis(n=1, path='Predictions_{0}'.format(str_target_concept))
                # Get only the top hypothesis
                hypothesis = list(model.best_hypotheses(n=1))
                predictions = model.predict(individuals=list(typed_pos | typed_neg),
                                            hypotheses=hypothesis)

                print(hypothesis[0])

                # Measuring F1-score and Accuracy in the original graph using the hypotheses generated in the sampled
                # graph.
                kb = KnowledgeBase(path=settings['data_path'])
                p = set(examples['positive_examples'])
                n = set(examples['negative_examples'])
                typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
                typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
                lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
                encoded_lp = kb.encode_learning_problem(lp)

                e = kb.evaluate_concept(hypothesis[0].concept, F1(), encoded_lp)
                e2 = kb.evaluate_concept(hypothesis[0].concept, Accuracy(), encoded_lp)
                print(f'F1: {e.q} Accuracy: {e2.q}')
                QualityList.append(e.q)
                QualityList2.append(e2.q)
                f1_sum += e.q
                accuracy_sum += e2.q
                print(f"Done: {i + 1}")

        # calculating standard deviation
        f1_mean = f1_sum / iterations
        accuracy_mean = accuracy_sum / iterations
        f1_sd = 0
        accuracy_sd = 0
        for q in QualityList:
            d = abs(q - f1_mean)
            d_2 = d * d
            f1_sd += d_2

        f1_sd = f1_sd / iterations
        f1_sd = math.sqrt(f1_sd)

        for q in QualityList2:
            d = abs(q - accuracy_mean)
            d_2 = d * d
            accuracy_sd += d_2
        accuracy_sd = accuracy_sd / iterations
        accuracy_sd = math.sqrt(accuracy_sd)

        print("Quality of concepts generated on the sample graph, tested on the original graph shown by F1-score:")
        print(f"F1 mean value: {f1_mean} | F1 standard deviation: {f1_sd}")
        print()
        print("Quality of concepts generated on the sample graph, tested on the original graph shown by Accuracy:")
        print(f"Accuracy mean value: {accuracy_mean} | Accuracy standard deviation: {accuracy_sd}")

        dataset = path.split("_")[0]
        evaluation_table.append(
            EvaluationRow(dataset, smp, f1_mean, f1_sd, accuracy_mean, accuracy_sd, average_runtime / iterations))
for result in evaluation_table:
    print("Evaluation Results:")
    print(
        f"Dataset: {sampling_percentage * 100}% of {result.dataset} Sampler: {result.sampler} F1={result.f1_score} ± {result.f1_standard_deviation} "
        f"Accuracy={result.accuracy} ± {result.accuracy_standard_deviation} and avg_runtime of Evolearner = {result.average_runtime}")
    print()
