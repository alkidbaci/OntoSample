import json
import math
import time
import csv
from argparse import ArgumentParser

from ontosample.classic_samplers import *
from ontosample.lpc_samplers import *

try:
    from ontolearn.knowledge_base import KnowledgeBase
    from ontolearn.learning_problem import PosNegLPStandard
    from ontolearn.metrics import F1, Accuracy
    from ontolearn.concept_learner import EvoLearner, CELOE  # <--- pip install ontolearn
except ModuleNotFoundError:
    from ontolearn_light.knowledge_base import KnowledgeBase
    from ontolearn_light.learning_problem import PosNegLPStandard
    from ontolearn_light.metrics import F1, Accuracy

from owlapy.owl_individual import OWLNamedIndividual, IRI
from ontolearn_light.utils import setup_logging

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


def start(args):
    # Learning problems: "learning_problems/mutagenesis_lp.json",
    #            "learning_problems/premier-league_lp.json",
    #            "learning_problems/nctrer_lp.json",
    #            "learning_problems/hepatitis_lp.json",
    #            "learning_problems/carcinogenesis_lp.json"
    datasets_path = args.datasets

    # Samplers: "RNLPC","RWLPC","RWJLPC","RWPLPC","RWJPLPC","RELPC","FFLPC", "RN", "RW", "RWJ", "RWP", "RWJP",
    # "RE", "FF"

    samplers = args.samplers

    evaluation_table = list()
    sampling_percentage = args.sampling_size  # <-- sampling faction
    with open(args.csv_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset", "Sampler", "Sampling %", "Hypothesis", "F1-score", "Accuracy",
                         "Runtime", "Average F1-score +- standard deviation", "Average Accuracy +- standard deviation",
                         "Average Runtime"])
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
                iterations = args.iterations  # <-- number of iterations
                f1_sum = 0
                accuracy_sum = 0
                quality_list = list()
                quality_list2 = list()
                average_runtime = 0
                is_lpc = False
                for i in range(0, iterations):
                    kb = KnowledgeBase(path=settings['data_path'])
                    if smp == "RN":
                        sampler = RandomNodeSampler(kb)
                        is_lpc = False
                    elif smp == "RW":
                        sampler = RandomWalkSampler(kb)
                        is_lpc = False
                    elif smp == "RWJ":
                        sampler = RandomWalkerJumpsSampler(kb)
                        is_lpc = False
                    elif smp == "RWP":
                        sampler = RandomWalkerWithPrioritizationSampler(kb)
                        is_lpc = False
                    elif smp == "RE":
                        sampler = RandomEdgeSampler(kb)
                        is_lpc = False
                    elif smp == "RWJP":
                        sampler = RandomWalkerJumpsWithPrioritizationSampler(kb)
                        is_lpc = False
                    elif smp == "FF":
                        sampler = ForestFireSampler(kb)
                        is_lpc = False
                    elif smp == "FFLPC":
                        sampler = ForestFireSamplerLPCentralized(kb)
                        is_lpc = True
                    elif smp == "RWLPC":
                        sampler = RandomWalkSamplerLPCentralized(kb)
                        is_lpc = True
                    elif smp == "RWJLPC":
                        sampler = RandomWalkerJumpsSamplerLPCentralized(kb)
                        is_lpc = True
                    elif smp == "RWPLPC":
                        sampler = RandomWalkerWithPrioritizationSamplerLPCentralized(kb)
                        is_lpc = True
                    elif smp == "RWJPLPC":
                        sampler = RandomWalkerJumpsWithPrioritizationSamplerLPCentralized(kb)
                        is_lpc = True
                    elif smp == "RNLPC":
                        sampler = RandomNodeSamplerLPCentralized(kb)
                        is_lpc = True
                    elif smp == "RELPC":
                        sampler = RandomEdgeSamplerLPCentralized(kb)
                        is_lpc = True

                    print(f"---------Dataset:{path} Sampler: {smp}----------")
                    if is_lpc:
                        sampled_kb = sampler.sample(samples_nr, lp_path=path)
                    else:
                        sampled_kb = sampler.sample(samples_nr)

                    removed_individuals = set(kb.individuals()) - set(sampled_kb.individuals())

                    for str_target_concept, examples in settings['lp1'].items():
                        p = set(examples['positive_examples'])
                        n = set(examples['negative_examples'])
                        for individual in removed_individuals:
                            individual_as_str = individual.get_iri().as_str()
                            if individual_as_str in p:
                                p.remove(individual_as_str)
                            if individual_as_str in n:
                                n.remove(individual_as_str)
                        if len(p) == 0:
                            inds = set(sampled_kb.all_individuals_set())
                            typed_pos = set(random.sample(inds, int(len(inds) / 2)))
                            typed_neg = set()
                        else:
                            typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
                            typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
                        lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
                        if args.learner == "evolearner":
                            model = EvoLearner(knowledge_base=sampled_kb, max_runtime=600, quality_func=F1())
                            start_time = time.time()
                            model.fit(lp, verbose=False)
                            elapsed_time = round(time.time() - start_time, 4)
                            average_runtime += elapsed_time
                        else:
                            model = CELOE(knowledge_base=sampled_kb)
                            start_time = time.time()
                            model.fit(lp, verbose=False)
                            elapsed_time = round(time.time() - start_time, 4)
                            average_runtime += elapsed_time

                        print(f'--Elapsed Time--:{elapsed_time}')
                        model.save_best_hypothesis(n=1, path='Predictions_{0}'.format(str_target_concept))
                        # Get only the top hypothesis
                        hypothesis = list(model.best_hypotheses(n=1))
                        predictions = model.predict(individuals=list(typed_pos | typed_neg),
                                                    hypotheses=hypothesis)

                        print(hypothesis[0])

                        # Measuring F1-score and Accuracy in the original graph using the hypotheses generated in
                        # the sampled graph.
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
                        quality_list.append(e.q)
                        quality_list2.append(e2.q)
                        f1_sum += e.q
                        accuracy_sum += e2.q
                        print(f"Done: {i + 1}")
                        writer.writerow([f'{path.split("_")[0]}', smp, f'{sampling_percentage * 100}%', str(hypothesis[0])[54:], e.q, e2.q,
                                         elapsed_time, "", "", ""])
                # calculating standard deviation
                f1_mean = f1_sum / iterations
                accuracy_mean = accuracy_sum / iterations
                f1_sd = 0
                accuracy_sd = 0
                for q in quality_list:
                    d = abs(q - f1_mean)
                    d_2 = d * d
                    f1_sd += d_2

                f1_sd = f1_sd / iterations
                f1_sd = math.sqrt(f1_sd)

                for q in quality_list2:
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
                writer.writerow([dataset, smp, f'{sampling_percentage * 100}%', "", "", "",
                                 "", f'{f1_mean} +- {f1_sd}', f'{accuracy_mean} +- {accuracy_sd}',
                                 f"{average_runtime / iterations}"])
    print("Evaluation Results:")
    for result in evaluation_table:
        print(
            f"Dataset: {sampling_percentage * 100}% of {result.dataset} Sampler: {result.sampler} F1={result.f1_score} ± {result.f1_standard_deviation} "
            f"Accuracy={result.accuracy} ± {result.accuracy_standard_deviation} and avg_runtime of Evolearner = {round(result.average_runtime, 4)}")
        print()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--learner", type=str, default="evolearner", choices={"evolearner", "celoe"})
    parser.add_argument("--datasets_and_lp", nargs='+', default=["learning_problems/mutagenesis_lp.json",
                                                          "learning_problems/premier-league_lp.json",
                                                          "learning_problems/nctrer_lp.json",
                                                          "learning_problems/hepatitis_lp.json",
                                                          "learning_problems/carcinogenesis_lp.json"])
    parser.add_argument("--samplers", nargs='+', default=["RNLPC", "RWLPC", "RWJLPC", "RWPLPC", "RWJPLPC", "RELPC",
                                                          "FFLPC", "RN", "RW", "RWJ", "RWP", "RWJP", "RE",
                                                          "FF"])
    parser.add_argument("--csv_path", type=str, default="evaluation_results.csv")
    parser.add_argument("--sampling_size", type=float, default=0.10, help=" 1 ≥ sampling size > 0")
    parser.add_argument("--iterations", type=int, default=10)

    start(parser.parse_args())
