import json
from owlapy.iri import IRI
from ontosample.classic_samplers import *
from ontosample.lpc_samplers import *
from ontosample.lpf_samplers import *
from ontolearn_light.knowledge_base import KnowledgeBase

# 0. Load json that stores the learning problem
with open("learning_problems/uncle_lp.json") as json_file:
    examples = json.load(json_file)

# 1. Initialize KnowledgeBase object using the path of the ontology
kb = KnowledgeBase(path="../KGs/Family/family-benchmark_rich_background.owl")
print(f'Initial individuals: {kb.individuals_count()}')

# 2. Initialize learning problem (required only for LPF and LPC samplers)
# This does not necessarily need to be the set of learning problem nodes. It can be any set of nodes that you want to
# certainly include in the sample (LPF) or have the sample graph centralized at them (LPC).
pos = set(map(OWLNamedIndividual, map(IRI.create, set(examples['positive_examples']))))
neg = set(map(OWLNamedIndividual, map(IRI.create, set(examples['negative_examples']))))
lp = pos.union(neg)

# 3. Initialize the sampler and generate the sample
sampler = RandomWalkerJumpsSamplerLPCentralized(kb, lp)
sampled_kb = sampler.sample(30)  # will generate a sample with 30 nodes

print(f'Removed individuals: {kb.individuals_count() - sampled_kb.individuals_count()}')
print(f'Remaining individuals: {sampled_kb.individuals_count()}')

# 4. Save the sampled ontology into a file named 'sampled_kb' (this file will be created on runtime)
sampler.save_sample(sampled_kb, 'sampled_kb')
