import json
from owlapy.model import IRI
from ontosample.classic_samplers import *
from ontosample.lpc_samplers import *
from ontosample.lpf_samplers import *
from ontolearn.knowledge_base import KnowledgeBase

# 0. Load json that stores the learning problem
with open("learning_problems/uncle_lp.json") as json_file:
    examples = json.load(json_file)

# 1. Initialize knowledge base
kb = KnowledgeBase(path="../KGs/Family/family-benchmark_rich_background.owl")
print(f'Initial individuals: {kb.individuals_count()}')

# 2. Initialize learning problem (required only for LPF and LPC samplers)
pos = set(map(OWLNamedIndividual, map(IRI.create, set(examples['positive_examples']))))
neg = set(map(OWLNamedIndividual, map(IRI.create, set(examples['negative_examples']))))
lp = pos.union(neg)

# 3. Initialize the sampler and generate the sample
sampler = RandomWalkerJumpsSamplerLPCentralized(kb, lp)
sampled_kb = sampler.sample(30)

print(f'Removed individuals: {len(sampler.get_removed_nodes())}')
print(f'Remaining individuals: {sampled_kb.individuals_count()}')

# 4. Save the sampled graph
sampler.save_sample()
