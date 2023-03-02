import json
import os
import numpy as np
from littleballoffur import RandomNodeSampler, RandomWalkSampler, RandomEdgeSampler, RandomNodeEdgeSampler, \
    HybridNodeEdgeSampler, RandomEdgeSamplerWithInduction, RandomEdgeSamplerWithPartialInduction, \
    NonBackTrackingRandomWalkSampler, PageRankBasedSampler, MetropolisHastingsRandomWalkSampler, \
    CommonNeighborAwareRandomWalkSampler, FrontierSampler

import networkx as nx
from ontolearn.knowledge_base import KnowledgeBase

from ontolearn.utils import setup_logging
from owlapy.model import OWLObjectProperty, OWLObjectPropertyAssertionAxiom, OWLClassAssertionAxiom
import random

setup_logging()

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])
onto = kb.ontology()
manager = onto.get_owl_ontology_manager()
reasoner = kb.reasoner()
ind = kb.all_individuals_set()
ind = sorted(ind)
objectProperties = onto.object_properties_in_signature()


# Function to add classes to the Dictionary
def append_Classes(classes_list1):
    for i in classes_list1:
        Dictionary_1.append(i)


# Function to add the individuals
def append_Individuals(ind1):
    for i in ind1:
        i_i = get_iristring(i)
        Dictionary_1.append(i_i[32:])


# Function to fetch individuals from the IRI string
def get_iristring(x):
    return x.get_iri().as_str()


# Function to add dummy node to the Graph
def add_Dummy_Node_To_Graph(G1, Dictionary_11):
    for i in Dictionary_11:
        G1.add_node(Dictionary_11.index(i))
    return G1


def add_Edges():
    for i in ind:
        i_i = get_iristring(i)
        objectProperties = onto.object_properties_in_signature()
        for j in objectProperties:
            op_of_1j = reasoner.object_property_values(i, j)
            if op_of_1j is not None:
                for k in op_of_1j:
                    k_k = get_iristring(k)
                    G.add_edge(Dictionary_1.index(i_i[32:]), Dictionary_1.index(k_k[32:]))


# Function to connect all the nodes to the dummy node
def add_Dummy_Node(Dictionary_11, G1):
    for i in range(0, len(Dictionary_11) - 1, 1):
        G1.add_edge(Dictionary_11.index(Dictionary_11[i]), Dictionary_11.index(Dictionary_11[-1]))
    return G1


def fitness_values(G1):
    print(f"Graph avg degree: {sum([x[1] for x in G1.degree]) / len(G1.degree):0.4f}")

    # Graph Transitivity or Avg clustering coefficient
    print(f"Graph transitivity : {nx.transitivity(G):0.4f}")

    # Graph Correlation Coefficient
    print(f"Graph degree correlations: {nx.degree_pearson_correlation_coefficient(G):0.4f}")


# Function to do Sampling
def Sampling_Sampler(G1, list_of_sampled_individuals1, seed1, nn):
    flag1 = 0
    # node_sampler = RandomNodeSampler(int(percentage_to_sample1 * G1.number_of_nodes()), seed1)
    print("Inside")
    sampler = RandomNodeSampler(nn, seed1)

    sampled_graph1 = sampler.sample(G1)

    sampled_graph_graph_nodes = sampled_graph1.nodes()

    for ii in range(0, len(Dictionary_1)):
        if ii in sampled_graph_graph_nodes:
            list_of_sampled_individuals1.append(Dictionary_1[ii])

    if dummy_node in list_of_sampled_individuals1:
        flag1 = 0

    return sampled_graph1, flag1


# while looping the function again and again to make sure that the sampled graph does not have the dummy node

if __name__ == '__main__':

    classes_list = ["Brother", "Child", "Daughter", "Father", "Female", "GrandChild", "GrandDaughter", "GrandFather",
                    "GrandMother", "GrandParent", "Grandson", "Male", "Mother", "Parent", "Person",
                    "PersonWithASibling",
                    "Sister", "Son"]

    Dictionary_1 = []
    percentage_to_sample = 0.99
    list_of_sampled_individuals = []
    seed, flag = 1, 1
    dummy_node = "Z Last"
    G = nx.Graph()

    append_Classes(classes_list)
    append_Individuals(ind)
    Dictionary_1.append(dummy_node)  # Adding a last node
    Dictionary_1.sort()
    random.shuffle(Dictionary_1)

    G_with_Dummy_Node = add_Dummy_Node_To_Graph(G, Dictionary_1)
    add_Edges()
    G_with_Dummy_Node = add_Dummy_Node(Dictionary_1, G)
    while flag == 1:
        list_of_sampled_individuals = []
        seed += 1
        # ne = int(0.05 * G.number_of_edges())
        nn = int(0.05 * G.number_of_nodes())
        sampled_graph, flag = Sampling_Sampler(G_with_Dummy_Node, list_of_sampled_individuals, seed, nn)
        print("Flag =", flag)

    print("fitness values of original graph")
    fitness_values(G)

    print("\nfitness values of original graph")
    fitness_values(sampled_graph)
