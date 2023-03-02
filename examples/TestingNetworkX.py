import networkx as nx
from littleballoffur.node_sampling import RandomNodeSampler
G = nx.Graph()

sampler = RandomNodeSampler()

G.add_node(1)
G.add_node(2)
G.add_node(3)

G.add_edge(1, 2)
G.add_edge(1, 3) ## proves that you dont need to add a node directly ... when you add the edge it automatically ads a node if it is not present

print(G.number_of_nodes())

#new_graph = sampler.sample(G)

#print(new_graph)\



############################################## Actual code of testing concepts learning with evolearner

""" 
import json
import os
import numpy as np

from littleballoffur.node_sampling import RandomNodeSampler



# NetworkX start
import networkx as nx

G = nx.Graph()

# NetworkX end

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import EvoLearner
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.model import OWLClass, OWLNamedIndividual, IRI
from ontolearn.utils import setup_logging

from owlapy.model import OWLObjectProperty, OWLObjectPropertyAssertionAxiom, OWLClassAssertionAxiom

setup_logging()

try:
    os.chdir("examples")
except FileNotFoundError:
    pass

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])

onto = kb.ontology()
manager = onto.get_owl_ontology_manager()
reasoner = kb.reasoner()

ind = kb.all_individuals_set()
ind = sorted(ind)

sampler = RandomNodeSampler()
objectProperties = onto.object_properties_in_signature()

classes_list =["Brother", "Child", "Daughter", "Father", "Female", "GrandChild", "GrandDaughter", "GrandFather", "GrandMother", "GrandParent", "Grandson", "Male", "Mother", "Parent", "Person", "PersonWithASibling", "Sister", "Son"]

Dictionary_1 = []
Dictionary_index_key = 0

for i in classes_list:
    Dictionary_1.append([Dictionary_index_key, i])
    Dictionary_index_key += 1

# what i will be doing now is make a dict, or a 2xn list where 1st value will be node index and 2nd will be the value (from family data set)

def get_iristring(x):
    return x.get_iri().as_str()



for i in ind:
    G.add_node(i)
    i_i = get_iristring(i)
    Dictionary_1.append([Dictionary_index_key, i_i[32:]])
    Dictionary_index_key += 1

    typesOfI = reasoner.types(i)
    objectProperties = onto.object_properties_in_signature()
    print(i, typesOfI)
    for j in objectProperties:
        #G.add_edge(i, j)
        #print(j)
        # print("Was ist dis ? ", j)
        op_of_1j = reasoner.object_property_values(i, j)

        if op_of_1j is not None:

            for k in op_of_1j:
                print("k")
                #Dictionary_1[Dictionary_index_key] = k
                #Dictionary_index_key += 1

                # print("and what is this ?", k)
                # G.add_edge(i, k) # you dont need to add a node directly ... when you add the edge it automatically adds a node if it is not present

#H = nx.Graph()
# H.add_nodes_from(sorted(G.nodes(data=True)))
# H.add_edges_from(G.edges(data=True))
# print("H sorted =", H)
# print(nx.is_connected(H))

for i in Dictionary_1:
    print(i[0],i[1])

"""

"""
print("############")
a = list(G.nodes)
b = a[0].get_iri().as_str() #and then you just do substring
print(a)
print(b[32:])
mycount = 0
print(nx.is_connected(G))
"""
#sd

"""
# @This code was used to find the total number of char in the iri string till #
for i in b:
    if i != "#":
        mycount += 1
    else :
        break
print(mycount)
print(type(b))

"""



#print([e for e in G.edges])
"""
for e in G.edges:
    print(e)
    print()
"""
# print("nodes = ", G.nodes)
#
# nx.topological_sort(G)
#
# print("sorted nodes = ", G.nodes)
#
#
####new_graph = sampler.sample(G)
#print(new_graph)


# print("no of nodes in the NetworkX graph = ", G.number_of_nodes())
# print("::::::::::::::::::::::::::::::::::::::::::::::")
# print("no of edges in the NetworkX graph = ", G.number_of_edges())
# print("total edges in the original graph = ", total_edges_counter)
#
#
# print("########")



"""
Brother
Child
Daughter
Father
Female 
GrandChild
GrandDaughter
GrandFather
GrandMother
GrandParent
Grandson
Male
Mother
Parent
Person
PersonWithASibling
Sister
Son

"""
