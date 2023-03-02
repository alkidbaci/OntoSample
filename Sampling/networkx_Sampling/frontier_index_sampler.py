from littleballoffur.node_sampling import RandomNodeSampler
from littleballoffur.exploration_sampling import FrontierSampler
from littleballoffur import GraphReader
import numpy as np


reader = GraphReader("FamilyIndexed")

G = reader.get_graph()

Rnumber_of_nodes = int(0.5*G.number_of_nodes())
node_sampler = RandomNodeSampler(number_of_nodes=Rnumber_of_nodes)
frontier_sampler = FrontierSampler(number_of_nodes=Rnumber_of_nodes)


G_nodes = G.number_of_nodes()
G_edges = G.number_of_edges()

print("Nodes = ", G_nodes, " Edges = ",G_edges)


randomnode_sampled_graph = node_sampler.sample(G)

randomnode_sampled_graph_graph_nodes = randomnode_sampled_graph.number_of_nodes()
randomnode_sampled_graph_graph_edges = randomnode_sampled_graph.number_of_edges()


print("Random Node Sampler \n Nodes = ", randomnode_sampled_graph_graph_nodes, " Edges = ", G2_graph_edges)

frontier_sampled = frontier_sampler.sample(G)
frontier_sampled_graph_nodes = frontier_sampled.number_of_nodes()
frontier_sampled_graph_edges = frontier_sampled.number_of_edges()

print("Frontier Sampler \n Nodes = ", frontier_sampled_graph_nodes, " Edges = ", G3_graph_edges)

nodes_list = np.array(list(randomnode_sampled_graph.nodes()))
nodes_list2 = np.array(list(frontier_sampled.nodes()))
nodes_list.sort()
nodes_list2.sort()

print("Random Node Sampler Graph Indexed Nodes are \n",nodes_list)
print("Frontier Sampler Graph Indexed Nodes are \n",nodes_list2)
