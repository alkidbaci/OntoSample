"""This file  checks the average degree, degree correlation. clustering coefficient
of the original and the sampled graph created by different samplers"""

import numpy as np
import networkx as nx



from littleballoffur import GraphReader, RandomWalkSampler, RandomEdgeSampler, RandomNodeEdgeSampler, \
    HybridNodeEdgeSampler, RandomEdgeSamplerWithInduction, RandomEdgeSamplerWithPartialInduction, \
    NonBackTrackingRandomWalkSampler, RandomEdgeSamplerWithPartialInduction, PageRankBasedSampler, \
    MetropolisHastingsRandomWalkSampler, CommonNeighborAwareRandomWalkSampler, RandomNodeSampler,\
    FrontierSampler, RandomWalkWithRestartSampler



def print_stats(G):
    # sampled_graph_graph_nodes = sampled_graph.number_of_nodes()
    # sampled_graph_graph_edges = sampled_graph.number_of_edges()
    print(f"Graph avg degree: {sum([x[1] for x in G.degree]) / len(G.degree):0.4f}")

    # Graph transitivity is also called average of the clustering coefficient
    print(f"Graph transitivity : {nx.transitivity(G):0.4f}")
    print(f"Graph degree correlations: {nx.degree_pearson_correlation_coefficient(G):0.4f}")

def sg(g):
    print(list(g.nodes))
    print(list(g.edges))





def Sampled_graph_performance(ne, nn, seed:int=42):

     for sampler in [
           RandomEdgeSampler(number_of_edges=ne), RandomEdgeSamplerWithInduction(number_of_edges=ne),
                    RandomEdgeSamplerWithPartialInduction(),
                    RandomNodeSampler(number_of_nodes=nn), FrontierSampler(number_of_nodes=nn),
                    RandomWalkSampler(number_of_nodes=nn),
                    PageRankBasedSampler(number_of_nodes=nn), NonBackTrackingRandomWalkSampler(number_of_nodes=nn),
                    MetropolisHastingsRandomWalkSampler(number_of_nodes=nn),
                    CommonNeighborAwareRandomWalkSampler(number_of_nodes=nn),
                    RandomNodeEdgeSampler(number_of_edges=ne),
                   HybridNodeEdgeSampler(number_of_edges=ne),
                     RandomWalkWithRestartSampler(number_of_nodes=nn, seed=seed)

                    ]:

        sampled_graph = sampler.sample(G)

        print(f"\n\n{sampler.__class__}")
        print(f"Nodes: {sampled_graph.number_of_nodes()}\t Edges: {sampled_graph.number_of_edges()}")
        print_stats(sampled_graph)
        print(f"Connected: {nx.is_connected(sampled_graph)}")
        #sg(sampled_graph)





if __name__ == '__main__':
    reader = GraphReader("Family_disconnected_id_id")




    reader.base_url = "file:///Users/rishigarg/PycharmProjects/sampling/sampling_expts/data/"

    G = reader.get_graph()
    # g_nodes = G.number_of_nodes()
    # g_edges = G.number_of_edges()
    print(f"\n\nORIGINAL GRAPH.")
    print(f"Nodes: {G.number_of_nodes()}\t Edges: {G.number_of_edges()}")
    print(print_stats(G))
    print(f"Connected: {nx.is_connected(G)}")


    # Sampling for Half percent of NODES/EDGES
    # Rnumber_of_nodes = int(0.05*G.number_of_nodes())



    ne = int(0.25 * G.number_of_edges())
    nn = int(0.25 * G.number_of_nodes())

    Sampled_graph_performance(ne, nn, seed=42)
    Sampled_graph_performance(ne, nn, seed=37)



    # edge

    # networkx is the foundation of ball of fur. to compute the metrics in table 3 you can do the following (all based on networkx functions).

    # // *for degree correlation*
    # reference: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.assortativity.degree_pearson_correlation_coefficient.html
    # G = nx.path_graph(4)
    # r = nx.degree_pearson_correlation_coefficient(G)
    # print(f"{r:3.1f}")
    # -0.5

    # // *for average degree*
    # nx.info(G) // this gives the average degree
    # sum(G.degree().values())/float(len(G)) // or you can use this formula.

    # // *for transitivity (also called average of the clustering coefficient)*
    # transitivity = nx.transitivity(graph)


