import random

from Sampling.Samplers.Sampler import Sampler, Neighbor
from ontolearn.knowledge_base import KnowledgeBase
import logging

logger = logging.getLogger(__name__)


class Node:
    """
        Class structure to save individuals (nodes) of the graph in a way that we can calculate pagerank for each of
        them.
    """
    def __init__(self, IRI):
        self.IRI = IRI
        self.outgoing = []
        self.incoming = []
        self.pagerank = 1.0

    def update_pagerank(self, d, n):
        """
            Updates the pagerank of a single node

            :param d: dumping factor, tha weakly connects all the nodes in the graph
            :param n: total number of nodes in the graph
        """
        incoming = self.incoming
        pr_sum = sum((node.pagerank / len(node.outgoing)) for node in incoming)
        self.pagerank = d / n + (1 - d) * pr_sum


class RandomWalkerWithPrioritizationSampler(Sampler):
    """
        Implementation of random walker with prioritization sampling.
    """

    def _nodes_list_set_up(self):
        """
            creates a dictionary that is used to save each individual(node) in the graph by the new structure "Node" which
            will be used to calculate page rank for each of them.
        """
        self._nodes = self.graph.all_individuals_set()
        self._nodes_dict = dict()
        object_properties = list(self.graph.ontology().object_properties_in_signature())
        reasoner = self.graph.reasoner()
        for i in self._nodes:
            iri = i.get_iri().as_str()
            self._nodes_dict[iri] = Node(iri)
            for op in object_properties:
                op_of_i = reasoner.object_property_values(i, op)
                if op_of_i is not None:
                    for obj3ct in op_of_i:
                        self._nodes_dict[iri].outgoing.append(obj3ct)

        for node in self._nodes_dict.values():
            for child in node.outgoing:
                self._nodes_dict[child.get_iri().as_str()].incoming.append(self._nodes_dict[node.IRI])

    def _distribute_pagerank(self, d):
        """
            Single iteration over all the nodes updating their page rank.

            :param d: dumping factor, tha weakly connects all the nodes in the graph
        """
        for node in self._nodes_dict.values():
            node.update_pagerank(d, len(self._nodes))

    def _set_pagerank(self, d=0.15, iteration=100):
        """
            Iterate x number of times over the graph to update the pagerank of each node to a certain degree.

            :param d: dumping factor, tha weakly connects all the nodes in the graph
            :param iteration: number of iterations that the page rank will update in the graph.
        """
        self._nodes_list_set_up()
        for i in range(iteration):
            self._distribute_pagerank(d)

    def _create_initial_node_set(self):
        """
           Choosing an initial node and add it to the _sampled_nodes_edges dictionary.
        """
        self._current_node = next(iter(self._nodes))
        self._sampled_nodes_edges = dict()
        self._sampled_nodes_edges[self._current_node] = set()

    def _next_node(self):
        """
        Doing a single prioritized/random step.
        """
        neighbor = self.get_prioritized_neighbor(self._current_node, self._nodes_dict)
        if neighbor is None:
            # used the same variable name "neighbor" here to avoid a 2nd repetition of the lines 101-103
            neighbor = Neighbor(None, random.choice(list(self._nodes)))
        else:
            if not any(n.edge_type == neighbor.edge_type and n.node == neighbor.node for n in
                       self._sampled_nodes_edges[self._current_node]):
                self._sampled_nodes_edges[self._current_node].add(neighbor)

        if neighbor.node not in self._sampled_nodes_edges.keys():
            self._sampled_nodes_edges[neighbor.node] = set()
        self._current_node = neighbor.node

    def sample(self, nodes_number: int, dpp=1.0) -> KnowledgeBase:
        """
        Performs the sampling of the graph.

        :param dpp: Percentage of data properties inclusion for each node( values from 0-1 )
        :param nodes_number: Number of distinct nodes to be sampled.
        :return: Sampled graph.
        """
        self._set_pagerank()
        logger.info("Finished setting the PageRanks.")
        self._create_initial_node_set()
        if nodes_number > len(self._nodes):
            raise ValueError("The number of nodes is too large. Please make sure it "
                             "is smaller than the total number of nodes (total nodes: {})".format(len(self._nodes)))
        if dpp > 1 or dpp < 0:
            raise ValueError("Data properties sample percentage must be a value between 1 and 0")

        while len(self._sampled_nodes_edges.keys()) < nodes_number:
            self._next_node()
        new_graph = self.get_subgraph_by_remove(self._sampled_nodes_edges, dpp)
        return new_graph

    def get_removed_nodes(self):
        return set(self._nodes) - set(self._sampled_nodes_edges.keys())
