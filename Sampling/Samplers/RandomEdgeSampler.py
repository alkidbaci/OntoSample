import random

from Sampling.Samplers.Sampler import Sampler
from ontolearn.knowledge_base import KnowledgeBase
import logging

logger = logging.getLogger(__name__)


class RandomEdgeSampler(Sampler):
    """
        Implementation of random edge sampling. Creates a subgraph by sampling 'x' amount of edges in the
        graph.

        Args: graph (KnowledgeBase)
    """

    def __init__(self, graph: KnowledgeBase):
        super().__init__(graph)
        self._nodes = self.graph.all_individuals_set()
        self._sampled_nodes_edges = dict()

    def _next_edge(self):
        """
        Adding a single randomly selected edge in the sample set.
        """
        node1 = random.choice(list(self._nodes))
        node2 = self.get_random_neighbor(node1)
        if node2 is None:
            self._next_edge()
            return
        else:
            if node1 not in self._sampled_nodes_edges.keys():
                self._sampled_nodes_edges[node1] = set()
            if not any(n.edge_type == node2.edge_type and n.node == node2.node for n in
                       self._sampled_nodes_edges[node1]):
                self._sampled_nodes_edges[node1].add(node2)

    def sample(self, edges_number: int, data_properties_percentage=1.0) -> KnowledgeBase:
        """
        Sampling nodes with a single random walk.

        :param data_properties_percentage: Percentage of data properties inclusion for each node( values from 0-1 )
        :param edges_number: Number of distinct edges to be sampled.
        :return: Sampled graph.
        """
        total_edges = self.number_of_edges()
        if edges_number > total_edges:
            raise ValueError('The number of edges is too large. Please make sure it '
                             'is smaller than the total number of edges (total edges: {})'.format(total_edges))
        if data_properties_percentage > 1 or data_properties_percentage < 0:
            raise ValueError("Data properties sample percentage must be a value between 1 and 0")

        while sum(len(v) for v in self._sampled_nodes_edges.values()) < edges_number:
            self._next_edge()
        new_graph = self.get_subgraph_by_remove(self._sampled_nodes_edges, data_properties_percentage)
        return new_graph

    def get_removed_nodes(self):
        return set(self._nodes) - set(self._sampled_nodes_edges.keys())

    def number_of_edges(self):
        """
            Counts the number of distinct edges in the graph
        """
        reasoner = self.graph.reasoner()
        ontology = self.graph.ontology()
        individuals = set(self.graph.all_individuals_set())
        object_properties = set(ontology.object_properties_in_signature())
        edge_counter = 0
        for ind in individuals:
            for op in object_properties:
                op_of_ind = reasoner.object_property_values(ind, op)
                if op_of_ind is not None:
                    for obj3ct in op_of_ind:
                        if obj3ct is not None:
                            edge_counter += 1
        return edge_counter
