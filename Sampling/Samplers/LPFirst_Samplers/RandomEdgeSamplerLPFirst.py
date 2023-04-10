import json
import random

from Sampling.Samplers.Sampler import Sampler
from ontolearn.knowledge_base import KnowledgeBase
import logging

logger = logging.getLogger(__name__)


class RandomEdgeSamplerLPFirst(Sampler):
    """
        Implementation of random walk sampling. Creates a subgraph by performing a simple random walk in the
        graph.

        Args: graph (KnowledgeBase)
    """

    def __init__(self, graph: KnowledgeBase):
        super().__init__(graph)
        self._lpi = None
        self._nodes = self.graph.all_individuals_set()
        self._sampled_nodes_edges = dict()

    def _next_edge(self):
        """
        Retrieving a single edge randomly.
        """
        if self._lpi:
            node1 = self._lpi.pop()
        else:
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

    def sample(self, nodes_number: int, lp_path,  data_properties_percentage=1.0) -> KnowledgeBase:
        """
        Sampling nodes with a single random walk.

        :param lp_path: Path of the .json file containing the learning problem
        :param data_properties_percentage: Percentage of data properties inclusion for each node( values from 0-1 )
        :return: Sampled graph.
        """
        if data_properties_percentage > 1 or data_properties_percentage < 0:
            raise ValueError("Data properties sample percentage must be a value between 1 and 0")
        self._lpi = list(self.get_lp_individuals(lp_path))
        while len(self._sampled_nodes_edges.keys()) < nodes_number:
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

    def get_lp_individuals(self, lp_path):
        with open(lp_path) as json_file:
            settings = json.load(json_file)
        prop = list(settings.items())
        for str_target_concept, examples in settings[prop[1][0]].items():
            p = set(examples['positive_examples'])
            n = set(examples['negative_examples'])
            pn = p.union(n)
            lpi = (ind for ind in self._nodes if ind.get_iri().as_str() in pn)
            return lpi
