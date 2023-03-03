import json
import random

from Sampling.Samplers.RandomWalkerWithPrioritizationSampler import RandomWalkerWithPrioritizationSampler
from Sampling.Samplers.Sampler import Sampler, Neighbor
from ontolearn.knowledge_base import KnowledgeBase
import logging

logger = logging.getLogger(__name__)


class RandomWalkerWithPrioritizationSamplerLPCentralized(RandomWalkerWithPrioritizationSampler):
    """
        Implementation of random walker with prioritization sampling LP centralized. Focuses around LP nodes.
    """

    def __init__(self, graph: KnowledgeBase):
        super().__init__(graph)
        self._lpi_backup = None
        self._lpi = None

    def _create_initial_node_set(self):
        """
           Choosing an initial node from LP nodes and add it to the _sampled_nodes_edges dictionary.
        """
        self._current_node = self._lpi.pop()
        self._sampled_nodes_edges = dict()
        self._sampled_nodes_edges[self._current_node] = set()

    def _next_node(self):
        """
        Doing a single centralized prioritized/random step.
        """
        neighbor = self.get_prioritized_neighbor(self._current_node, self._nodes_dict)
        if neighbor is None:
            if self._lpi:
                neighbor = Neighbor(None, self._lpi.pop())
            else:
                neighbor = Neighbor(None, random.choice(list(self._lpi_backup)))
        else:
            if not any(n.edge_type == neighbor.edge_type and n.node == neighbor.node for n in
                       self._sampled_nodes_edges[self._current_node]):
                self._sampled_nodes_edges[self._current_node].add(neighbor)

        if neighbor.node not in self._sampled_nodes_edges.keys():
            self._sampled_nodes_edges[neighbor.node] = set()
        self._current_node = neighbor.node

    def sample(self, nodes_number: int, lp_path, dpp=1.0) -> KnowledgeBase:
        """
        Performs the sampling of the graph.

        :param lp_path: Path of the .json file containing the learning problem
        :param dpp: Percentage of data properties inclusion for each node( values from 0-1 )
        :param nodes_number: Number of distinct nodes to be sampled.
        :return: Sampled graph.
        """
        self._set_pagerank()
        logger.info("Finished setting the PageRanks.")
        self._lpi = list(self.get_lp_individuals(lp_path))
        self._lpi_backup = list(self.get_lp_individuals(lp_path))
        self._create_initial_node_set()
        if nodes_number > len(self._nodes):
            raise ValueError("The number of nodes is too large. Please make sure it "
                             "is smaller than the total number of nodes (total nodes: {})".format(len(self._nodes)))
        if dpp > 1 or dpp < 0:
            raise ValueError("Data properties sample percentage must be a value between 1 and 0")
        stop_centralized_search_threshold = len(self._nodes) * 0.05
        no_new_nodes_counter = 0
        while len(self._sampled_nodes_edges.keys()) < nodes_number:
            current_sampled_node_amount = len(self._sampled_nodes_edges.keys())
            self._next_node()
            post_sampled_node_amount = len(self._sampled_nodes_edges.keys())  # amount of nodes after a single step
            if current_sampled_node_amount == post_sampled_node_amount:
                no_new_nodes_counter += 1
            else:
                no_new_nodes_counter = 0
            if no_new_nodes_counter > stop_centralized_search_threshold:
                self._lpi_backup = list(self._nodes)  # stop restricting the sampler only to LP nodes
        new_graph = self.get_subgraph_by_remove(self._sampled_nodes_edges, dpp)
        return new_graph

    def get_lp_individuals(self, lp_path):
        """
            :param lp_path: path of the .json file that contains the learning problems
            :return: learning problem nodes as individuals (OWLNamedIndividual)
        """
        with open(lp_path) as json_file:
            settings = json.load(json_file)
        prop = list(settings.items())
        for str_target_concept, examples in settings[prop[1][0]].items():
            p = set(examples['positive_examples'])
            n = set(examples['negative_examples'])
            pn = p.union(n)
            lpi = (ind for ind in self._nodes if ind.get_iri().as_str() in pn)
            return lpi
