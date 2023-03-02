import json
import random
import time
from Sampling.Samplers.Sampler import Sampler, Neighbor
from ontolearn.knowledge_base import KnowledgeBase
import logging

logger = logging.getLogger(__name__)


class RandomWalkerJumpsSamplerLPCentralized(Sampler):
    """
        Implementation of random walker jumps sampling learning problem(LP) centralized. Creates a subgraph by walking
        randomly in neighborhood of the LP nodes in the graph. Jump probability helps to escape loops
        (infinite execution).
    """

    def __init__(self, graph: KnowledgeBase):
        super().__init__(graph)
        self._lpi_backup = None
        self._lpi = None
        self.jump_prob = None

    def _create_initial_node_set(self):
        """
            Choosing an initial node from the LP nodes and add it to the _sampled_nodes_edges dictionary.
        """
        self._nodes = self.graph.all_individuals_set()
        # self._current_node = next(iter(self._nodes))
        self._current_node = self._lpi.pop()
        self._sampled_nodes_edges = dict()
        self._sampled_nodes_edges[self._current_node] = set()

    def _next_node(self):
        """
            Doing a single LP centralized random walk step.
        """
        score = random.uniform(0, 1)
        if score < self.jump_prob:
            neighbor = Neighbor(None, random.choice(list(self._nodes)))
        else:
            neighbor = self.get_random_neighbor(self._current_node)
            if neighbor is None:
                if self._lpi:
                    neighbor = Neighbor(None, self._lpi.pop())
                else:
                    neighbor = Neighbor(None, random.choice(self._lpi_backup))
            else:
                if not any(n.edge_type == neighbor.edge_type and n.node == neighbor.node for n in self._sampled_nodes_edges[self._current_node]):
                    self._sampled_nodes_edges[self._current_node].add(neighbor)

        if neighbor.node not in self._sampled_nodes_edges.keys():
            self._sampled_nodes_edges[neighbor.node] = set()
        self._current_node = neighbor.node

    def sample(self, nodes_number: int, lp_path, data_properties_percentage=1.0, jump_prob: float = 0.1) -> KnowledgeBase:
        """
            Performs the sampling of the graph.

            :param lp_path: Path of the learning problem json file
            :param data_properties_percentage: Percentage of data properties inclusion for each node( values from 0-1 )
            :param nodes_number: Number of distinct nodes to be sampled.
            :param jump_prob: the probability to perform a jump
            :return: Sampled graph.
        """
        self._lpi = list(self.get_lp_individuals(lp_path))
        self._lpi_backup = list(self.get_lp_individuals(lp_path))
        self._create_initial_node_set()
        self.jump_prob = jump_prob
        if nodes_number > len(self._nodes):
            raise ValueError("The number of nodes is too large. Please make sure it "
                             "is smaller than the total number of nodes (total nodes: {})".format(len(self._nodes)))
        if data_properties_percentage > 1 or data_properties_percentage < 0:
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
                self._lpi_backup = list(self._nodes)  # stop restricting RW only to LP nodes
        new_graph = self.get_subgraph_by_remove(self._sampled_nodes_edges, data_properties_percentage)
        return new_graph

    def get_removed_nodes(self):
        return set(self._nodes) - set(self._sampled_nodes_edges.keys())

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
