import json
import random
from collections import deque

import numpy as np

from Sampling.Samplers.Sampler import Sampler, Neighbor
from ontolearn.knowledge_base import KnowledgeBase
import logging

logger = logging.getLogger(__name__)


class ForestFireSamplerLPFirst(Sampler):
    """
        Implementation of forest fire sampling. Creates a subgraph by "burning" nodes in the graph with a certain
        probability. Start the burning with LP nodes.
    """

    def __init__(self, graph: KnowledgeBase, p=0.4, max_visited_nodes_backlog=100, restart_hop_size=10):
        super().__init__(graph)
        self._lpi = None
        self._nodes_number = None
        self.p = p
        self.max_visited_nodes_backlog = max_visited_nodes_backlog
        self.restart_hop_size = restart_hop_size

    def _create_initial_node_set(self):
        """
            Create initial structures.
        """
        self._nodes = self.graph.all_individuals_set()
        self._sampled_nodes_edges = dict()
        self._visited_nodes = deque(maxlen=self.max_visited_nodes_backlog)

    def _start_a_fire(self, first_iter):
        """
            Starts a new "fire" to "burn" more nodes. Basically, node_queue is used to store the nodes that
            will be processed in a single call of this method. "Procession nodes" here, means to burn an amount of these
            nodes and the burned node will be added to the sample set. Since this is an LPFirst modification of FF it
            starts by burning LP nodes first.
        """
        if not first_iter:
            remaining_nodes = list(self.get_removed_nodes())
            ignition_node = random.choice(remaining_nodes)
            self._sampled_nodes_edges[ignition_node] = set()
            node_queue = deque([ignition_node])
        else:
            node_queue = deque()
            node_queue.extendleft(self._lpi)
        while len(self._sampled_nodes_edges) < self._nodes_number:
            if len(node_queue) == 0:
                node_queue = deque([self._visited_nodes.popleft()
                                    for i in range(min(self.restart_hop_size, len(self._visited_nodes)))
                                    ])
                if len(node_queue) == 0:
                    # no more nodes to burn
                    break
            top_node = node_queue.popleft()
            if top_node not in self._sampled_nodes_edges.keys():
                self._sampled_nodes_edges[top_node] = list()
            neighbors = set(self.get_neighbors(top_node))
            self._sampled_nodes_edges[top_node] = neighbors
            unvisited_neighbors = set(n.node for n in neighbors if n.node not in self._sampled_nodes_edges.keys() and n.node not in node_queue)
            score = np.random.geometric(self.p)
            size = min(len(unvisited_neighbors), score)
            burned_neighbors = random.sample(unvisited_neighbors, size)
            # self._sampled_nodes_edges[top_node] = set(n for n in neighbors if n.node in unvisited_neighbors)
            visited_neighbors = unvisited_neighbors - set(burned_neighbors)
            self._visited_nodes.extendleft(n for n in visited_neighbors if n not in self._visited_nodes)
            node_queue.extend(burned_neighbors)

    def sample(self, nodes_number: int, lp_path, data_properties_percentage=1.0 ) -> KnowledgeBase:
        """
            Performs the sampling of the graph.

            :param lp_path: Path of the learning problem json file
            :param data_properties_percentage: Percentage of data properties inclusion for each node( values from 0-1 )
            :param nodes_number: Number of distinct nodes to be sampled.
            :return: Sampled graph.
        """

        self._create_initial_node_set()
        self._nodes_number = nodes_number
        self._lpi = set(self.get_lp_individuals(lp_path))
        if nodes_number > len(self._nodes):
            raise ValueError("The number of nodes is too large. Please make sure it "
                             "is smaller than the total number of nodes (total nodes: {})".format(len(self._nodes)))
        if data_properties_percentage > 1 or data_properties_percentage < 0:
            raise ValueError("Data properties sample percentage must be a value between 1 and 0")

        first_iter = True
        while len(self._sampled_nodes_edges.keys()) < nodes_number:
            self._start_a_fire(first_iter)
            first_iter = False
        new_graph = self.get_subgraph_by_remove(self._sampled_nodes_edges, data_properties_percentage)
        return new_graph

    def get_removed_nodes(self):
        return set(self._nodes) - set(self._sampled_nodes_edges.keys())

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
