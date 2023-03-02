import random

from Sampling.Samplers.LPFirst_Samplers.RandomWalkerWithPrioritizationSamplerLPFirst import RandomWalkerWithPrioritizationSamplerLPFirst
from Sampling.Samplers.Sampler import Neighbor
from ontolearn.knowledge_base import KnowledgeBase

import logging

logger = logging.getLogger(__name__)

class RandomWalkerJumpsWithPrioritizationSamplerLPFirst(RandomWalkerWithPrioritizationSamplerLPFirst):
    """
        Implementation of random walker jumps with prioritization sampling LP first. Starts with LP nodes first.
    """

    def __init__(self, graph: KnowledgeBase):
        super().__init__(graph)
        self.jump_prob = None
        self._lpi = None

    def _get_new_neighbor(self):
        if self._lpi:
            neighbor = Neighbor(None, self._lpi.pop())
        else:
            neighbor = Neighbor(None, random.choice(list(self._nodes)))
        return neighbor

    def _next_node(self):
        """
        Doing a single prioritized/random step.
        Depending on the jump probability here is decided if a jump will be performed or continue to a neighbor node.
        Starts with LP nodes first.
        """

        score = random.uniform(0, 1)
        if score < self.jump_prob:
            # used the same variable name "neighbor" here to avoid a 2nd repetition of the lines 49-51
            neighbor = self._get_new_neighbor()
        else:
            neighbor = self.get_prioritized_neighbor(self._current_node, self._nodes_dict)
            if neighbor is None:
                # used the same variable name "neighbor" here to avoid a 2nd repetition of the lines 49-51
                neighbor = self._get_new_neighbor()
            else:
                if not any(n.edge_type == neighbor.edge_type and n.node == neighbor.node for n in
                           self._sampled_nodes_edges[self._current_node]):
                    self._sampled_nodes_edges[self._current_node].add(neighbor)

        if neighbor.node not in self._sampled_nodes_edges.keys():
            self._sampled_nodes_edges[neighbor.node] = set()
        self._current_node = neighbor.node

    def sample(self, nodes_number: int, lp_path, data_properties_percentage=1.0, jump_prob: float = 0.1) -> KnowledgeBase:
        """
        Performs the sampling of the graph.

        :param nodes_number: Number of distinct nodes to be sampled.
        :param lp_path: Path of the .json file containing the learning problem
        :param data_properties_percentage: Percentage of data properties inclusion for each node( values from 0-1 )
        :param jump_prob: the probability to perform a random jump

        :return: Sampled graph.
        """

        self.jump_prob = jump_prob
        self._set_pagerank()
        logger.info("Finished setting the PageRanks.")
        self._lpi = list(self.get_lp_individuals(lp_path))
        self._create_initial_node_set()
        if nodes_number > len(self._nodes):
            raise ValueError("The number of nodes is too large. Please make sure it "
                             "is smaller than the total number of nodes (total nodes: {})".format(len(self._nodes)))
        if data_properties_percentage > 1 or data_properties_percentage < 0:
            raise ValueError("Data properties sample percentage must be a value between 1 and 0")

        while len(self._sampled_nodes_edges.keys()) < nodes_number:
            self._next_node()
        new_graph = self.get_subgraph_by_remove(self._sampled_nodes_edges, data_properties_percentage)
        return new_graph


