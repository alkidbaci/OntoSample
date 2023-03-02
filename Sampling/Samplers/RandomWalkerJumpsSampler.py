import random
from ontolearn.knowledge_base import KnowledgeBase
from Sampling.Samplers.Sampler import Sampler, Neighbor
import logging

logger = logging.getLogger(__name__)


class RandomWalkerJumpsSampler(Sampler):
    """
        Implementation of random walker jumps sampling. In comparison with RandomWalkSampler here is introduced the
        "jump" probability which we make use on the _next_node method, meaning that for a certain probability the
        walker will randomly transfer to a node, or it will just take a neighbor of the current node.
        This sampler is useful to avoid getting stuck in a loop on the graph.
    """

    def __init__(self, graph: KnowledgeBase):
        super().__init__(graph)
        self.jump_prob = None

    def _create_initial_node_set(self):
        """
            Choosing an initial node and add it to the _sampled_nodes_edges dictionary.
        """
        self._nodes = self.graph.all_individuals_set()
        self._current_node = next(iter(self._nodes))
        self._sampled_nodes_edges = dict()
        self._sampled_nodes_edges[self._current_node] = set()

    def _next_node(self):
        """
        Doing a single random walk step.
        Depending on the jump probability here is decided if a jump will be performed or continue to a neighbor node.
        """
        score = random.uniform(0, 1)
        if score < self.jump_prob:
            # used the same variable name "neighbor" here to avoid a 2nd repetition of the lines 49-51
            neighbor = Neighbor(None, random.choice(list(self._nodes)))
        else:
            neighbor = self.get_random_neighbor(self._current_node)
            if neighbor is None:
                # used the same variable name "neighbor" here to avoid a 2nd repetition of the lines 49-51
                neighbor = Neighbor(None, random.choice(list(self._nodes)))
            else:
                if not any(n.edge_type == neighbor.edge_type and n.node == neighbor.node for n in
                           self._sampled_nodes_edges[self._current_node]):
                    self._sampled_nodes_edges[self._current_node].add(neighbor)

        if neighbor.node not in self._sampled_nodes_edges.keys():
            self._sampled_nodes_edges[neighbor.node] = set()
        self._current_node = neighbor.node

    def sample(self, nodes_number: int, data_properties_percentage=1.0, jump_prob: float = 0.1) -> KnowledgeBase:
        """
        Performs the sampling of the graph.

        :param data_properties_percentage: Percentage of data properties inclusion for each node( values from 0-1 )
        :param jump_prob: the probability to perform a jump
        :param nodes_number: Number of distinct nodes to be sampled.
        :return: Sampled graph.
        """
        self.jump_prob = jump_prob
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

    def get_removed_nodes(self):
        return set(self._nodes) - set(self._sampled_nodes_edges.keys())
