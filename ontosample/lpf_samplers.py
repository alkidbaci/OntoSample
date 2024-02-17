import random
import logging
from typing import List, Iterable
from owlapy.model import OWLNamedIndividual
from ontosample._base import Neighbor
try:
    from ontolearn.knowledge_base import KnowledgeBase
except ModuleNotFoundError:
    from ontolearn_light.knowledge_base import KnowledgeBase
from ontosample.classic_samplers import RandomWalkerWithPrioritizationSampler, RandomNodeSampler, \
    RandomEdgeSampler, RandomWalkSampler

logger = logging.getLogger(__name__)


class RandomNodeSamplerLPFirst(RandomNodeSampler):
    """Random node sampler LP First. Samples 'x' number of nodes from the knowledge base and returns
        the sampled graph. LP nodes goes first."""
    def __init__(self, graph: KnowledgeBase, lp_nodes: Iterable[OWLNamedIndividual]):
        super().__init__(graph)
        self._lpi = lp_nodes

    def sample(self, nodes_number: int, data_properties_percentage=1.0) -> KnowledgeBase:

        self.check_input(nodes_number, data_properties_percentage)
        nodes = self._nodes.copy()
        for node in self._lpi:
            if len(self._sampled_nodes_edges.keys()) < nodes_number:
                self._sampled_nodes_edges[node] = set()
                nodes.remove(node)
        self._get_random_nodes_from_list(nodes_number, nodes)

        return self.get_subgraph_by_remove(data_properties_percentage, True)


class RandomEdgeSamplerLPFirst(RandomEdgeSampler):
    """
        Random Edge Sampler LP First. Creates a subgraph by sampling 'x' amount of nodes in the
        graph using random edges. Starts with LP nodes first.
    """

    def __init__(self, graph: KnowledgeBase, lp_nodes: Iterable[OWLNamedIndividual]):
        super().__init__(graph)
        self._lpi = list(lp_nodes)

    def _next_edge(self):
        """
        Retrieving a single edge randomly.
        """
        if self._lpi:
            node1 = self._lpi.pop()
        else:
            node1 = random.choice(list(self._nodes))
        node2 = self.get_random_neighbor(node1)
        self._process_edge(node1, node2)


class RandomWalkSamplerLPFirst(RandomWalkSampler):
    """
        Random walk sampler LP first. Creates a subgraph by walking randomly in the graph like RandomWalkSampler, but it
        starts with LP nodes first.
    """

    def __init__(self, graph: KnowledgeBase, lp_nodes: Iterable[OWLNamedIndividual]):
        super().__init__(graph)
        self._lpi = list(lp_nodes)

    def _create_initial_node_set(self):
        """
            Choosing an initial node from LP nodes and add it to the _sampled_nodes_edges dictionary.
        """
        self._current_node = self._lpi.pop()
        self._sampled_nodes_edges[self._current_node] = set()

    def _process_neighbor(self, neighbor) -> Neighbor:
        if neighbor is None:
            if self._lpi:
                neighbor = Neighbor(None, self._lpi.pop())
            else:
                neighbor = Neighbor(None, random.choice(self._nodes))
        else:
            if not any(n.edge_type == neighbor.edge_type and n.node == neighbor.node for n in
                       self._sampled_nodes_edges[self._current_node]):
                self._sampled_nodes_edges[self._current_node].add(neighbor)
        return neighbor


class RandomWalkerJumpsSamplerLPFirst(RandomWalkSamplerLPFirst):
    """
        Random walker with jumps sampler LP First. Like RandomWalkerJumpsSampler but it starts with LP nodes first.
    """

    def __init__(self, graph: KnowledgeBase, lp_nodes: Iterable[OWLNamedIndividual]):
        super().__init__(graph, lp_nodes)
        self.jump_prob = None

    def _get_new_neighbor(self):
        if self._lpi:
            neighbor = Neighbor(None, self._lpi.pop())
        else:
            neighbor = Neighbor(None, random.choice(self._nodes))
        return neighbor

    def _next_node(self):
        score = random.uniform(0, 1)
        if score < self.jump_prob:
            next_node = self._get_new_neighbor()
        else:
            next_node = self._process_neighbor(self.get_random_neighbor(self._current_node))

        self._state_update(next_node)

    def sample(self, nodes_number: int, data_properties_percentage=1.0, jump_prob: float = 0.1) -> KnowledgeBase:
        self.jump_prob = jump_prob
        sample = super().sample(nodes_number, data_properties_percentage)
        return sample


class RandomWalkerWithPrioritizationSamplerLPFirst(RandomWalkSamplerLPFirst, RandomWalkerWithPrioritizationSampler):
    """
        Random walker with prioritization sampling LP first. Like RandomWalkerWithPrioritizationSampler, but it starts
        with LP nodes first.
    """

    def _next_node(self):
        """
        Doing a single prioritized/random step. Starts with LP nodes first.
        """
        next_node = self._process_neighbor(self.get_prioritized_neighbor(self._current_node, self._nodes_dict))
        self._state_update(next_node)

    def sample(self, nodes_number: int, data_properties_percentage=1.0) -> KnowledgeBase:
        sample = RandomWalkerWithPrioritizationSampler.sample(self, nodes_number, data_properties_percentage)
        return sample


class RandomWalkerJumpsWithPrioritizationSamplerLPFirst(RandomWalkerWithPrioritizationSamplerLPFirst,
                                                        RandomWalkerJumpsSamplerLPFirst):
    """
        Random walker with jumps with prioritization sampling LP first. Like RandomWalkerJumpsWithPrioritizationSampler,
        but it starts with LP nodes first.
    """

    def __init__(self, graph: KnowledgeBase, lp_nodes: Iterable[OWLNamedIndividual]):
        super().__init__(graph, lp_nodes)
        self.jump_prob = None

    def _next_node(self):
        """
        Doing a single random walk step.
        Depending on the jump probability here is decided if a jump will be performed or continue to a neighbor node.
        Starts with LP nodes first.
        """
        score = random.uniform(0, 1)
        if score < self.jump_prob:
            next_node = RandomWalkerJumpsSamplerLPFirst._get_new_neighbor(self)
        else:
            next_node = self._process_neighbor(self.get_prioritized_neighbor(self._current_node, self._nodes_dict))

        self._state_update(next_node)

    def sample(self, nodes_number: int, data_properties_percentage=1.0, jump_prob: float = 0.1) -> KnowledgeBase:

        self.jump_prob = jump_prob
        sample = super().sample(nodes_number, data_properties_percentage)
        return sample

# No forest fire LP first. FFLPF and FFLPC are the same thing.

