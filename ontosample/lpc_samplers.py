import random
import logging
from typing import Iterable
from owlapy.model import OWLNamedIndividual
from collections import deque
from ontosample._base import Neighbor
from ontolearn.knowledge_base import KnowledgeBase
from ontosample.classic_samplers import RandomWalkerWithPrioritizationSampler, ForestFireSampler,\
    RandomWalkSampler, RandomEdgeSampler, RandomNodeSampler


logger = logging.getLogger(__name__)


class RandomNodeSamplerLPCentralized(RandomNodeSampler):
    """
        Random node sampler LP Centralized. Samples 'x' number of nodes from the knowledge base and returns
        the sampled graph. Focuses around LP nodes.
    """
    def __init__(self, graph: KnowledgeBase, lp_nodes: Iterable[OWLNamedIndividual]):
        """
         Args:
             graph: Knowledge base to sample.
             lp_nodes: Learning problem nodes, positive and negative or whatever you want the graph to be centralized
             around.
        """
        super().__init__(graph)
        assert lp_nodes, "List of learning problems should contain at least 1 element"
        self._lpi = lp_nodes

    def sample(self, nodes_number: int, data_properties_percentage=1.0) -> KnowledgeBase:

        one_hop_neighbor_nodes = set(self.get_neighborhood_from_nodes(self._lpi))
        two_hop_neighbor_nodes = set(self.get_neighborhood_from_nodes(one_hop_neighbor_nodes))
        two_hop_neighbor_nodes = two_hop_neighbor_nodes.union(one_hop_neighbor_nodes)
        other_nodes = set(self._nodes) - two_hop_neighbor_nodes

        for node in self._lpi:
            if len(self._sampled_nodes_edges.keys()) < nodes_number:
                self._sampled_nodes_edges[node] = set()

        self._get_random_nodes_from_list(nodes_number, list(one_hop_neighbor_nodes))
        self._get_random_nodes_from_list(nodes_number, list(two_hop_neighbor_nodes))
        self._get_random_nodes_from_list(nodes_number, list(other_nodes))

        return self.get_subgraph_by_remove(data_properties_percentage, True)


class RandomEdgeSamplerLPCentralized(RandomEdgeSampler):
    """
        Random edge sampling LP Centralized. Creates a subgraph by sampling 'x' amount of nodes in the
        graph using random edges. Starts with LP nodes first and continues to take edges from the LP neighborhood.
    """

    def __init__(self, graph: KnowledgeBase, lp_nodes: Iterable[OWLNamedIndividual]):
        """
         Args:
             graph: Knowledge base to sample.
             lp_nodes: Learning problem nodes, positive and negative or whatever you want the graph to be centralized
             around.
        """
        super().__init__(graph)
        assert lp_nodes, "List of learning problems should contain at least 1 element"
        self._lpi = list(lp_nodes)
        self._exploration_limit = list(lp_nodes)

    def _next_edge(self):
        """
            Doing a single LP centralized random edge step.
        """
        if self._lpi:
            node1 = self._lpi.pop()
        else:
            node1 = random.choice(self._exploration_limit)
        neighbor = self.get_random_neighbor(node1)
        self._process_edge(node1, neighbor)

    def sample(self, nodes_number: int, data_properties_percentage=1.0) -> KnowledgeBase:

        self.check_input(nodes_number, data_properties_percentage)

        one_hop_neighbor_nodes = self.get_neighborhood_from_nodes(self._lpi)
        two_hop_neighbor_nodes = self.get_neighborhood_from_nodes(one_hop_neighbor_nodes)
        two_hop_neighbor_nodes.extend(one_hop_neighbor_nodes)

        stop_centralized_search_threshold = len(self._nodes) * 0.05
        no_new_nodes_counter = 0
        is_init_limit = True
        while len(self._sampled_nodes_edges.keys()) < nodes_number:
            current_sampled_node_amount = len(self._sampled_nodes_edges.keys())
            self._next_edge()
            post_sampled_node_amount = len(self._sampled_nodes_edges.keys())  # amount of nodes after a single step
            if current_sampled_node_amount == post_sampled_node_amount:
                no_new_nodes_counter += 1
            else:
                no_new_nodes_counter = 0
            if no_new_nodes_counter > stop_centralized_search_threshold:
                no_new_nodes_counter = 0
                if is_init_limit:
                    self._exploration_limit = one_hop_neighbor_nodes
                    is_init_limit = False
                elif self._exploration_limit == one_hop_neighbor_nodes:
                    self._exploration_limit = two_hop_neighbor_nodes
                elif self._exploration_limit == two_hop_neighbor_nodes:
                    self._exploration_limit = self._nodes
                else:
                    # in case there are not enough edges in the graph, fill the remaining sample set with random nodes
                    unexplored_nodes = self.get_removed_nodes()
                    nodes_left_to_sample = nodes_number - len(self._sampled_nodes_edges.keys())
                    filler_nodes = random.sample(unexplored_nodes, nodes_left_to_sample)
                    for fn in filler_nodes:
                        self._sampled_nodes_edges[fn] = set()
                    break

        return self.get_subgraph_by_remove(data_properties_percentage)


class RandomWalkSamplerLPCentralized(RandomWalkSampler):
    """
        Random walker sampler learning problem(LP) centralized. Creates a subgraph by walking randomly
        in the neighborhood of the LP nodes in the graph.
    """

    def __init__(self, graph: KnowledgeBase, lp_nodes: Iterable[OWLNamedIndividual]):
        """
         Args:
             graph: Knowledge base to sample.
             lp_nodes: Learning problem nodes, positive and negative or whatever you want the graph to be centralized
             around.
        """
        super().__init__(graph)
        assert lp_nodes, "List of learning problems should contain at least 1 element"
        self._exploration_limit = list(lp_nodes)
        self._lpi = list(lp_nodes)

    def _create_initial_node_set(self):
        # self._current_node = next(iter(self._nodes))
        self._current_node = self._lpi.pop()
        self._sampled_nodes_edges[self._current_node] = set()

    def _process_neighbor(self, neighbor) -> Neighbor:

        if neighbor is None:
            if self._lpi:
                neighbor = Neighbor(None, self._lpi.pop())
            else:
                neighbor = Neighbor(None, random.choice(self._exploration_limit))
        else:
            if not any(n.edge_type == neighbor.edge_type and n.node == neighbor.node for n in
                       self._sampled_nodes_edges[self._current_node]):
                self._sampled_nodes_edges[self._current_node].add(neighbor)
        return neighbor

    def _start_sampling(self, nodes_number):
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
                self._exploration_limit = list(self._nodes)  # stop restricting RW only to LP nodes

    def sample(self, nodes_number: int, data_properties_percentage=1.0) -> KnowledgeBase:

        self._create_initial_node_set()
        self.check_input(nodes_number, data_properties_percentage)
        self._start_sampling(nodes_number)

        return self.get_subgraph_by_remove(data_properties_percentage)


class RandomWalkerJumpsSamplerLPCentralized(RandomWalkSamplerLPCentralized):
    """
        Random walker jumps sampling learning problem(LP) centralized. Creates a subgraph by walking
        randomly in neighborhood of the LP nodes in the graph. Jump probability helps to escape loops
        (infinite execution).
    """

    def __init__(self, graph: KnowledgeBase, lp_nodes: Iterable[OWLNamedIndividual]):
        """
         Args:
             graph: Knowledge base to sample.
             lp_nodes: Learning problem nodes, positive and negative or whatever you want the graph to be centralized
             around.
        """
        super().__init__(graph, lp_nodes)
        self.jump_prob = None

    def _next_node(self):
        """
            Doing a single LP centralized random walk step.
        """
        score = random.uniform(0, 1)
        if score < self.jump_prob:
            next_node = Neighbor(None, random.choice(list(self._exploration_limit)))
        else:
            next_node = self._process_neighbor(self.get_random_neighbor(self._current_node))
        self._state_update(next_node)

    def sample(self, nodes_number: int, data_properties_percentage=1.0, jump_prob: float = 0.1) -> KnowledgeBase:

        self._create_initial_node_set()
        self.jump_prob = jump_prob
        self.check_input(nodes_number, data_properties_percentage)
        self._start_sampling(nodes_number)

        return self.get_subgraph_by_remove(data_properties_percentage)


class RandomWalkerWithPrioritizationSamplerLPCentralized(RandomWalkSamplerLPCentralized,
                                                         RandomWalkerWithPrioritizationSampler):
    """
        Random walker with prioritization sampling LP centralized.
    """

    def _next_node(self):
        """
        Doing a single centralized prioritized/random step.
        """
        next_node = self._process_neighbor(self.get_prioritized_neighbor(self._current_node, self._nodes_dict))
        self._state_update(next_node)

    def sample(self, nodes_number: int, data_properties_percentage=1.0) -> KnowledgeBase:
        self._set_pagerank()
        logger.info("Finished setting the PageRanks.")
        self._create_initial_node_set()
        self.check_input(nodes_number, data_properties_percentage)
        self._start_sampling(nodes_number)

        return self.get_subgraph_by_remove(data_properties_percentage)


class RandomWalkerJumpsWithPrioritizationSamplerLPCentralized(RandomWalkerWithPrioritizationSamplerLPCentralized):
    """
        Implementation of random walker jumps with prioritization sampling LP centralized.
    """

    def __init__(self, graph: KnowledgeBase, lp_nodes: Iterable[OWLNamedIndividual]):
        super().__init__(graph, lp_nodes)
        self.jump_prob = None

    def _create_initial_node_set(self):
        self._current_node = self._lpi.pop()
        self._sampled_nodes_edges = dict()
        self._sampled_nodes_edges[self._current_node] = set()

    def _next_node(self):
        """
        Doing a single centralized prioritized/random step.
        """
        score = random.uniform(0, 1)
        if score < self.jump_prob:
            next_node = Neighbor(None, random.choice(list(self._exploration_limit)))
        else:
            next_node = self._process_neighbor(self.get_prioritized_neighbor(self._current_node, self._nodes_dict))
        self._state_update(next_node)

    def sample(self, nodes_number: int, data_properties_percentage=1.0, jump_prob: float = 0.1) -> KnowledgeBase:
        self.jump_prob = jump_prob
        sample = super().sample(nodes_number, data_properties_percentage)
        return sample


class ForestFireSamplerLPCentralized(ForestFireSampler):
    """
        Forest fire sampler LP centralized. Creates a subgraph by "burning" nodes in the graph with a certain
        probability. Start the burning with LP nodes and focus around them.
    """

    def __init__(self, graph: KnowledgeBase, lp_nodes: Iterable[OWLNamedIndividual], p=0.4, max_visited_nodes_backlog=100,
                 restart_hop_size=10):
        """
         Args:
             graph: Knowledge base to sample.
             lp_nodes: Learning problem nodes, positive and negative or whatever you want the graph to be centralized
             around.
             p: burning probability.
             max_visited_nodes_backlog: size of the backlog that store visited nodes (unburned nodes).
             restart_hop_size: The size of the queue to process after the previous queue was exhausted.
        """
        super().__init__(graph, p, max_visited_nodes_backlog, restart_hop_size)
        self._first_iter = True
        self._lpi = list(lp_nodes)

    def _start_a_fire(self):
        if not self._first_iter:
            remaining_nodes = list(self.get_removed_nodes())
            ignition_node = random.choice(remaining_nodes)
            self._sampled_nodes_edges[ignition_node] = set()
            node_queue = deque([ignition_node])
        else:
            node_queue = deque()
            node_queue.extendleft(self._lpi)
        self._burn(node_queue)

    def sample(self, nodes_number: int, data_properties_percentage=1.0) -> KnowledgeBase:

        self._create_initial_node_set()
        self._nodes_number = nodes_number
        self.check_input(nodes_number, data_properties_percentage)

        while len(self._sampled_nodes_edges.keys()) < nodes_number:
            self._start_a_fire()
            self._first_iter = False
        new_graph = self.get_subgraph_by_remove(data_properties_percentage)
        return new_graph
