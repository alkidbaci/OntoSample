import random
import logging
import numpy as np
from typing import List
from collections import deque
from owlapy.owl_individual import OWLNamedIndividual
from ontosample._base import Sampler, Neighbor, Node
try:
    from ontolearn.knowledge_base import KnowledgeBase
except ModuleNotFoundError:
    from ontolearn_light.knowledge_base import KnowledgeBase


logger = logging.getLogger(__name__)


class RandomNodeSampler(Sampler):
    """
        Random Node Sampler samples x number of nodes from the knowledge base and returns
        the sampled graph.
    """

    def _get_random_nodes_from_list(self, nodes_number: int, nodes: List[OWLNamedIndividual]):
        while len(self._sampled_nodes_edges.keys()) < nodes_number and len(nodes) > 0:
            next_node = random.choice(nodes)
            if next_node not in self._sampled_nodes_edges.keys():
                self._sampled_nodes_edges[next_node] = set()
            nodes.remove(next_node)

    def sample(self, nodes_number: int, data_properties_percentage=1.0) -> KnowledgeBase:
        """
        Perform the sampling of the knowledge base.

        Args:
            data_properties_percentage: Percentage of data properties inclusion for each node
                (represented in values from 0-1).
            nodes_number: Number of distinct nodes to be sampled.

        Returns:
            Sampled graph.
        """

        self.check_input(nodes_number, data_properties_percentage)
        self._get_random_nodes_from_list(nodes_number, self._nodes.copy())

        return self.get_subgraph_by_remove(data_properties_percentage, True)


class RandomEdgeSampler(Sampler):
    """
        Random edge sampling. Creates a subgraph by sampling 'x' amount of nodes using
        random selected edges in the graph.
    """

    def _process_edge(self, node1: OWLNamedIndividual, neighbor: Neighbor):
        if neighbor is None:
            self._next_edge()
            return
        else:
            if node1 not in self._sampled_nodes_edges.keys():
                self._sampled_nodes_edges[node1] = set()
            if neighbor.node not in self._sampled_nodes_edges.keys():
                self._sampled_nodes_edges[neighbor.node] = set()
            if not any(n.edge_type == neighbor.edge_type and n.node == neighbor.node for n in
                       self._sampled_nodes_edges[node1]):
                self._sampled_nodes_edges[node1].add(neighbor)

    def _next_edge(self):
        """
        Adding a single randomly selected edge in the sample set.
        """
        node1 = random.choice(list(self._nodes))
        neighbor = self.get_random_neighbor(node1)
        self._process_edge(node1, neighbor)

    def sample(self, nodes_number: int, data_properties_percentage=1.0) -> KnowledgeBase:
        """
        Perform the sampling of the knowledge base.

        Args:
            data_properties_percentage: Percentage of data properties inclusion for each node
                (represented in values from 0-1).
            nodes_number: Number of distinct nodes to be sampled.

        Returns:
            Sampled graph.
        """
        self.check_input(nodes_number, data_properties_percentage)
        try:
            while len(self._sampled_nodes_edges) < nodes_number:
                self._next_edge()
        except RecursionError:
            print("WARN  RandomEdgeSampler    :: No new edges found")
            print(f'INFO  RandomEdgeSampler    :: Total number of nodes connected by edges: '
                  f'{len(self._sampled_nodes_edges)}')
            print(f"INFO  RandomEdgeSampler    :: Defaulting to RandomNodeSampler to fill the rest of the required "
                  f"nodes: {nodes_number-len(self._sampled_nodes_edges)} ")
            sampler = RandomNodeSampler(self.graph)
            sampler._sampled_nodes_edges = self._sampled_nodes_edges
            sampler.sample(nodes_number, data_properties_percentage)
            self._sampled_nodes_edges = sampler._sampled_nodes_edges

        return self.get_subgraph_by_remove(data_properties_percentage)


class RandomWalkSampler(Sampler):
    """
        Random walk sampling. Creates a subgraph by walking randomly in the graph. More likely to end up in a
        never ending loop. Use RWJ if your graph is disconnected.
    """

    def _create_initial_node_set(self):
        """
            Choosing an initial node and add it to the _sampled_nodes_edges dictionary.
        """
        self._current_node = next(iter(self._nodes))
        self._sampled_nodes_edges[self._current_node] = set()

    def _process_neighbor(self, neighbor) -> Neighbor:

        if neighbor is None:
            # if no neighbor found, select a random node
            neighbor = Neighbor(None, random.choice(list(self._nodes)))
        else:
            if not any(n.edge_type == neighbor.edge_type and n.node == neighbor.node for n in
                       self._sampled_nodes_edges[self._current_node]):
                self._sampled_nodes_edges[self._current_node].add(neighbor)
        return neighbor

    def _state_update(self, neighbor):
        if neighbor.node not in self._sampled_nodes_edges.keys():
            self._sampled_nodes_edges[neighbor.node] = set()
        self._current_node = neighbor.node

    def _next_node(self):
        """
            Doing a single random walk step.
        """
        next_node = self._process_neighbor(self.get_random_neighbor(self._current_node))
        self._state_update(next_node)

    def sample(self, nodes_number: int, data_properties_percentage=1.0) -> KnowledgeBase:
        """
        Perform the sampling of the knowledge base.

        Args:
            data_properties_percentage: Percentage of data properties inclusion for each node
                (represented in values from 0-1).
            nodes_number: Number of distinct nodes to be sampled.

        Returns:
            Sampled graph.
        """
        self._create_initial_node_set()
        self.check_input(nodes_number, data_properties_percentage)

        while len(self._sampled_nodes_edges.keys()) < nodes_number:
            self._next_node()
        new_graph = self.get_subgraph_by_remove(data_properties_percentage)
        return new_graph


class RandomWalkerJumpsSampler(RandomWalkSampler):
    """
        Random walker jumps sampling. In comparison with RandomWalkSampler here is introduced the
        "jump" probability which we make use on the _next_node method, meaning that for a certain probability the
        walker will randomly transfer to a node, or it will just take a neighbor of the current node.
        This sampler is useful to avoid getting stuck in any loops on the graph.
    """

    def __init__(self, graph: KnowledgeBase):
        super().__init__(graph)
        self.jump_prob = None

    def _next_node(self):
        """
        Doing a single random walk step.
        Depending on the jump probability here is decided if a jump will be performed or continue to a neighbor node.
        """
        score = random.uniform(0, 1)
        if score < self.jump_prob:
            next_node = Neighbor(None, random.choice(list(self._nodes)))
        else:
            next_node = self._process_neighbor(self.get_random_neighbor(self._current_node))
        self._state_update(next_node)

    def sample(self, nodes_number: int, data_properties_percentage=1.0, jump_prob: float = 0.1) -> KnowledgeBase:
        """
        Perform the sampling of the knowledge base.

        Args:
            data_properties_percentage: Percentage of data properties inclusion for each node
                (represented in values from 0-1).
            nodes_number: Number of distinct nodes to be sampled.
            jump_prob: The probability to perform a random jump on every step.

        Returns:
            Sampled graph.
        """
        self.jump_prob = jump_prob
        self._create_initial_node_set()
        self.check_input(nodes_number, data_properties_percentage)

        while len(self._sampled_nodes_edges.keys()) < nodes_number:
            self._next_node()

        new_graph = self.get_subgraph_by_remove(data_properties_percentage)
        return new_graph


class RandomWalkerWithPrioritizationSampler(RandomWalkSampler):
    """
        Random walker with prioritization sampling. Like random walk sampler, but it prioritizes the nodes based on
        their page rank value. The higher the page rank, the more chance for it to be selected. More likely to end up in
        a never ending loop. Use RWJP if your graph is disconnected.
    """

    def _nodes_list_set_up(self):
        """
            Creates a dictionary that is used to save each individual(node) in the graph by the new structure "Node"
            which will be used to calculate page rank for each of them.
        """
        self._nodes_dict = dict()
        object_properties = list(self._ontology.object_properties_in_signature())
        reasoner = self._reasoner
        for i in self._nodes:
            iri = i.str
            self._nodes_dict[iri] = Node(iri)
            for op in object_properties:
                op_of_i = reasoner.object_property_values(i, op)
                if op_of_i is not None:
                    for obj3ct in op_of_i:
                        self._nodes_dict[iri].outgoing.append(obj3ct)

        for node in self._nodes_dict.values():
            for child in node.outgoing:
                self._nodes_dict[child.str].incoming.append(self._nodes_dict[node.iri])

    def _distribute_pagerank(self, d):
        """
            Single iteration over all the nodes updating their page rank.

            Args:
                d: dumping factor, tha weakly connects all the nodes in the graph
        """
        for node in self._nodes_dict.values():
            node.update_pagerank(d, len(self._nodes))

    def _set_pagerank(self, d=0.15, iteration=100):
        """
            Iterate x number of times over the graph to update the pagerank of each node to a certain degree.

            Args:
                d: dumping factor, tha weakly connects all the nodes in the graph
                iteration: number of iterations that the page rank will update in the graph.
        """
        self._nodes_list_set_up()
        for i in range(iteration):
            self._distribute_pagerank(d)

    def _next_node(self):
        """
        Doing a single prioritized/random step.
        """
        next_node = self._process_neighbor(self.get_prioritized_neighbor(self._current_node, self._nodes_dict))
        self._state_update(next_node)

    def sample(self, nodes_number: int, data_properties_percentage=1.0) -> KnowledgeBase:
        """
        Perform the sampling of the knowledge base.

        Args:
            data_properties_percentage: Percentage of data properties inclusion for each node
                (represented in values from 0-1).
            nodes_number: Number of distinct nodes to be sampled.

        Returns:
            Sampled graph.
        """
        self._set_pagerank()
        logger.info("Finished setting the PageRanks.")
        self._create_initial_node_set()
        self.check_input(nodes_number, data_properties_percentage)

        while len(self._sampled_nodes_edges.keys()) < nodes_number:
            self._next_node()

        return self.get_subgraph_by_remove(data_properties_percentage)


class RandomWalkerJumpsWithPrioritizationSampler(RandomWalkerWithPrioritizationSampler):
    """
        Random walker jumps with prioritization sampling. Like random walker jumps but with prioritization.
    """

    def __init__(self, graph: KnowledgeBase):
        super().__init__(graph)
        self.jump_prob = None

    def _next_node(self):
        """
        Doing a single prioritized/random step.
        Depending on the jump probability here is decided if a jump will be performed or continue to a neighbor node.
        """
        score = random.uniform(0, 1)
        if score < self.jump_prob:
            next_node = Neighbor(None, random.choice(list(self._nodes)))
        else:
            next_node = self._process_neighbor(self.get_prioritized_neighbor(self._current_node, self._nodes_dict))
        self._state_update(next_node)

    def sample(self, nodes_number: int, data_properties_percentage=1.0, jump_prob: float = 0.1) -> KnowledgeBase:
        """
        Perform the sampling of the knowledge base.

        Args:
            data_properties_percentage: Percentage of data properties inclusion for each node
                (represented in values from 0-1).
            nodes_number: Number of distinct nodes to be sampled.
            jump_prob: The probability to perform a random jump on every step.
        Returns:
            Sampled graph.
        """

        self.jump_prob = jump_prob
        self._set_pagerank()
        logger.info("Finished setting the PageRanks.")
        self._create_initial_node_set()
        self.check_input(nodes_number, data_properties_percentage)

        while len(self._sampled_nodes_edges.keys()) < nodes_number:
            self._next_node()
        new_graph = self.get_subgraph_by_remove(data_properties_percentage)
        return new_graph


class ForestFireSampler(Sampler):
    """
        Forest fire sampling. Creates a subgraph by "burning" nodes in the graph with a certain probability.
    """

    def __init__(self, graph: KnowledgeBase, p=0.4, max_visited_nodes_backlog=100, restart_hop_size=10):
        """
        Args:
            graph: Knowledge base to sample.
            p: burning probability.
            max_visited_nodes_backlog: size of the backlog that store visited nodes (unburned nodes).
            restart_hop_size: The size of the queue to process after the previous queue was exhausted.
        """
        super().__init__(graph)
        self._nodes_number = None
        self.p = p
        self.max_visited_nodes_backlog = max_visited_nodes_backlog
        self.restart_hop_size = restart_hop_size

    def _create_initial_node_set(self):
        """
            Create initial structures.
        """
        self._visited_nodes = deque(maxlen=self.max_visited_nodes_backlog)

    def _burn(self, node_queue):
        """Let it burn"""
        while len(self._sampled_nodes_edges) < self._nodes_number:
            # if no node_queue is empty go and get some of the previously visited node to process them.
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
            burned_neighbors = random.sample(list(unvisited_neighbors), size)
            # self._sampled_nodes_edges[top_node] = set(n for n in neighbors if n.node in unvisited_neighbors)
            visited_neighbors = unvisited_neighbors - set(burned_neighbors)
            self._visited_nodes.extendleft(n for n in visited_neighbors if n not in self._visited_nodes)
            node_queue.extend(burned_neighbors)

    def _start_a_fire(self):

        """
            Starts a new "fire" to "burn" more nodes. node_queue is used to store the nodes that
            will be processed in a single call of this method. "Procession nodes" here, means to burn an amount of these
            nodes and the burned node will be added to the sample set.
        """

        remaining_nodes = list(self._get_removed_nodes())
        ignition_node = random.choice(remaining_nodes)
        self._sampled_nodes_edges[ignition_node] = set()
        self._burn(deque([ignition_node]))

    def sample(self, nodes_number: int, data_properties_percentage=1.0) -> KnowledgeBase:
        """
        Perform the sampling of the knowledge base.

        Args:
            data_properties_percentage: Percentage of data properties inclusion for each node
                (represented in values from 0-1).
            nodes_number: Number of distinct nodes to be sampled.

        Returns:
            Sampled graph.
        """

        self._create_initial_node_set()
        self._nodes_number = nodes_number
        self.check_input(nodes_number, data_properties_percentage)

        while len(self._sampled_nodes_edges.keys()) < nodes_number:
            self._start_a_fire()
        new_graph = self.get_subgraph_by_remove(data_properties_percentage)
        return new_graph


