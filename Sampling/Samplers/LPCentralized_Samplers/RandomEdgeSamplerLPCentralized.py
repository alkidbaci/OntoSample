import json
import random

from Sampling.Samplers.Sampler import Sampler
from ontolearn.knowledge_base import KnowledgeBase
import logging

logger = logging.getLogger(__name__)


class RandomEdgeSamplerLPCentralized(Sampler):
    """
        Implementation of random edge sampling LP Centralized. Creates a subgraph by sampling 'x' amount of edges in the
        graph. Starts with LP nodes first and continues to take edges from the LP neighborhood.

        Args: graph (KnowledgeBase)
    """

    def __init__(self, graph: KnowledgeBase):
        super().__init__(graph)
        self._lpi = None
        self._lpi_backup = None
        self._nodes = self.graph.all_individuals_set()
        self._sampled_nodes_edges = dict()
        self._one_hop_neighbors = list()
        self._two_hop_neighbors = list()

    def _next_edge(self):
        """
            Doing a single LP centralized random edge step.
        """
        if self._lpi:
            node1 = self._lpi.pop()
        else:
            node1 = random.choice(list(self._lpi_backup))
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
        Performs the sampling of the graph.

        :param nodes_number: Number of distinct nodes to be sampled.
        :param lp_path: Path of the .json file containing the learning problem
        :param data_properties_percentage: Percentage of data properties inclusion for each node( values from 0-1 )
        :return: Sampled graph.
        """
        total_edges = self.number_of_edges()
        self._lpi_backup = list(self.get_lp_individuals(lp_path))

        if nodes_number > len(self._nodes):
            raise ValueError("The number of nodes is too large. Please make sure it "
                             "is smaller than the total number of nodes (total nodes: {})".format(len(self._nodes)))
        if data_properties_percentage > 1 or data_properties_percentage < 0:
            raise ValueError("Data properties sample percentage must be a value between 1 and 0")
        self._lpi = list(self.get_lp_individuals(lp_path))
        self.set_one_and_two_hop_neighbors()
        stop_centralized_search_threshold = len(self._nodes) * 0.05
        no_new_nodes_counter = 0
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
                if self._lpi_backup == list(self.get_lp_individuals(lp_path)):
                    self._lpi_backup = list(self._one_hop_neighbors)
                elif self._lpi_backup == list(self._one_hop_neighbors):
                    self._lpi_backup = list(self._two_hop_neighbors)
                elif self._lpi_backup == list(self._two_hop_neighbors):
                    self._lpi_backup = list(self._nodes)
                else:
                    # in case there are not enough edges in the graph, fill the remaining sample set with random nodes
                    unexplored_nodes = self.get_removed_nodes()
                    nodes_left_to_sample = nodes_number - len(self._sampled_nodes_edges.keys())
                    filler_nodes = random.sample(unexplored_nodes, nodes_left_to_sample)
                    for fn in filler_nodes:
                        self._sampled_nodes_edges[fn] = set()
                    break
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

    def set_one_and_two_hop_neighbors(self):

        helper = Sampler(self.graph)

        # one-hop neighbors
        for node in self._lpi:
            neighbors = helper.get_neighbors(node)
            if neighbors is not None:
                self._one_hop_neighbors.extend(ngb.node for ngb in neighbors)

        self._two_hop_neighbors.extend(self._one_hop_neighbors)
        # two-hop neighbors
        for neighbor in self._one_hop_neighbors:
            neighbors = helper.get_neighbors(neighbor)
            if neighbors is not None:
                self._two_hop_neighbors.extend(ngb.node for ngb in neighbors)

