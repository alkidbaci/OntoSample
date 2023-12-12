import random
import logging
from typing import Iterable
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.base import OWLReasoner_FastInstanceChecker, OWLReasoner_Owlready2
from owlapy.model import OWLNamedIndividual, OWLObjectPropertyAssertionAxiom, \
    OWLDataPropertyAssertionAxiom, OWLDeclarationAxiom, IRI

logger = logging.getLogger(__name__)


class Neighbor:
    """
        Class structure to save a neighbor node in such a way that it also stores the edge type which make the
        connection.
    """

    def __init__(self, edge_type, node):
        self.edge_type = edge_type
        self.node = node


class Node:
    """
        Class structure to save individuals (nodes) of the graph in a way that we can calculate pagerank for each of
        them.
    """
    def __init__(self, IRI):
        self.IRI = IRI
        self.outgoing = []
        self.incoming = []
        self.pagerank = 1.0

    def update_pagerank(self, d, n):
        """
            Updates the pagerank of a single node

            :param d: dumping factor, tha weakly connects all the nodes in the graph
            :param n: total number of nodes in the graph
        """
        incoming = self.incoming
        pr_sum = sum((node.pagerank / len(node.outgoing)) for node in incoming)
        self.pagerank = d / n + (1 - d) * pr_sum


class Sampler:
    """
       Base class for sampling techniques.

       Args:
           graph (KnowledgeBase): The knowledge base object that you want to sample.
    """

    def __init__(self, graph: KnowledgeBase):
        """
            Init

           Args:
               graph (KnowledgeBase): The knowledge base object that you want to sample.
        """
        self._sampled_nodes_edges = dict()
        self.graph = graph
        self._reasoner = graph.reasoner()
        self._ontology = graph.ontology()
        self._manager = self._ontology.get_owl_ontology_manager()
        self._nodes = list(graph.all_individuals_set())
        self._object_properties = list(self._ontology.object_properties_in_signature())
        self._data_properties = list(self._ontology.data_properties_in_signature())
        self._all_dp_axioms = dict()

    def get_prioritized_neighbor(self, node, nodes_dict):
        """
            Gets the neighbor of the current node which has the highest pagerank.

            :param node: the current node
            :param nodes_dict: The dictionary which has for each OWLNamedIndividual(node) of the graph the corresponding
            representation by "Node" class structure
            :return: neighbor of the current node with the highest pagerank or None if the current node does not have
            any neighbors
        """

        neighbors = self.get_neighbors(node)
        neighbor_pagerank_dict = dict()

        if neighbors:
            for neighbor in neighbors:
                neighbor_pagerank_dict[neighbor] = nodes_dict[neighbor.node.get_iri().as_str()].pagerank
            return random.choices(list(neighbor_pagerank_dict.keys()), weights=list(neighbor_pagerank_dict.values()), k=1).pop()
        else:
            return None

    def get_random_neighbor(self, node: OWLNamedIndividual):
        """
            Gets a random neighbor node of the current node.

            :param node: the current node
            :return: random neighbor of the current node as an object type Neighbor or None if the current node does not
            have any neighbors
        """

        neighbors = self.get_neighbors(node)

        if neighbors:
            neighbor = random.choice(neighbors)
            return neighbor
        else:
            return None

    def get_neighbors(self, node) -> list[Neighbor]:
        neighbors = []
        for op in self._object_properties:
            object_nodes = self._reasoner.object_property_values(node, op)
            if object_nodes:
                for on in object_nodes:
                    neighbors.append(Neighbor(op, on))
        return neighbors

    def get_neighborhood_from_nodes(self, starting_set: Iterable[OWLNamedIndividual]):

        neighborhood = list()
        # one-hop neighbors
        for node in starting_set:
            neighbors = self.get_neighbors(node)
            if neighbors is not None:
                neighborhood.extend(ngb.node for ngb in neighbors)

        return neighborhood

    def get_subgraph_by_remove(self, data_properties_percentage, include_all_edges=False) -> KnowledgeBase:
        """
            Builds and returns the sampled graph based on the sampled_nodes_edges and dpp by removing axioms from
            the original ontology.

            :param include_all_edges Whether to include all edges (object properties) that connect the remaining
            nodes (individuals) in the sample graph
            :param data_properties_percentage: Percentage of data properties inclusion for each node( values from 0-1 )
            edges as values
            :return: Sample of the Graph
        """
        assert len(self._sampled_nodes_edges) > 0, "The current sample is empty"

        for node in self._nodes:
            if node not in self._sampled_nodes_edges:
                self._manager.remove_axiom(self._ontology, OWLDeclarationAxiom(node))
            else:
                if not include_all_edges:
                    # Removing all the edges of "node" except the one selected by the sampler
                    self._remove_unselected_edges(node)
                # Storing every data property for each node
                if data_properties_percentage < 1:
                    self._store_data_properties(node)
        self._remove_unused_data_properties()
        # Removing specific percentage (data_properties_percentage) of data properties for each node
        if data_properties_percentage < 1:
            self._sample_data_properties(data_properties_percentage)

        new_base_reasoner = OWLReasoner_Owlready2(ontology=self._ontology)
        new_reasoner = OWLReasoner_FastInstanceChecker(ontology=self._ontology,
                                                       base_reasoner=new_base_reasoner)
        new_graph = KnowledgeBase(ontology=self._ontology, reasoner=new_reasoner, path=self.graph.path)

        # self._manager.save_ontology(ontology=self._ontology, document_iri=IRI.create('file:/test.owl'))
        return new_graph

    def save_sample(self):
        filename = f'file:/{self.graph.path.split("/")[-1].replace(".owl", "")}_sample_' \
                   f'{len(self._sampled_nodes_edges)}.owl'
        self._manager.save_ontology(ontology=self._ontology, document_iri=IRI.create(filename))

    def get_sampled_nodes(self):
        return self._sampled_nodes_edges.keys()

    def get_removed_nodes(self):
        return set(self._nodes) - set(self._sampled_nodes_edges.keys())

    def check_input(self, nodes_number, data_properties_percentage):
        if nodes_number > len(self._nodes):
            raise ValueError("The number of nodes is too large. Please make sure it "
                             "is smaller than the total number of nodes (total nodes: {})".format(len(self._nodes)))

        if data_properties_percentage > 1 or data_properties_percentage < 0:
            raise ValueError("Data properties sample percentage must be a value between 1 and 0")

    def _remove_unselected_edges(self, node):
        """
            Removing all the edges of "node" except the one selected by the sampler

            :param node: node, edges of which, will be iterated over.
        """
        for op in self._object_properties:
            object_nodes = self._reasoner.object_property_values(node, op)
            if object_nodes is not None:
                for obj3ct in object_nodes:
                    if not any(neighbor.edge_type == op and neighbor.node == obj3ct for neighbor in
                               self._sampled_nodes_edges[node]):
                        self._manager.remove_axiom(self._ontology, OWLObjectPropertyAssertionAxiom(node, op, obj3ct))

    def _store_data_properties(self, node):
        """
            Storing every data property for each node to the _all_dp_axioms dictionary

            :param node: node, data properties of which, will be stored.
        """
        self._all_dp_axioms[node] = list()
        for dp in self._data_properties:
            dp_of_node = self._reasoner.data_property_values(node, dp)
            if dp_of_node is not None:
                for literal in dp_of_node:
                    self._all_dp_axioms[node].append(OWLDataPropertyAssertionAxiom(node, dp, literal))

    def _sample_data_properties(self, dpp):
        """
            Removing specific percentage (dpp) of data properties for each node

            :param dpp: Percentage of data properties inclusion for each node( values from 0-1 )
        """
        for node in self._all_dp_axioms.keys():
            nr_of_dp_axioms_of_node = len(self._all_dp_axioms[node])
            nr_of_removed_axioms = 0
            while self._all_dp_axioms[node] and 1 - dpp > nr_of_removed_axioms / nr_of_dp_axioms_of_node:
                self._manager.remove_axiom(self._ontology, self._all_dp_axioms[node].pop())
                nr_of_removed_axioms += 1

    def _remove_unused_data_properties(self):
        """
            Check for unused data properties and removes them because some concept learners like EvoLearner
            will throw exception if they aren't removed
        """
        for dp in self._data_properties:
            skip = False
            for ind in self._sampled_nodes_edges.keys():
                if list(self._reasoner.data_property_values(ind, dp)):
                    skip = True
                    break
            if skip:
                continue
            else:
                self._manager.remove_axiom(self._ontology, OWLDeclarationAxiom(dp))
