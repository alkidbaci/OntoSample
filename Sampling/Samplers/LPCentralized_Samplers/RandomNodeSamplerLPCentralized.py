import json

from Sampling.Samplers.Sampler import Sampler
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker
from owlapy.model import OWLDeclarationAxiom, IRI
from owlapy.owlready2.temp_classes import OWLReasoner_Owlready2_TempClasses
import logging


logger = logging.getLogger(__name__)

class RandomNodeSamplerLPCentralized:
    """
        Random Node Sampler LP Centralized samples x number of nodes from the knowledge base and returns
        the sampled graph. Focuses around LP nodes.
    """
    def __init__(self, graph: KnowledgeBase):
        self._reasoner = None
        self._manager = None
        self._onto = None
        self._removed_nodes = None
        self._nodes = None
        self._sampled_nodes = None
        self.graph = graph
        self._lpi = None
        self._lpi_backup = None

    def sample(self, number_of_nodes: int, lp_path) -> KnowledgeBase:
        """
        Perform the sampling of the knowledge base

        :param number_of_nodes: Number of nodes that will be sampled
        :param lp_path: Path of the .json file containing the learning problem
        :return: Sampled knowledge base
        """
        if number_of_nodes == 0:
            raise ValueError("Sampling is possible for node number > 0")
        # Getting the OWLOntology object from KB
        self._onto = self.graph.ontology()
        # Getting the OWLOntologyManager object from onto
        self._manager = self._onto.get_owl_ontology_manager()
        # Getting all OWLNamedIndividual objects from KB
        self._nodes = self.graph.all_individuals_set()
        self._sampled_nodes = set()
        self._removed_nodes = set()
        self._lpi = list(self.get_lp_individuals(lp_path))
        self._reasoner = self.graph.reasoner()
        self._nodes_to_remove_len = len(self._nodes) - number_of_nodes
        if self._nodes_to_remove_len < 0:
            raise ValueError("Number of nodes to sample is greater than number of total nodes in the knowledge base")

        helper = Sampler(self.graph)
        # we consider central nodes, the lp nodes and the nodes up to two hop neighbors.

        one_hop_neighbors = list()
        two_hop_neighbors = list()

        # one-hop neighbors
        for node in self._lpi:
            neighbors = helper.get_neighbors(node)
            if neighbors is not None:
                one_hop_neighbors.extend(neighbors)

        two_hop_neighbors.extend(one_hop_neighbors)
        # two-hop neighbors
        for neighbor in one_hop_neighbors:
            neighbors = helper.get_neighbors(neighbor.node)
            if neighbors is not None:
                two_hop_neighbors.extend(neighbors)

        two_hop_nodes_to_ignore = set(neighbor.node for neighbor in two_hop_neighbors)
        one_hop_nodes_to_ignore = set(neighbor.node for neighbor in one_hop_neighbors)

        # since graph.all_individuals_set() returns a frozenset, the iteration is random each time.
        self._remove_nodes(self._nodes, two_hop_nodes_to_ignore)

        # in case the number of sampled nodes is not complete then remove from the two hop neighbor nodes.
        if len(self._removed_nodes) < self._nodes_to_remove_len:
            self._remove_nodes(two_hop_nodes_to_ignore, one_hop_nodes_to_ignore)

        # in case the number of sampled nodes is still not complete then remove from the one hop neighbor nodes
        if len(self._removed_nodes) < self._nodes_to_remove_len:
            self._remove_nodes(one_hop_nodes_to_ignore, None)

        self._sampled_nodes = set(self._nodes) - self._removed_nodes

        logger.info("Individuals removed: {}".format(len(self._removed_nodes)))

        # check for unused data properties, concept learners like EvoLearner will throw exception if they aren't removed
        data_properties = list(self._onto.data_properties_in_signature())
        for dp in data_properties:
            skip = False
            for ind in self._sampled_nodes:
                if list(self._reasoner.data_property_values(ind, dp)):
                    skip = True
                    break
            if skip:
                continue
            else:
                self._manager.remove_axiom(self._onto, OWLDeclarationAxiom(dp))

        # Building new reasoner
        new_base_reasoner = OWLReasoner_Owlready2_TempClasses(ontology=self._onto)
        new_reasoner = OWLReasoner_FastInstanceChecker(ontology=self._onto,
                                                       base_reasoner=new_base_reasoner)

        new_graph = KnowledgeBase(ontology=self._onto, reasoner=new_reasoner, path=self.graph.path)

        # self._manager.save_ontology(ontology=self._onto, document_iri=IRI.create('file:/test.owl'))
        return new_graph

    def get_removed_nodes(self):
        """
        :return: The set of removed individuals
        """
        return set(self._removed_nodes)

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

    def _remove_nodes(self, nodes, nodes_to_ignore):
        """
        :param nodes: nodes to iterate over
        :param nodes_to_ignore: nodes to ignore in the removal process
        """
        for i in nodes:
            if nodes_to_ignore is None:
                if i in self._lpi:
                    continue
            else:
                if i in self._lpi or i in nodes_to_ignore:
                    continue
            self._removed_nodes.add(i)
            self._manager.remove_axiom(self._onto, OWLDeclarationAxiom(i))
            if len(self._removed_nodes) >= self._nodes_to_remove_len:
                break
