import json

from ontolearn.knowledge_base import KnowledgeBase
from owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker
from owlapy.model import OWLDeclarationAxiom, IRI
from owlapy.owlready2.temp_classes import OWLReasoner_Owlready2_TempClasses
import logging


logger = logging.getLogger(__name__)

class RandomNodeSamplerLPFirst:
    """
        Random Node Sampler LP first samples x number of nodes from the knowledge base and returns
        the sampled graph. Starts with LP nodes first.
    """
    def __init__(self, graph: KnowledgeBase):
        self._removed_nodes = None
        self._nodes = None
        self._sampled_nodes = None
        self.graph = graph
        self._lpi = None

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
        onto = self.graph.ontology()
        # Getting the OWLOntologyManager object from onto
        manager = onto.get_owl_ontology_manager()
        # Getting all OWLNamedIndividual objects from KB
        self._nodes = self.graph.all_individuals_set()
        self._sampled_nodes = set()
        self._removed_nodes = set()
        self._lpi = list(self.get_lp_individuals(lp_path))

        nodes_to_remove_len = len(self._nodes) - number_of_nodes
        if nodes_to_remove_len < 0:
            raise ValueError("Number of nodes to sample is greater than number of total nodes in the knowledge base")

        # since graph.all_individuals_set() returns a frozenset, the iteration is random each time.
        for i in self._nodes:
            if i in self._lpi:
                continue
            self._removed_nodes.add(i)
            # Remove the individual along with all Abox axioms associated to them
            manager.remove_axiom(onto, OWLDeclarationAxiom(i))
            if len(self._removed_nodes) >= nodes_to_remove_len:
                break
        self._sampled_nodes = set(self._nodes) - self._removed_nodes

        logger.info("Individuals removed: {}".format(len(self._removed_nodes)))
        # Building new reasoner
        new_base_reasoner = OWLReasoner_Owlready2_TempClasses(ontology=onto)
        new_reasoner = OWLReasoner_FastInstanceChecker(ontology=onto,
                                                       base_reasoner=new_base_reasoner)

        new_graph = KnowledgeBase(ontology=onto, reasoner=new_reasoner, path=self.graph.path)

        # manager.save_ontology(ontology=onto, document_iri=IRI.create('file:/test.owl'))
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