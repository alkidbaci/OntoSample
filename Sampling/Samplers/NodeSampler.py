from ontolearn.knowledge_base import KnowledgeBase
from owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker
from owlapy.model import OWLDeclarationAxiom, IRI
from owlapy.owlready2.temp_classes import OWLReasoner_Owlready2_TempClasses
import logging


logger = logging.getLogger(__name__)

class NodeSampler:
    """
        Random Node Sampler samples x number of nodes from the knowledge base and returns
        the sampled graph.
    """
    def __init__(self, graph: KnowledgeBase):
        self._removed_nodes = None
        self._nodes = None
        self._sampled_nodes = None
        self.graph = graph

    def sample(self, number_of_nodes: int) -> KnowledgeBase:
        """
        Perform the sampling of the knowledge base
        :param number_of_nodes: Number of nodes that will be sampled
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

        nodes_to_remove_len = len(self._nodes) - number_of_nodes
        if nodes_to_remove_len < 0:
            raise ValueError("Number of nodes to sample is greater than number of total nodes in the knowledge base")

        # since graph.all_individuals_set() returns a frozenset, the iteration is random each time.
        for i in self._nodes:
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
