from itertools import chain
from typing import Optional, Callable, Tuple, Generator, List, Union, Final
from owlapy.class_expression import OWLClass, OWLClassExpression
from owlapy.iri import IRI
from owlapy.owl_axiom import OWLEquivalentClassesAxiom
from owlapy.owl_ontology import OWLOntology
from owlapy.owl_ontology_manager import OWLOntologyManager
from owlapy.owl_hierarchy import ClassHierarchy, ObjectPropertyHierarchy, DatatypePropertyHierarchy
from owlapy.utils import OWLClassExpressionLengthMetric, LRUCache
import traceback


def init_length_metric(length_metric: Optional[OWLClassExpressionLengthMetric] = None,
                       length_metric_factory: Optional[Callable[[], OWLClassExpressionLengthMetric]] = None):
    """ Initialize the technique on computing length of a concept"""
    if length_metric is not None:
        pass
    elif length_metric_factory is not None:
        length_metric = length_metric_factory()
    else:
        length_metric = OWLClassExpressionLengthMetric.get_default()

    return length_metric


def init_hierarchy_instances(reasoner, class_hierarchy, object_property_hierarchy, data_property_hierarchy) -> Tuple[
    ClassHierarchy, ObjectPropertyHierarchy, DatatypePropertyHierarchy]:
    """ Initialize class, object property, and data property hierarchies """
    if class_hierarchy is None:
        class_hierarchy = ClassHierarchy(reasoner)

    if object_property_hierarchy is None:
        object_property_hierarchy = ObjectPropertyHierarchy(reasoner)

    if data_property_hierarchy is None:
        data_property_hierarchy = DatatypePropertyHierarchy(reasoner)

    assert class_hierarchy is None or isinstance(class_hierarchy, ClassHierarchy)
    assert object_property_hierarchy is None or isinstance(object_property_hierarchy, ObjectPropertyHierarchy)
    assert data_property_hierarchy is None or isinstance(data_property_hierarchy, DatatypePropertyHierarchy)

    return class_hierarchy, object_property_hierarchy, data_property_hierarchy


def init_named_individuals(individuals_cache_size, ):
    use_individuals_cache = individuals_cache_size > 0
    if use_individuals_cache:
        _ind_cache = LRUCache(maxsize=individuals_cache_size)
    else:
        _ind_cache = None
    return use_individuals_cache, _ind_cache


def init_individuals_from_concepts(include_implicit_individuals: bool = None, reasoner=None, ontology=None,
                                   individuals_per_concept=None):
    assert isinstance(include_implicit_individuals, bool), f"{include_implicit_individuals} must be boolean"
    if include_implicit_individuals:
        assert isinstance(individuals_per_concept, Generator)
        # get all individuals from concepts
        _ind_set = frozenset(chain.from_iterable(individuals_per_concept))
    else:
        # @TODO: needs to be explained
        individuals = ontology.individuals_in_signature()
        _ind_set = frozenset(individuals)
    return _ind_set


def compute_tp_fn_fp_tn(individuals, pos, neg):
    tp = len(pos.intersection(individuals))
    tn = len(neg.difference(individuals))
    fp = len(neg.intersection(individuals))
    fn = len(pos.difference(individuals))
    return tp, fn, fp, tn


def compute_f1_score(individuals, pos, neg) -> float:
    """ Compute F1-score of a concept
    """
    assert type(individuals) == type(pos) == type(neg), f"Types must match:{type(individuals)},{type(pos)},{type(neg)}"
    # true positive: |E^+ AND R(C)  |
    tp = len(pos.intersection(individuals))
    # true negative : |E^- AND R(C)|
    tn = len(neg.difference(individuals))

    # false positive : |E^- AND R(C)|
    fp = len(neg.intersection(individuals))
    # false negative : |E^- \ R(C)|
    fn = len(pos.difference(individuals))

    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        return 0.0

    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        return 0.0

    if precision == 0 or recall == 0:
        return 0.0

    f_1 = 2 * ((precision * recall) / (precision + recall))
    return f_1

def save_owl_class_expressions(expressions: Union[OWLClassExpression, List[OWLClassExpression]],
                               path: str = 'Predictions',
                               rdf_format: str = 'rdfxml') -> None:
    assert isinstance(expressions, OWLClassExpression) or isinstance(expressions[0],
                                                                     OWLClassExpression), "expressions must be either OWLClassExpression or a list of OWLClassExpression"
    if isinstance(expressions, OWLClassExpression):
        expressions = [expressions]
    NS: Final = 'https://dice-research.org/predictions#'

    if rdf_format != 'rdfxml':
        raise NotImplementedError(f'Format {rdf_format} not implemented.')
    from owlapy.owl_ontology_manager import OntologyManager
    # ()
    manager: OWLOntologyManager = OntologyManager()
    # ()
    ontology: OWLOntology = manager.create_ontology(IRI.create(NS))
    # () Iterate over concepts
    for th, i in enumerate(expressions):
        cls_a = OWLClass(IRI.create(NS, str(th)))
        equivalent_classes_axiom = OWLEquivalentClassesAxiom([cls_a, i])
        try:
            manager.add_axiom(ontology, equivalent_classes_axiom)
        except AttributeError:
            print(traceback.format_exc())
            print("Exception at creating OWLEquivalentClassesAxiom")
            print(equivalent_classes_axiom)
            print(cls_a)
            print(i)
            print(expressions)
            exit(1)
    manager.save_ontology(ontology, IRI.create('file:/' + path + '.owl'))
