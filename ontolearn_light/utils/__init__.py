"""Ontolearn utils."""
import datetime
import os
import pickle
import random
import time
from typing import Callable, Set, TypeVar, Tuple, Union
from owlapy.class_expression import OWLClass
from owlapy.iri import IRI
from owlapy.meta_classes import HasIRI
from owlapy.owl_individual import OWLNamedIndividual
from ontolearn_light.utils.log_config import setup_logging  # noqa: F401
import pandas as pd
from .static_funcs import compute_f1_score
Factory = Callable

# DEFAULT_FMT = '[{elapsed:0.8f}s] {name}({args}) -> {result}'
DEFAULT_FMT = 'Func:{name} took {elapsed:0.8f}s'
flag_for_performance = False


def parametrized_performance_debugger(fmt=DEFAULT_FMT):
    def decorate(func):
        if flag_for_performance:
            def clocked(*_args):
                t0 = time.time()
                _result = func(*_args)
                elapsed = time.time() - t0
                name = func.__name__
                args = ', '.join(repr(arg) for arg in _args)
                result = repr(_result)
                print(fmt.format(**locals()))
                return _result

            return clocked
        else:
            return func

    return decorate


def performance_debugger(func_name):
    def function_name_decorator(func):
        def debug(*args, **kwargs):
            start = time.time()
            r = func(*args, **kwargs)
            print(func_name, ' took ', round(time.time() - start, 4), ' seconds')

            return r

        return debug

    return function_name_decorator


def create_experiment_folder(folder_name='Log'):
    from ontolearn_light.utils import log_config
    if log_config.log_dirs:
        path_of_folder = log_config.log_dirs[-1]
    else:
        directory = os.getcwd() + '/' + folder_name + '/'
        folder_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path_of_folder = directory + folder_name
        os.makedirs(path_of_folder)
    return path_of_folder, path_of_folder[:path_of_folder.rfind('/')]


def serializer(*, object_: object, path: str, serialized_name: str):
    with open(path + '/' + serialized_name + ".p", "wb") as f:
        pickle.dump(object_, f)
    f.close()


def deserializer(*, path: str, serialized_name: str):
    with open(path + "/" + serialized_name + ".p", "rb") as f:
        obj_ = pickle.load(f)
    f.close()
    return obj_


def apply_TSNE_on_df(df) -> None:
    from sklearn.manifold import TSNE
    from matplotlib import pyplot as plt
    low_emb = TSNE(n_components=2).fit_transform(df)
    plt.scatter(low_emb[:, 0], low_emb[:, 1])
    plt.title('Instance Representatons via TSNE')
    plt.show()


def balanced_sets(a: set, b: set) -> Tuple[Set, Set]:
    """
    Balance given two sets through sampling without replacement.
    Returned sets have the same length.
    @param a:
    @param b:
    @return:
    """

    if len(a) > len(b):
        sampled_a = random.sample(list(a), len(b))
        return set(sampled_a), b
    elif len(b) > len(a):
        sampled_b = random.sample(list(b), len(a))
        return a, set(sampled_b)
    else:
        assert len(a) == len(b)
        return a, b


def read_csv(path)->Union[None,pd.DataFrame]:
    """
    Path leads a folder containing embeddings in csv format.
    indexes correspond subjects or predicates or objects in n-triple.
    @param path:
    @return:
    """
    if assertion_path_isfile(path):
        df = pd.read_csv(path, index_col=0)
        assert (df.all()).all()  # all columns and all rows are not none.
        return df
    else:
        return None


def assertion_path_isfile(path) -> bool:
    try:
        assert path is not None
    except AssertionError:
        print(f'Path can not be:{path}')
        return False

    try:
        assert os.path.isfile(path)
    except (AssertionError, TypeError):
        print(f'Input:{path} not found.')
        return False
    return True


def sanity_checking_args(args):
    try:
        assert os.path.isfile(args.path_knowledge_base)
    except AssertionError:
        print(f'--path_knowledge_base ***{args.path_knowledge_base}*** does not lead to a file.')
        raise

    try:
        assert os.path.isfile(args.path_knowledge_base_embeddings)
    except AssertionError:
        print(f'--path_knowledge_base_embeddings ***{args.path_knowledge_base_embeddings}*** does not lead to a file.')
        raise

    assert args.min_length > 0
    assert args.max_length > 0
    assert args.min_num_concepts > 0
    assert args.min_num_concepts > 0
    assert args.min_num_instances_per_concept > 0
    assert os.path.isfile(args.path_knowledge_base)
    if hasattr(args, 'num_fold_for_k_fold_cv'):
        assert args.num_fold_for_k_fold_cv > 0
    if hasattr(args, 'max_test_time_per_concept'):
        assert args.max_test_time_per_concept > 1

    if hasattr(args, 'num_of_sequential_actions'):
        assert args.num_of_sequential_actions > 0

    if hasattr(args, 'batch_size'):
        assert args.batch_size > 1


_T = TypeVar('_T', bound=HasIRI)


def _read_iri_file(file: str, type_: Factory[[IRI], _T]) -> Set[_T]:
    """Read a text file containing IRIs (one per line) and return the content as a set of instances created by the
    given type

    Args:
        file: path to the text file with the IRIs of the named individuals
        type_: factory or type to create from the IRI

    Returns:
        set of type_ instances with these IRIs
    """

    def optional_angles(iri: str):
        if iri.startswith('<'):
            return iri[1:-1]
        else:
            return iri

    with open(file, 'r') as f:
        inds = map(type_,
                   map(IRI.create,
                       map(optional_angles,
                           f.read().splitlines())))
    return set(inds)


def read_individuals_file(file: str) -> Set[OWLNamedIndividual]:
    """Read a text file containing IRIs of Named Individuals (one per line) and return the content as a set of OWL
    Named Individuals

    Args:
        file: path to the text file with the IRIs of the named individuals

    Returns:
        set of OWLNamedIndividual with these IRIs
    """
    return _read_iri_file(file, OWLNamedIndividual)


def read_named_classes_file(file: str) -> Set[OWLClass]:
    """Read a text file containing IRIs of OWL Named Classes (one per line) and return the content as a set of OWL
    Classes

    Args:
        file: path to the text file with the IRIs of the classes

    Returns:
        set of OWLNamedIndividual with these IRIs
    """
    return _read_iri_file(file, OWLClass)
