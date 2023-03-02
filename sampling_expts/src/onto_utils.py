""" Work in Progress. To convert the networkX into OWL format"""



import argparse
import logging
import os
from dataclasses import dataclass
from typing import List, Set, Dict, Any

import xmltodict

import sys

from sampling_expts.src.node_ids import NodeIDs



# Input = 7, 10
# Output = Family, Sister
# Given: map from 7 -> Family
# Given: input file with from 7, 10 (id id edges)
# Given: ontology file -- Family Sister xml dataset





@dataclass
class NetworkXEdge:
    id1: int
    id2: int
    str1: str
    str2: str

    def __repr__(self):
        return f"{self.str1},{self.str2}"

    def reversed__repr__(self):
        return f"{self.str2},{self.str1}"


class NetworkXGraph:
    def __init__(self):
        self.edges: List[NetworkXEdge] = []

    def add_edge(self, e: NetworkXEdge):
        self.edges.append(e)

    def get_bidirectional_edges(self) -> Set[str]:
        # a,b and b,a are present in the graph.
        f: Set[str] = set([f"{e}" for e in self.edges])  # {ab, ac, bc, bd, be, ca ...}
        r: Set[str] = set([f"{e.reversed__repr__()}" for e in self.edges])  # {ba, ca, cb, db, eb, ac...}
        return f.intersection(r)

    @staticmethod
    def load(sampled_graph_fp, id_node_mapping_fp, sep=',') -> "NetworkXGraph":
        """
        :param sampled_graph_fp (output from ball of fur e.g, data/sampled_graphs/random_walk.csv)
        :param id_node_mapping_fp e.g., "sampling_expts/data/Family_id_node_mapping.csv"
        """
        logging.info(f"Loading id node mapping {id_node_mapping_fp}..")
        o = NodeIDs.construct(id_node_mapping_fp)
        assert os.path.exists(sampled_graph_fp), f"Sampled graph file does not exist: {sampled_graph_fp}"

        logging.info(f"Loading Sampled Graph {sampled_graph_fp}..")
        g = NetworkXGraph()
        with open(sampled_graph_fp, 'r') as infile:
            for line in infile:
                node1_int, node2_int = line.strip().split(sep)
                node1_int = int(node1_int)
                node2_int = int(node2_int)
                node1_str, node2_str = o.id_to_txt[node1_int], o.id_to_txt[node2_int]
                e = NetworkXEdge(id1=node1_int, id2=node2_int, str1=node1_str, str2=node2_str)
                g.add_edge(e)
        logging.info(f"Loaded NetworkXGraph.")
        return g

    def save(self, fp):
        logging.info(f"\nSaving NetworkX graph at {fp}")
        with open(fp, 'w') as outfile:
            for e in self.edges:
                outfile.write(f"{e}\n")


class OWLNode:

    def __init__(self, rdf_about: str, rdf_type: str, rdf_edges: Dict[str, List[Any]]):
        # "http://www.benchmark.org/family#F10F174" -> F10F174
        self.src_id: str = OWLNode.extract_id(s=rdf_about)
        self.rdf_type = rdf_type
        self.rdf_edges = rdf_edges
        self.sampled_rdf_edges: Dict[str, List[Any]] = {}

    @staticmethod
    def extract_id(s):
        # "http://www.benchmark.org/family#F10F174" -> F10F174
        tgt_id_uri = s['@rdf:resource'] if isinstance(s, dict) else s
        return tgt_id_uri.split("#")[-1].strip()

    def is_valid(self, rdf_edges) -> bool:
        return len(rdf_edges) > 0

    def target_ids(self, rdf_edges) -> List[str]:
        for rel_name, all_rel_tgts in rdf_edges.items():
            for tgt_node in all_rel_tgts:
                yield OWLNode.extract_id(tgt_node)

    def csv_edges(self, rdf_edges, sep=",") -> List[str]:
        return [f"{self.format_as_id_id(src_id=self.src_id, sep=sep, tgt_id=tgt_id)}"
                for tgt_id in self.target_ids(rdf_edges=rdf_edges)]

    @staticmethod
    def format_as_id_id(src_id, tgt_id, sep=","):
        tgt_id_uri = tgt_id['@rdf:resource'] if isinstance(tgt_id, dict) else tgt_id
        return f"{src_id.strip()} {sep} {tgt_id_uri.strip()}"

    def prune_rdf_edges(self, edges_csv_cand_pool: List[str], sep = ","):
        """
        @param: keep_edges is the postprocessed output from a sampling library such as littleballoffur
        # F10F172, F10F181
        # F10F172, F10F179
        # F10F001, F10F165
        #    ....
        """
        # remove some items from self.target_edges
        cand_edges_set: Set[str] = set() # TODO pass set instead of list to avoid doing this for every thing block
        for e in edges_csv_cand_pool:
            # F10F001, F10F165
            cand_src, cand_tgt = [x.strip() for x in e.split(sep)]
            cand_edges_set.add(self.format_as_id_id(src_id=cand_src, tgt_id=cand_tgt, sep=sep))

        for rel_name, rel_tgts in self.rdf_edges.items():

            # Sometimes rel_tgts can be list of dicts, and sometimes single dict
            if not isinstance(rel_tgts, list):
                rel_tgts = [rel_tgts]

            for rel_tgt_dict in rel_tgts:
                rel_tgt = rel_tgt_dict["@rdf:resource"] if isinstance(rel_tgt_dict, dict) else rel_tgt_dict
                # F10F001, F10F165
                if not rel_tgt:
                    continue
                # rel_tgt looks like so: {'@rdf:resource': 'http://www.benchmark.org/family#F10F179'}
                if self.format_as_id_id(src_id=self.src_id, tgt_id=self.extract_id(rel_tgt), sep=sep) in cand_edges_set:
                    # e.g, add this line:
                    # <family:hasChild rdf:resource="http://www.benchmark.org/family#F10F186"/>
                    if rel_name not in self.sampled_rdf_edges:
                        self.sampled_rdf_edges[rel_name] = []
                    self.sampled_rdf_edges[rel_name].append(rel_tgt)

        return self.sampled_rdf_edges


class OWLData:
    """
      Creates an object over owl file that looks like so:

        #     <Class rdf:about="http://www.benchmark.org/family#Sister">
        #         <rdfs:subClassOf rdf:resource="http://www.benchmark.org/family#Female"/>
        #         <rdfs:subClassOf rdf:resource="http://www.benchmark.org/family#PersonWithASibling"/>
        #     </Class>

        # self.onto_edges: List[OWLRelations] = []
        #     <Thing rdf:about="http://www.benchmark.org/family#F10F172">
        #         <rdf:type rdf:resource="http://www.w3.org/2002/07/owl#NamedIndividual"/>
        #         <rdf:type rdf:resource="http://www.benchmark.org/family#Female"/>
        #         <rdf:type rdf:resource="http://www.benchmark.org/family#Grandmother"/>
        #         <rdf:type rdf:resource="http://www.benchmark.org/family#Mother"/>
        #         <rdf:type rdf:resource="http://www.benchmark.org/family#Person"/>
        #         <family:hasChild rdf:resource="http://www.benchmark.org/family#F10F179"/>
        #         <family:hasChild rdf:resource="http://www.benchmark.org/family#F10F181"/>
        #         <family:hasChild rdf:resource="http://www.benchmark.org/family#F10F186"/>
        #         <family:hasChild rdf:resource="http://www.benchmark.org/family#F10F195"/>
        #         <family:hasChild rdf:resource="http://www.benchmark.org/family#F10M173"/>
        #         <family:married rdf:resource="http://www.benchmark.org/family#F10M171"/>
        #     </Thing>
    """
    def __init__(self, owl_fp):
        self.owl_fp = owl_fp
        with open(owl_fp) as owl_file:
            self.owl_as_json = xmltodict.parse(owl_file.read())

        # owl -> str_csv  [F10F172, F10F181 (theirs)]
        # F10F172, F10F181
        # F10F172, F10F179
        # F10F001, F10F165
        #    ....
        # csv -> nx
        # F10F172, F10F181 -> 7, 9
        # F10F172, F10F179 -> 7, 8
        # F10F001, F10F165 -> 1, 3
        #    ....
        # sample nx
        # 7, 8
        # 1, 3
        # nx to str_csv
        # F10F172, F10F179
        # F10F001, F10F165
        # str_csv to owl
        # Thing: F10F001
        #    ....
        # Thing: F10F165
        #    ....
        self.onto_nodes: List[OWLNode] = []
        for thing in self.owl_as_json["rdf:RDF"]["Thing"]:

            # Sometimes rel_tgts can be list of dicts, and sometimes single dict
            # So, we now ensure that the value is an array of dicts.
            rdf_edges = {k: thing[k] if isinstance(thing[k], list) else [thing[k]] for k in thing.keys() if k != "@rdf:about" and k != "rdf:type"}

            thing_obj = OWLNode(rdf_about=thing["@rdf:about"],
                                rdf_type=thing["rdf:type"],
                                rdf_edges= rdf_edges
                                )
            self.onto_nodes.append(thing_obj)

    def csv_edges(self, of_sampled_graph:bool, sep=",") -> List[str]:
        csv_edges: List[str] = []
        for n in self.onto_nodes:
            # csv_edges block by block:
            curr_node_edges = n.csv_edges(rdf_edges=n.sampled_rdf_edges if of_sampled_graph else n.rdf_edges, sep=sep)
            csv_edges.extend(curr_node_edges)
        return csv_edges

    def prune_rdf_edges(self, edges_csv_cand_pool: List[str], sep = ","):
        # for each onto_node, call prune_rdf_edges to set pruned_rdf_edges attr.
        # prune_rdf_edges block by block:
        # e.g., a block looks like this:
        #     <Thing rdf:about="http://www.benchmark.org/family#F10F177">
        #         <rdf:type rdf:resource="http://www.w3.org/2002/07/owl#NamedIndividual"/>
        #         <rdf:type rdf:resource="http://www.benchmark.org/family#Daughter"/>
        #         <rdf:type rdf:resource="http://www.benchmark.org/family#Female"/>
        #         <rdf:type rdf:resource="http://www.benchmark.org/family#Granddaughter"/>
        #         <rdf:type rdf:resource="http://www.benchmark.org/family#Person"/>
        #         <rdf:type rdf:resource="http://www.benchmark.org/family#Sister"/>
        #         <family:hasParent rdf:resource="http://www.benchmark.org/family#F10F174"/>
        #         <family:hasParent rdf:resource="http://www.benchmark.org/family#F10M173"/>
        #         <family:hasSibling rdf:resource="http://www.benchmark.org/family#F10F175"/>
        #         <family:married rdf:resource="http://www.benchmark.org/family#F10M178"/>
        #     </Thing>
        for n in self.onto_nodes:
            n.prune_rdf_edges(edges_csv_cand_pool=edges_csv_cand_pool, sep=sep)



@dataclass
class NetworkxOverlaidWithOWL:
    g: NetworkXGraph
    owl: OWLData

    def convert(self):
        # from g, get a set of node str. (e.g., 10% of the original nodes)
        # g_nodes_str: Set[str] = {}
        # g_edges_str: Set[str] = {}
        #
        # # Should we remove?     <Thing rdf:about="http://www.benchmark.org/family#F10F172">
        # rel_subset: OWLRelations = [x for x in owl.edges if x.present_in(g_nodes_str)]
        # # Should we remove?     <Thing rdf:about="http://www.benchmark.org/family#F10F172">
        # #                       <family:hasChild rdf:resource="http://www.benchmark.org/family#F10F181"/>
        # rel_edge_subset = [x for x in rel_subset if x.present_in(g_edges_str)]
        pass


if __name__ == '__main__':
    # DONE: add argparse to take the following four paths as python arguments.
    # DONE call entire pipeline through a shell script
    # DONE streamlit demo
    # DONE add test cases
    # POSTPONED TODO sampled graph str to owl

    parser = argparse.ArgumentParser()
    parser.add_argument("--sampled_graph_fp", required=True, type=str, help="Sampled graph file path in id(int),id(int) format e.g, 10, 13 e.g., sampling_expts/sampled_data/Family/random_walk.csv")
    parser.add_argument("--id_node_mapping_fp", required=True, type=str, help="ID node mapping fp id(int),nodetxt(str), e.g. sampling_expts/data/Family_id_node_mapping.csv")
    parser.add_argument("--sampled_graph_str_fp", required=True, type=str, help="Dir where your sampled graph will be saved e.g. sampling_expts/sampled_data/Family/random_walk_str.csv")
    parser.add_argument("--owl_fp", required=True, type=str, help="OWL file path containing node relations e.g. hasChild, hasSibling e.g sampling_expts/data/Family_benchmark_rich_background.owl ")
    args = parser.parse_args()

    newline = "\n"

    g = NetworkXGraph.load(sampled_graph_fp=args.sampled_graph_fp,
                           id_node_mapping_fp=args.id_node_mapping_fp)
    g.save(fp=args.sampled_graph_str_fp)
    print(f"The following edges are bidrectional:\n{newline.join(g.get_bidirectional_edges())}")

    nx_to_owl = NetworkxOverlaidWithOWL(g=g, owl=OWLData(args.owl_fp))
