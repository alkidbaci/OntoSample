import argparse
import logging
import os

from typing import Dict, List

logging.basicConfig(level=logging.INFO)


class NodeIDs:
    """
    This is a utility class that converts csv file of edges into networkx format e.g. (F2M32,F2F30) -> (0,1)
    i.e., it implements cnvt (G_csv → G_nx)

    e.g., if input graph = Family graph, then, input will be Family_edges
            id1,id2
            F1089,Class

    Output will have unique integer id for every instance
            id1,id2
            10, 87

    Usage:  python sampling_expts/src/node_ids.py \
             --node_node_fp "sampling_expts/data/Family_edges.csv" \
             --txt_id_mapping_fp "sampling_expts/data/Family_node_ids.csv" \
             --id_node_mapping_fp "sampling_expts/data/Family_id_node_mapping.csv" \
             --id_id_mapping_fp "sampling_expts/data/Family_id_id_edges.csv"

    """
    def __init__(self):
        """We have 2 dictionaries
         id_to_txt: It will store the id(int) to txt(str) mapping of every node (1, F2F30)
         txt_to_id: It will store the txt(str) to id(int) mapping of every node (F2F30, 1)
           """
        self.id_to_txt: Dict[int, str] = dict()
        self.txt_to_id: Dict[str, int] = dict()

    def load_file(self, node_node_fp,  remove_nodes: List[str], sep=",", has_header=True):
        """ Function performs 2 tasks.
        1.fills up the dictionary/Hash map that will have a unique integer id for
          every Node. Example--> F2F30,1
        2. Once integer id have been created, it makes a list of the ids that are
           connected together in our Knowledgebase.
        @param: node_node_fp : File path of the input file containing edges in str format  (F1123,F1208)
        @param:remove_nodes : List containing the nodes that the user wishes to remove and not include in the dictionary
        @param: sep : Separator in the CSV file
        @param: has_header : True to verify that input file contains a header line
        @returns: loaded: List containing the edges in integer format e.g. (2,12)    """

        logging.info(f"loading file {node_node_fp}...")
        assert os.path.exists(node_node_fp), f"Node node file does not exist."
        loaded: List[List[int]] = []  # [[1,9], [3,8], [2,8]]
        with open(node_node_fp, 'r') as infile:
            for line_num, line in enumerate(infile):
                if has_header and line_num == 0:
                    continue
                node1, node2 = line.strip().split(sep)  # child\trishi
                for n in [node1, node2]:
                    if n in remove_nodes:
                        continue
                    if n not in self.txt_to_id:
                        self.txt_to_id[n] = len(self.txt_to_id)  # child-8, rishi-9
                if node1 in self.txt_to_id and node2 in self.txt_to_id:
                    loaded.append([self.txt_to_id[node1], self.txt_to_id[node2]])
                else:
                    logging.error(f"One of the nodes not found when loading file. {node1} or {node2}")
        logging.info(f"loaded.")
        return loaded

    @staticmethod
    def count_ids(id_id_mapping_fp: str, sep=',', has_header=True):
        """Function to find out the occurrence of each node id
          @param: id_id_mapping_fp: Path of the CSV file containing the id(int) to id(int) mapping of nodes(2,7)
         @return: id_count: Dictionary which has every node id(int) as key and its count as value
         """
        id_count: Dict[str, int] = dict()  # {id:count}
        with open(id_id_mapping_fp, 'r') as infile:
            for line_num, line in enumerate(infile):
                if line_num == 0 and has_header:
                    continue
                node1, node2 = line.strip().split(sep)
                if node1 not in id_count:
                    id_count[node1] = 0
                if node2 not in id_count:
                    id_count[node2] = 0
                id_count[node1] += 1
                id_count[node2] += 1
        return id_count

    @staticmethod
    def construct(id_to_txt_fp, sep=",") -> "NodeIDs":
        """
        @param: id_to_txt_fp: File path of the dictionary containing the integer id of the node
                            and the node in str format e,g,(5, Child)
        @param: sep: Separator in the CSV File
        """
        o = NodeIDs()
        with open(id_to_txt_fp) as id_to_txt_file:
            for line in id_to_txt_file:
                cols = line.strip().split(sep)  # 205,F9M155
                nodeid = int(cols[0])
                nodetxt = cols[1]
                o.id_to_txt[nodeid] = nodetxt
                o.txt_to_id[nodetxt] = nodeid
        return o

    def save_id_to_nodetxt(self, id_node_mapping_fp: str, out_sep=','):
        """
         Makes a dictionary that has integer id as key and the Nodes of graph as Value.
        @param: id_node_mapping_fp: Loads the file containing the edge in int, str format (1,F2F30)
        """
        assert os.path.exists(id_node_mapping_fp), f"Node id file does not exist."
        with open(id_node_mapping_fp, 'w') as id_txt_file:
            for txt_key, id_value in self.txt_to_id.items():
                self.id_to_txt[id_value] = txt_key
            for id, txt in self.id_to_txt.items():
                id_txt_file.write(f"{id}{out_sep}{txt}\n")

    def dump(self, txt_id_mapping_fp: str, loaded_edges: List[List[int]], id_id_mapping_fp: str, out_sep = ","):
        """Dump function dumps the dictionary of Node-id dictionary into a csv file and also
           dumps the list of nodes that areconnected with each other into another csv file
           @param: tx_id_mapping_fp: File path of the csv file to be created that will contain the id(str) to id(int) mapping (F2F30,1)
           @param: id_id_mapping_fp: File path of the csv file to be created that will contain the id(int) to id(int) mapping (2,7)
           @param: loaded_edges: List containing the edges in the id(int) id(int) format
           """
        logging.info(f"Dumping file {txt_id_mapping_fp}...")
        with open(txt_id_mapping_fp, 'w') as node_id_file:
            for txt, id in self.txt_to_id.items():
                node_id_file.write(f"{txt}{out_sep}{id}\n")
        logging.info(f"Dumping file {id_id_mapping_fp}...")
        with open(id_id_mapping_fp, 'w') as id_id_edges_file:
            id_id_edges_file.write(f"id_1{out_sep}id_2\n")
            for edge in loaded_edges:  # [1,9]
                id_id_edges_file.write(f"{edge[0]}{out_sep}{edge[1]}\n")

    def find_node(self, node):
        """Functions receives a node id(int) and returns its corresponding node(str)
          @param: node: Stores the integer id of a node
          @return The str value of that node"""
        return self.id_to_txt[node]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--node_node_fp", required=True, type=str, help="Family dataset in format id(str) id(str) e.g. F1020, F1234")
    parser.add_argument("--txt_id_mapping_fp", required=True, type=str, help="File containing the mapping of txt(str) to id(int) format e.g. F1020,4")
    parser.add_argument("--id_node_mapping_fp", required=True, type=str, help="File containg the mapping of id(int) to node(str) e.g. 4, F1020")
    parser.add_argument("--id_id_mapping_fp", required=True, type=str, help="File created will have the edges in the id(int) id(int) format, e.g. 3,6")
    parser.add_argument("--remove_node_csv", required=False, type=str, default="", help="File created will have the edges in the id(int) id(int) format, e.g. 3,6")
    args = parser.parse_args()
    o = NodeIDs()
    loaded_edges = o.load_file(node_node_fp=args.node_node_fp, remove_nodes = [x.strip() for x in args.remove_node_csv.split(",")])
    o.dump(loaded_edges=loaded_edges,  txt_id_mapping_fp=args.txt_id_mapping_fp,
           id_id_mapping_fp=args.id_id_mapping_fp)

    o.save_id_to_nodetxt(id_node_mapping_fp=args.id_node_mapping_fp)

    id_count = o.count_ids(id_id_mapping_fp=args.id_id_mapping_fp)
    logging.info(f"Isolated Nodes: {[k for k,v in o.count_ids(args.id_id_mapping_fp).items() if v==1]}")
