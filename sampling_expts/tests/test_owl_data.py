import tempfile
import unittest

import xmltodict

from sampling_expts.src import file_utils
from sampling_expts.src.onto_utils import OWLData, OWLNode


class OwlDataTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fp = "sampling_expts/data/Family_benchmark_rich_background.owl"
        self.owl = OWLData(self.fp)
        self.j = self.owl.owl_as_json["rdf:RDF"]
        # file_utils.write_list_to_file(list_data = OWLData(self.fp).csv_edges(of_sampled_graph=False), fp="xxx")

    def test_csv_file_creation(self):
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
        file_utils.write_list_to_file(list_data=OWLData(self.fp).csv_edges(of_sampled_graph=False), fp=out_path)
        print(f"csv file is at {out_path}")
        assert file_utils.read_tsv(out_path)[0] is not None

    def test_to_json(self):
        assert self.j['@xmlns:family'] == 'http://www.benchmark.org/family#'

    def test_owl_nodes(self):
        edges_pool = ["F10F172, F10F179",
                      "F10F172, F10M173",
                      "F10F177, F10F175",
                      "F10F177, F10M178"]
        self.owl.prune_rdf_edges(edges_csv_cand_pool=edges_pool)
        # TODO @rdf:resource showing up as node id (esp. tgt id).
        expected_pruned_edges = [OWLNode.format_as_id_id(src_id=x.split(",")[0], tgt_id=x.split(",")[1]) for x in edges_pool]
        obtained_pruned_edges = self.owl.csv_edges(of_sampled_graph=True)
        assert obtained_pruned_edges == expected_pruned_edges, f"Expected: {expected_pruned_edges}\n but got:\n{obtained_pruned_edges}"

        owl_parsed_as_xml = xmltodict.unparse(self.owl.owl_as_json, pretty=True)
        self.assertGreater(len(owl_parsed_as_xml), 10, msg="The pruned OWL (after sampling some edges and nodes) "
                                                           "is not corrected formatted.")


if __name__ == '__main__':
    unittest.main()
