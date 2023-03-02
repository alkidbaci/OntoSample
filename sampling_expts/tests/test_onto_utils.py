import unittest

import sys
sys.path.append("../src")

from sampling_expts.src.onto_utils import NetworkXGraph


class NxToOwlTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.id_to_txt_fp = "sampling_expts/data/Family_id_node_mapping.csv"
        self.sample_graph_fp = "sampling_expts/sampled_data/Family/random_walk.csv"

    def test_construct_nx(self):
        self.g = NetworkXGraph.load(self.sample_graph_fp,self.id_to_txt_fp)
        # F10F175, Person
        self.assertEqual(self.g.edges[23].str1, "Daughter")
        self.assertEqual(self.g.edges[23].str2, "F1F5")


if __name__ == '__main__':
    unittest.main()
