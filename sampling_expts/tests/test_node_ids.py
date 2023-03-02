import unittest

import sys
sys.path.append(".")
sys.path.append("../src")

from sampling_expts.src.node_ids import NodeIDs


class NodeIDsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.id_to_txt_fp = "sampling_expts/data/Family_id_node_mapping.csv"

    def test_id_to_txt(self):
        self.o = NodeIDs.construct(self.id_to_txt_fp)
        self.assertEqual(self.o.id_to_txt[13], "Grandmother")
        self.assertEqual(self.o.txt_to_id["Grandmother"], 13)


if __name__ == '__main__':
    unittest.main()
