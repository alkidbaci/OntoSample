import unittest
import sys
import os
from littleballoffur import GraphReader

sys.path.append(".")
sys.path.append("../src")
from sampling_expts.src.perf_results import SamplingPerformance


class SamplingMetricsTestCase(unittest.TestCase):

    def setUp(self) -> None:
        reader = GraphReader("Family_id_id")
        reader.base_url = f"file://{os.path.join(os.path.abspath(os.curdir), 'sampling_expts')}/data/"
        self.G = reader.get_graph()

    def test_avg_degree(self):
        ob = SamplingPerformance(G=self.G)
        assert ob.metric_AvgDegree(self.G) > 0, "Average degree cannot be less than 0"


if __name__ == '__main__':
    unittest.main()
