"""This file implements 9 samplers. It implements the grid sampling of nodes and edges (5% to 25%).
More samplers can be added with a minor change in the code. """

import argparse
import logging
import math
import os
from dataclasses import dataclass
import tempfile

import networkx as nx


from typing import List, Dict

from littleballoffur import GraphReader, RandomWalkSampler, RandomEdgeSampler, RandomNodeEdgeSampler, \
    HybridNodeEdgeSampler, PageRankBasedSampler, MetropolisHastingsRandomWalkSampler,\
    CommonNeighborAwareRandomWalkSampler, RandomNodeSampler, Sampler,FrontierSampler

logging.basicConfig(level=logging.INFO)


@dataclass
class SamplingMetrics:
    """ This class holds the values/parameters of the graph
         average_degree , degree_correlation, cluster_coefficient
         as_tsv() : returns the above parameters
         dist(): Calculates the L2 Norm (avg_degree, cluster_coeff,degree_corr) of each sampler """
    avg_degree: float
    cluster_coeff: float
    degree_corr: float
    sampler_name: str
    num_nodes: int
    num_edges: int

    def as_tsv(self):
        """@return: The name , avg degree, degree correlation, correlation coefficient of a graph
        """
        return f"{self.sampler_name}\t{self.avg_degree:0.3f}\t{self.cluster_coeff:0.3f}\t{self.degree_corr:0.3f}"

    def dist(self, other: "SamplingMetrics") -> float:
        """Calculates the L2 Norm of the Sampling Metrics of the original Graph
        and the Sampled graph
        @param """
        return math.sqrt(
            math.pow(self.avg_degree - other.avg_degree, 2) +
            math.pow(self.cluster_coeff - other.cluster_coeff, 2) +
            math.pow(self.degree_corr - other.degree_corr, 2)
        )


class SamplingPerformance:
    """ It calculates the metrics of the Graph
     Metrics: <Average Degree, Degree Correlation, Cluster Coefficient>
     Usage: python sampling_expts/src/perf_results.py
          --graph_name Family_id_id
          --percent_nodes 25
          --percent_edges 25
          --samplers all
          --sampled_graph_dir "sampling_expts/data/"

     """
    def __init__(self, G):
        self.G = G

    def metric_AvgDegree(self, g) -> float:
        """Calculates the average degree of graph g
        @param: g: Input Graph. It can be the original graph or the sampled graph  """
        try:
            return sum([x[1] for x in g.degree]) / len(g.degree)
        except Exception as e:
            return 0.0  # Max L2 dist from gold ref for 0.0.

    def metric_ClusterCoefficient(self, g) -> float:
        """ Calculates the cluster coefficient of graph G
        @param: g: Input Graph. It can be the original graph or the sampled graph"""
        try:
            return nx.transitivity(g)
        except Exception as e:
            return 0.0  # Max L2 dist from gold ref for 0.0.

    def metric_DegreeCorrelation(self, g) -> float:
        """ Calculates the degree correlation of graph G
        @param: g: Input Graph. It can be the original graph or the sampled graph"""
        try:
            return nx.degree_pearson_correlation_coefficient(g)
        except Exception as e:
            return 0.0  # Max L2 dist from gold ref for 0.0.

    def measure(self, sampler):
        """It calculates the metrics of the sampled graph
        @param: sampler: This contains different sampler used in the code
        @return: SamplerMetrics"""
        sampler_name = (f"{sampler.__class__}").split(".")[-1].replace("'>", "")
        logging.info(f"Measuring perf of {sampler_name} ...")
        sampled_graph = sampler.sample(self.G)
        return SamplingMetrics(avg_degree=self.metric_AvgDegree(sampled_graph),
                               cluster_coeff=self.metric_ClusterCoefficient(sampled_graph),
                               degree_corr=self.metric_DegreeCorrelation(sampled_graph),
                               sampler_name=sampler_name,
                               num_nodes=sampled_graph.number_of_nodes(),
                               num_edges=sampled_graph.number_of_edges()
                               ), sampled_graph

    def measure_orig(self) -> SamplingMetrics:
        """It calculates the metrics of the original graph
         @return: SamplerMetrics"""
        return SamplingMetrics(avg_degree=self.metric_AvgDegree(self.G),
                               cluster_coeff=self.metric_ClusterCoefficient(self.G),
                               degree_corr=self.metric_DegreeCorrelation(self.G),
                               sampler_name="100pc",
                               num_nodes=self.G.number_of_nodes(),
                               num_edges=self.G.number_of_edges()
                               )

    @staticmethod
    def sort_metrics_by_perf(gold_ref, metrics: List[SamplingMetrics]) -> List[SamplingMetrics]:
        """Sort the metrics of different samplers
        @param: gold_ref : List Containing the metrics of the original(unsampled) graph
        @param: metrics: List containing the metrics of different samplers
        @return : List ->metrics in sorted order"""
        metrics.sort(key=lambda x: x.dist(gold_ref), reverse=False)
        return metrics

@dataclass
class SamplingWinner:
    """ Holds the values such as metrics(of Graph) and performs the grid search to bring out the best sampler by sampling
        nodes and edges (5 to 25)% each.
        """
    metrics: SamplingMetrics
    dist_ref: float
    pc_nodes: float
    pc_edges: float

    @staticmethod
    def find(ob: SamplingPerformance, samplers) -> "SamplingWinner":
        """ It returns the L2 Norm of the input Sampler
        @param: samplers: List of smaplers
        @return : The metrics of the sampled graph"""
        metrics = [ob.measure(sampler=sampler)[0] for sampler in samplers]
        gold_ref = ob.measure_orig()
        ob.sort_metrics_by_perf(gold_ref=gold_ref, metrics=metrics)
        # best performing sampler is metrics[0]:
        return SamplingWinner(metrics=metrics[0],
                              dist_ref=metrics[0].dist(gold_ref),
                              pc_nodes=100.0 * (metrics[0].num_nodes / gold_ref.num_nodes),
                              pc_edges=100.0 * (metrics[0].num_edges / gold_ref.num_edges),
                              )

    def as_tsv(self):
        return f"{self.pc_nodes:0.3f}\t{self.pc_edges:0.3f}\t{self.metrics.as_tsv()}\t{self.dist_ref:0.3f}"

    @staticmethod
    def grid_search(ob: SamplingPerformance, pc_node_min:int, pc_node_max:int, pc_node_step:int, pc_edge_min:int, pc_edge_max:int, pc_edge_step:int):
        """ It brings the best samplers
        @param: ob: object of type SamplingPerformance
        @param: pc_node_min: Stores the value of min percentage of nodes to be sampled. default = 5
        @param: pc_node_max: Stores the value of max percentage of nodes to be sampled. default = 25
        @param: pc_node_step: Step size of the for node loop, default = 5
        @param: pc_edge_min: Stores the value of min percentage of edges to be sampled. default = 5
        @param: pc_node_max: Stores the value of max percentage ofedges to be sampled. default = 25
        @param: pc_edge_step: Step size of the for edge loop, default = 5
        """
        winners_grid: List[SamplingWinner] = []
        grid_item_num = 0
        for pc_node in range(pc_node_min,pc_node_max+1,pc_node_step):
            for pc_edge in range(pc_edge_min,pc_edge_max+1,pc_edge_step):
                grid_item_num  += 1
                logging.info(f"\n\n{'*'*80}\nGRID [{grid_item_num}]\tpc_node: {pc_node}\tpc_edge: {pc_edge}\t{nx.is_connected(ob.G)}")
                nn = int(ob.G.number_of_nodes() * (pc_node / 100.0))
                ne = int(ob.G.number_of_edges() * (pc_edge / 100.0))
                samplers = default_samplers(ne=ne, nn=nn)
                winners_grid.append(SamplingWinner.find(ob=ob, samplers=samplers))
        super_winners: Dict[str, int] = dict()
        for w in winners_grid:
            if w.metrics.sampler_name not in super_winners:
                super_winners[w.metrics.sampler_name] = 0
            super_winners[w.metrics.sampler_name] += 1

        return sorted(super_winners.items(), key=lambda item: item[1], reverse=True), winners_grid



def default_samplers(ne, nn) -> List[Sampler]:
    """@param: nn: Number of edges after n percent sampling
       @param: ne: Number of edges after n percent sampling
       @return: List of Sampler class """
    default_sampler_names  = [
        "randomedge", "randomnode", "randomwalk", "pagerankbased",
        "metropolishastingsrandomwalk", "commonneighborawarerandomwalk",
        "randomnodeedge","hybridnodeedge", "frontier"
    ]
    return [sampler_factory(x, ne=ne, nn=nn) for x in default_sampler_names]


def sampler_factory(sampler_name, ne, nn) -> Sampler:
    """@param: sampler_name: Name of the sampler
       @param: nn: Number of edges after n percent sampling
       @param: ne: Number of edges after n percent sampling
       @return: factory: List containing the samplers inputted by the user
       """
    sampler_name = sampler_name.lower().strip().replace(" ","").replace("_","")
    if sampler_name.endswith("sampler"):
        sampler_name = sampler_name.replace("sampler", "")
    factory = {"randomedge": RandomEdgeSampler(number_of_edges=ne),
               # RandomEdgeSamplerWithInduction(number_of_edges=ne),
               # RandomEdgeSamplerWithPartialInduction(),
               "randomnode": RandomNodeSampler(number_of_nodes=nn),
               "randomwalk": RandomWalkSampler(number_of_nodes=nn),
               "pagerankbased": PageRankBasedSampler(number_of_nodes=nn),
               # NonBackTrackingRandomWalkSampler(number_of_nodes=nn), # Like InductionSampler picks ALL edges
               "metropolishastingsrandomwalk": MetropolisHastingsRandomWalkSampler(number_of_nodes=nn),
               "commonneighborawarerandomwalk": CommonNeighborAwareRandomWalkSampler(number_of_nodes=nn),
               "randomnodeedge": RandomNodeEdgeSampler(number_of_edges=ne),
               "hybridnodeedge": HybridNodeEdgeSampler(number_of_edges=ne),
               "frontier": FrontierSampler(number_of_nodes=nn)

               }

    if sampler_name not in factory:
        raise Exception(f"Requested sampler {sampler_name} is not yet implemented.")
    return factory[sampler_name]


def main_perf_at_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", default=f"file://{os.path.join(os.path.abspath(os.curdir), 'sampling_expts')}/data/", required=False, type=str, help="Parent path to the directory where graph_name.csv resides")
    # parser.add_argument("--base_path", default="file:///Users/rishigarg/PycharmProjects/sampling/sampling_expts/data/", required=False, type=str, help="Parent path to the directory where graph_name.csv resides")
    parser.add_argument("--graph_name", required=True, type=str, help="Name of the graph e.g., Family (we will expect graph_name_edges.csv.")
    parser.add_argument("--percent_nodes", required=True, type=float, help="What percentage of the nodes to sample e.g, 25")
    parser.add_argument("--percent_edges", required=True, type=float, help="What percentage of the edges to sample e.g, 25")
    parser.add_argument("--samplers", required=True, type=str, help="all OR randomedge,randomwalk,...")
    # parser.add_argument("--random_seed", required=True, type=float, help="Initial random seed")
    parser.add_argument("--sampled_graph_dir", required=True, type=str, help="Where to save sampled graphs")
    parser.add_argument("--out_path", required=False, default=tempfile.NamedTemporaryFile(delete=False, suffix=".tsv").name, type=str, help="Where to save output table")
    args = parser.parse_args()

    reader = GraphReader(args.graph_name)
    # reader.base_url = "file:///Users/rishigarg/PycharmProjects/sampling/sampling_expts/"
    reader.base_url = args.base_path
    G = reader.get_graph()
    ob = SamplingPerformance(G)

    nn = int(G.number_of_nodes() * (args.percent_nodes if args.percent_nodes < 1 else args.percent_nodes / 100.0))
    ne = int(G.number_of_edges() * (args.percent_edges if args.percent_edges < 1 else args.percent_edges / 100.0))

    # What are the samplers.
    if not args.samplers:
        raise Exception(f"--samplers argument cannot be empty . Current value ({args.samplers} -- Expected one of [all, randomedge,randomwalk,...]")
    elif args.samplers == "all":
        expected_sampler_names = [
            "randomedge", "randomnode", "randomwalk", "pagerankbased",
            "metropolishastingsrandomwalk", "commonneighborawarerandomwalk",
            "randomnodeedge", "hybridnodeedge", "frontier"
        ]
    else:
        expected_sampler_names = [x.strip().lower() for x in args.samplers.split(",")] # made a list of the input samplers

    samplers = [sampler_factory(x, ne=ne, nn=nn) for x in expected_sampler_names]

    sampled_graph_metrics = [ob.measure(sampler=sampler) for sampler in samplers]
    metrics = [x[0] for x in sampled_graph_metrics]
    sampled_graphs = [x[1] for x in sampled_graph_metrics]

    # save the sampled graphs
    for sg, s_name in zip(sampled_graphs, expected_sampler_names):
        out_path = os.path.join(args.sampled_graph_dir, f"{s_name}.csv")
        with open(out_path, 'w') as outfile:
            print(f"Saving sampled graph ({s_name}) : {out_path}")
            sg_content = "\n".join([f"{sg_e[0]},{sg_e[1]}" for sg_e in sg.edges])

            outfile.write(sg_content)

    # measure perf
    gold_ref = ob.measure_orig()
    ob.sort_metrics_by_perf(gold_ref=gold_ref, metrics=metrics)
    metrics.append(gold_ref)

    with open(args.out_path, 'w') as outfile:
        outfile.write(f"sampler\tavg_degree\tdegree_corr\tcluster_coeff")
        for m in metrics:
            print(f"{m.as_tsv()}")
            outfile.write(f"{m.as_tsv()}\n")

    logging.info(f"\nOutput in {args.out_path}")

def main_perf_grid(base_path:str, graph_name:str, pc_node_min:int=5, pc_node_max:int=25, pc_node_step:int=5, pc_edge_min:int=5, pc_edge_max:int=25, pc_edge_step:int=5):
    reader = GraphReader(graph_name)
    reader.base_url = base_path
    G = reader.get_graph()
    ob = SamplingPerformance(G)

    super_winners_counter, config_wise_winners = SamplingWinner.grid_search(ob=ob, pc_node_min=pc_node_min,
                   pc_node_max=pc_node_max, pc_node_step=pc_node_step, pc_edge_min=pc_edge_min,
                   pc_edge_max=pc_edge_max, pc_edge_step=pc_edge_step)

    for c in config_wise_winners:
        print(c.as_tsv())

    topk = 5
    print(f"\n\n\n{'*' * 80}\n")
    for super_winner_name, num_wins in super_winners_counter[: topk]:
        print(f"SuperWinner: {super_winner_name}\t{num_wins}")

    return super_winners_counter, config_wise_winners


def main_perf_grid_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path",
                        default=f"file://{os.path.join(os.path.abspath(os.curdir), 'sampling_expts')}/data/",
                        required=False, type=str, help="Parent path to the directory where graph_name.csv resides")
    parser.add_argument("--graph_name", required=True, type=str,
                        help="Name of the graph e.g., Family (we will expect graphname_id_id_edges.csv.")
    # , pc_node_min:int, pc_node_max:int, pc_node_step:int, pc_edge_min:int, pc_edge_step:int, pc_edge_max:int
    parser.add_argument("--pc_node_min", required=False, default=5, type=int, help="Enter the minimum percent sampling value of the node e,g 5")
    parser.add_argument("--pc_node_max", required=False, default=25, type=int, help="Enter the maximum percent sampling value of the node e,g 25")
    parser.add_argument("--pc_node_step", required=False, default=5, type=int, help="Enter the step size of the node.e.g 5")
    parser.add_argument("--pc_edge_min", required=False, default=5, type=int, help="Enter the minimum percent sampling value of the edge e,g 5")
    parser.add_argument("--pc-edge_max", required=False, default=25, type=int, help="Enter the maximum percent sampling value of the edge e,g 25")
    parser.add_argument("--pc_edge_step", required=False, default=5, type=int, help="Enter the step size of the edge,g 5")
    args = parser.parse_args()
    main_perf_grid(base_path=args.base_path, graph_name=args.graph_name, pc_node_min=args.pc_node_min,
                   pc_node_max=args.pc_node_max, pc_node_step=args.pc_node_step, pc_edge_min=args.pc_edge_min,
                   pc_edge_max=args.pc_edge_max, pc_edge_step=args.pc_edge_step)


if __name__ == '__main__':
    # main_perf_grid()
    main_perf_at_config()