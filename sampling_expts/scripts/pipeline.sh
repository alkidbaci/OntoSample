#!/bin/bash

# sh sampling_expts/scripts/pipeline.sh Family
DATASET=${1:?"Missing dataset name e.g., Family"}
SAMPLERS=${2:?"all OR csv of samplers to try e.g., random_walk"}
PC_NODE=${3:?"what percentage of nodes to sample e.g. 10"}
PC_EDGE=${4:?"what percentage of edges to sample e.g. 10"}
DATASET_LOCATION=${5-"sampling_expts/data"}
SAMPLED_LOCATION=${6-"sampling_expts/sampled_data"}

set -e # print every command that is being executed.


D="${DATASET_LOCATION}/${DATASET}"

echo "Step 1/4: Preparing for sampling (creating node id)..."
python sampling_expts/src/node_ids.py --node_node_fp "${D}_edges.csv" --txt_id_mapping_fp "${D}_node_ids.csv" --id_node_mapping_fp "${D}_id_node_mapping.csv" --id_id_mapping_fp "${D}_id_id_edges.csv"

echo "STEP 2/4: Run a specific sampling (creating node id)..."
python sampling_expts/src/perf_results.py --graph_name "${DATASET}_id_id" --percent_nodes "${PC_NODE}" --percent_edges "${PC_EDGE}" --samplers "${SAMPLERS}" --sampled_graph_dir "${SAMPLED_LOCATION}/${DATASET}"

echo "STEP 3/4: Convert the sampled graph to nodetxt nodetxt format..."
python sampling_expts/src/onto_utils.py --sampled_graph_fp "${SAMPLED_LOCATION}/${DATASET}/${SAMPLERS}.csv" --id_node_mapping_fp "${D}_id_node_mapping.csv" --sampled_graph_str_fp "${SAMPLED_LOCATION}/${DATASET}/${SAMPLERS}_str.csv" --owl_fp "${D}_benchmark_rich_background.owl"

echo "STEP 4/4: Convert the sampled graph to OWL format... (TODO)"
