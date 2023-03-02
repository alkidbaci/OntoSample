"""Streamlit file to display the result of the grid performance
Usage:  streamlit run sampling_expts/demo/st_perf_results.py
Incase it shows Module not found error, use export PYTHONPATH=. and then run the usage
  """

import sys
sys.path.append("..")



import streamlit as st
import os

from sampling_expts.src import file_utils
from sampling_expts.src.perf_results import main_perf_grid

st.set_page_config(layout="wide")  # page, layout setup

def spaces(n: int, column_handler=None):
    for i in range(n):
        if column_handler:
            column_handler.write("")
        else:
            st.write("")


if __name__ == '__main__':
    # left, right = st.columns(2)
    st.title("Best sampler")

    with st.form("graph_config"):

        graph_name = st.text_input(label="graph name", value="Family_id_id")
        base_path = f"file://{os.path.join(os.path.abspath(os.curdir), 'sampling_expts')}/data/"
        idid_f = st.file_uploader("(Alternatively) Upload an ID(int) ID(int) file", type=["csv"])

        if st.form_submit_button(label="find best sampling"):

            if idid_f is not None:
                temp_path = file_utils.create_temp_file(suffix="_edges.csv")
                with open(temp_path, 'wb') as temp_out:
                    temp_out.write(idid_f.getbuffer())
                print(f"Created a tempfile at {temp_path}")
                idid_f.close()
                # Given      = /Users/rishig/tmp/abcd/xxxx.csv
                # base_url   = /Users/rishig/tmp/abcd/
                # graph_name = xxxx_edges
                base_path  = f"file://{os.path.dirname(temp_path)}"
                graph_name = os.path.basename(temp_path).replace("_edges.csv", "")
                print(f"Extracting base_path ({base_path}) and graph_name ({graph_name}) from uploader ")

            super_winners_counter, config_wise_winners = main_perf_grid(base_path=base_path, graph_name=graph_name)
            st.balloons()

            rows = [["pc_nodes", "pc_edges", "sampler_name", "avg_degree","cluster_coeff", "degree_c", "l2_norm"]]
            for c in config_wise_winners:
                rows.append( c.as_tsv().split("\t"))
            st.table(rows)
            spaces(1, st)



            topk = 5
            rows2 = [[f"Best Sampler after {len(config_wise_winners)} runs ", "num wins"]]
            for super_winner_name, num_wins in super_winners_counter[: topk]:
                rows2.append([super_winner_name, f"{num_wins}"])
            st.table(rows2)

