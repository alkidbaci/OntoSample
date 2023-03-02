setup:
export PYTHONPATH=.
pip install -r sampling_expts/requirements.txt

To run demo:
streamlit run sampling_expts/demo/st_perf_results.py

To run tests:
python -m pytest sampling_expts/tests