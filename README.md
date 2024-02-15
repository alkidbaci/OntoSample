# OntoSample

OntoSample is a python package that offers classic sampling techniques for OWL ontologies/knowledge 
bases. Furthermore, we have tailored the classic sampling techniques to the setting of concept 
learning making use of learning problem.


Paper: [Accelerating Concept Learning via Sampling](https://doi.org/10.1145/3583780.3615158)

## Installation

```shell
pip install ontosample
```

or

```shell
# 1. clone 
git clone https://github.com/dice-group/Ontolearn.git 
# 2. setup virtual environment
python -m venv venv 
# 3. activate the virtual environment
source venv/bin/activate # for Unix and macOS
.\venv\Scripts\activate  # for Windows
# 4. install dependencies
pip install -r requirements.txt
```

## Usage

```python
from ontolearn_light.knowledge_base import KnowledgeBase
from ontosample.classic_samplers import RandomNodeSampler

# 1. Initialize KnowledgeBase object using the path of the ontology
kb = KnowledgeBase(path="KGs/Family/family-benchmark_rich_background.owl")

# 2. Initialize the sampler and generate the sample
sampler = RandomNodeSampler(kb)
sampled_kb = sampler.sample(30)  # will generate a sample with 30 nodes

# 3. Save the sampled ontology
sampler.save_sample(kb=sampled_kb, filename='sampled_kb')

```

Check the [examples](https://github.com/alkidbaci/OntoSample/tree/main/examples) folder for more.


## About the paper

### Abstact

Node classification is an important task in many fields, e.g., predicting entity types in knowledge graphs, classifying papers in citation
graphs, or classifying nodes in social networks. In many cases, it
is crucial to explain why certain predictions are made. Towards
this end, concept learning has been proposed as a means of interpretable node classification: given positive and negative examples
in a knowledge base, concepts in description logics are learned that
serve as classification models. However, state-of-the-art concept
learners, including EvoLearner and CELOE exhibit long runtimes.
In this paper, we propose to accelerate concept learning with graph
sampling techniques. We experiment with seven techniques and tailor them to the setting of concept learning. In our experiments, we
achieve a reduction in training size by over 90% while maintaining
a high predictive performance.

### Reproducing paper results

You will find in examples folder the script used to generate the results in paper.
`evaluation_table_generator.py` generates every result for each dataset-sampler-sampling_size 
combination and store them in a csv.

#### To generate results of Table 2
Install the whole ontolearn package to use its learning algorithms like EvoLearner and CELOE because 
they are not included here to keep the number of dependencies low.

```shell
pip install ontolearn
```

The evaluation results for a certain sampling percentage can be simply reproduced by using `examples/evaluation_table_generator.py`.

There are the following arguments that the user can give:
- `learner` &rarr; type of learner: 'evolerner' or 'celeo'.
- `datasets_and_lp` &rarr; list containing the name of the json files that contains the path to the knowledge graph and
                           the learning problem.
- `samplers` &rarr; list of the abbreviation of the samplers as strings.
- `csv_path` &rarr; path of the csv file to save the results.
- `sampling_size` &rarr; the sampling percentage
- `iterations` &rarr; number of iterations for each sampler

Table 2 results can be  generated using the following instructions:

1. Execute the script `evaluation_table_generator.py` using the default parameters.
2. After the script has finished executing, set the argument `--learner` to `celoe`
3. Set the csv path to another path by using the `--csv_path` argument.
4. Execute again.

In the end you will have 2 csv files, one for each learner.

> **Note 1**: Not all datasets are included in the project because some of them are too large.
> You can download all the SML-bench datasets [here](https://github.com/SmartDataAnalytics/SML-Bench/tree/updates/learningtasks).
> They need to go to their respective folder named after them inside KGs directory.

> **Note 2**: Keep in mind that this file needs a considerable amount of time to execute (more than 40 hours for each concept learner
> depending on the machine specifications) when using the default values which were also used to construct 
> the results for the paper. 
> 
> If you want quicker execution, you can enter a lower number of iterations.

---------------------------------------------------

#### To generate results of Figure 1

To generate results used in Figure 1 you need to follow the instructions below
when writing the command to execute the script `examples/evaluation_table_generator.py`:


```shell
cd examples
python evaluation_table_generator.py --datasets_and_lp {"hepatitis_lp.json", "carcinogenesis_lp.json"} --samplers {"RNLPC", "RWJLPC", "RWJPLPC", "RELPC", "FFLPC"} --sampling_size 0.25
```

Repeat the command for sampling sizes of `0.20`, `0.15`, `0.10`, `0.5`


> **Note:** Make sure to set a different csv path using the `--csv_path` argument each time you execute to avoid
> overriding the previous results.


### Citing

```
@inproceedings{10.1145/3583780.3615158,
author = {Baci, Alkid and Heindorf, Stefan},
title = {Accelerating Concept Learning via Sampling},
year = {2023},
isbn = {9798400701245},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3583780.3615158},
doi = {10.1145/3583780.3615158},
abstract = {Node classification is an important task in many fields, e.g., predicting entity types in knowledge graphs, classifying papers in citation graphs, or classifying nodes in social networks. In many cases, it is crucial to explain why certain predictions are made. Towards this end, concept learning has been proposed as a means of interpretable node classification: given positive and negative examples in a knowledge base, concepts in description logics are learned that serve as classification models. However, state-of-the-art concept learners, including EvoLearner and CELOE exhibit long runtimes. In this paper, we propose to accelerate concept learning with graph sampling techniques. We experiment with seven techniques and tailor them to the setting of concept learning. In our experiments, we achieve a reduction in training size by over 90\% while maintaining a high predictive performance.},
booktitle = {Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
pages = {3733–3737},
numpages = {5},
keywords = {knowledge bases, concept learning, graph sampling},
location = {Birmingham, United Kingdom},
series = {CIKM '23}
}
```

In case of any question please feel free to open an issue.