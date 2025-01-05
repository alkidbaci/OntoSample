# OntoSample

[![Downloads](https://static.pepy.tech/badge/ontosample)](https://pepy.tech/project/ontosample)
[![Downloads](https://img.shields.io/pypi/dm/ontosample)](https://pypi.org/project/ontosample/)
[![Pypi](https://img.shields.io/badge/pypi-0.2.6-blue)](https://pypi.org/project/ontosample/0.2.6/)

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
git clone https://github.com/alkidbaci/OntoSample.git
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