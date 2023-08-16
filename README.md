# OntoSample
We have implemented our code inside the [OntoLearn](https://github.com/dice-group/Ontolearn/tree/develop) library 
and our work can be found inside the folder named Sampling.

The LPC samplers can be found inside `Sampling/Samplers/LPCentralized_Samplers`

> **Note**: Not all datasets are included in the project because some of them are too large.
> You can download all the SML-bench datasets [here](https://github.com/SmartDataAnalytics/SML-Bench/tree/updates/learningtasks).
> They need to go to their respective folder named after them inside KGs directory.

## Reproducing the results

The main results are given by Figure 1 and Table2. Below are given the 
instructions to reproduce the results. The following script generates every
single result for a dataset-sampler-sampling size combination and in the end
of each combination inside the csv, it adds the values that are then put in Table 2 and Figure 1.

### To generate results of Table 2

The evaluation results for a certain sampling percentage can be simply reproduced by using `Sampling/evaluation_table_generator.py`.

There are the following arguments that the user can give:
- `learner` &rarr; type of learner: 'evolerner' or 'celeo'.
- `datasets` &rarr; list containing the name of the json files that contains the path to the knowledge graph and
                           the learning problem.
- `samplers` &rarr; list of the abbreviation of the samplers as strings.
- `csv_path` &rarr; path of the csv file to save the results.
- `sampling_size` &rarr; the sampling percentage
- `iterations` &rarr; number of iterations for each sampler

Table 2 results can be  generated using the following instructions:

1. Execute the script `evaluation_table_generator.py` using the default parameters.
2. After the script has finished executing, set the argument `learner` to `celoe`
3. Set the csv path to another path by using the `csv_path` argument.
4. Execute again.

In the end you will have 2 csv files, one for each learner.

> **Note**: Keep in mind that this file needs a considerable amount of time to execute (more than 40 hours for each concept learner
> depending on the machine specifications) when using the default values which were also used to construct 
> the results for the paper. 
> 
> If you want quicker execution, you can enter a lower number of iterations.

---------------------------------------------------

### To generate results of Figure 1

To generate results used in Figure 1 you need to follow the instructions below
when writing the command to execute the script `Sampling/evaluation_table_generator.py`:

1. Set the `datasets` argument to `{"hepatitis_lp.json", "carcinogenesis_lp.json"}`.
2. Set the `samplers` argument to `{"RNLPC", "RWJLPC", "RWJPLPC", "RELPC", "FFLPC"}`.
3. Set the `sampling_size` argument to `0.25`.
4. Execute the script.
5. Repeat the execution for sampling sizes of `0.20`, `0.15`, `0.10`, `0.5`

> **Note:** Make sure to set a different csv path using the `csv_path` argument each time you execute to avoid
> overriding the previous results.