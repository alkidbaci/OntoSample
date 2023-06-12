We have implemented our code in the [Ontolearn](https://github.com/dice-group/Ontolearn/tree/develop) library 
and our work can be found inside the folder named Sampling.

The LPC samplers can be found inside `Sampling/Samplers/LPCentralized_Samplers`

> **Note**: Not all datasets are included here because some of them are to large. To download them you can use the 
> .link files or find it directly at its source in case of Premier League.

## Reproducing the results

There are 3 methods to generate the result starting with the most general.

1.  The evaluation results for a certain sampling percentage can be simply reproduced by running `Sampling/evaluation_table_generator.py`.
    There are a few adjustable variables as following:
    - `datasets_path` -> list containing the name of the json files that contains the path to the knowledge graph and
                               the learning problem.
    - `samplers` -> list of the abbreviation of the samplers as strings.
    - `sampling_percentage` -> the sampling percentage
    - `x` -> number of iterations for each sampler ( default value 100 )

    > **Note**: Keep in mind that this file needs a considerable time to execute ( more than 20 hours depending on the machine specifications).


2.  To perform the evaluation only for a single sampler in multiple samples, run `Sampling/sampling_multiple_run_performance_display.py`.
    There are a few adjustable variables as following:
    - name of the json file (enter directly in line 30)
    - `sampler` -> the sampler object
    - `x` -> number of nodes to sample

3.  To evaluate a single sampler only in a single sample, run `Sampling/sampling_single_run_performance_display.py`.
    There are a few adjustable variables as following:
    - name of the json file (enter directly in line 29)
    - `sampler` -> the sampler object
