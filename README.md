# Software for the RW functionalities paper (weakening neurons)

This code is intended to help you reproduce the results and plots from the Weakening Neurons paper.

We plan to publish data / results separately.

## Structure

* ```TransformerLens/``` (submodule): a fork of TransformerLens that supports OLMo.
* ```neuroscope/``` (submodule): Code and results for activation-based neuron case studies. Contains the dataset ```neuroscope/datasets/dolma-small/```.
* `src/weight_analysis_utils` (installable package): utilities for weight analysis, incl. plotting.
* ```interactive.ipynb```: interactive vector visualisations of RW classes. This may help to understand the definitions of the different RW classes in section 4.2 (of the ARR submission).
* Statistics of RW functionalities (section 5 in the ARR submission): Run ```main.py``` to reproduce the section. There are options for each sub-experiment (see the argparse part of the code).
* Ablation experiments (section 6):
  * attributes rate:
    * main code in ```attributes/```
    * ```wiki/```: code for producing the Wikipedia data used for subject-attribute mappings
  * other metrics (entropy etc.), incl. conditional ablations: The bash script ```entropy_interventions_all.sh``` runs all the edited models. It calls the code in ```entropy/```.
  * `*.ipynb` and `simple_eap.py` and the submodule ```EAP-IG/```: Analysis of a case of entropy reduction (section 6.3). Was not cleaned up for publication, results are only partially reported in the paper.
* Activation frequencies (section 7): code in ```freqs.py```.

The other files contain other small experiments that we did not include in the paper, e.g. model generations when ablating a given class of neurons.

## Steps to reproduce

### Environment

First, create your environment and install requirements:

```[bash]
conda create -n wcos --file environment.yml
conda activate wcos
git submodule init --recursive
pip install -e TransformerLens
```

You can ignore the version conflicts.

### Section 5 (IO functionalities by layer)

```[bash]
python main.py --refactor_glu #for the GLU-based models
python main.py --models vanilla #for the non-GLU models
```

(see the argparse part of the code for more options)

### Section 6 (Ablation experiments)
#### Entropy etc

Including conditional ablations.

To run the interventions:

```[bash]
export CUDA_VISIBLE_DEVICES=4,5,6,7 #or whatever
bash entropy_interventions_all.sh
```

To get the plots, run ```python -m entropy.compare``` with the options:
* ```--experiment_name``` to define how your plot will be named
* ```--neurons``` to define which runs (i.e. which neuron sets) you want to include in the plots.
For example, for the plot in the main paper:

```[bash]
python -m entropy.compare --experiment_name weakening_complete --neurons weakening weakening_gate+_post+ weakening_gate+_post- weakening_gate-_post+ weakening_gate-_post-
```

Finally, for the case study of entropy reduction, see the ipynb files (they are a bit chaotic, sorry).

#### Attributes rate (appendix only)

##### Preparation (may be skipped)

In this section we explain how to reproduce the data
that we then use for the ablation experiment on attributes rate.
You can skip this because we provide the resulting data.

Get the online datasets:

```[bash]
cd ../RW_functionalities_results/knowns
wget https://rome.baulab.info/data/dsets/known_1000.json
cd ../wiki_data
wget https://archive.org/download/enwiki-20211020/enwiki-20211020-pages-articles-multistream-index.txt.bz2
wget https://archive.org/download/enwiki-20211020/enwiki-20211020-pages-articles-multistream.xml.bz2
cd ../../RW_functionalities #back to the code repo
```

Filter ```known_1000``` to the items known by OLMo-7B-0424 (the model of interest):

```[bash]
python wiki/knowns.py
```

Create a dataset of relevant Wikipedia paragraphs for each subject:

```[bash]
python wiki/wiki_preprocess.py
python wiki/wiki_retrieve.py
```

##### Main experiment

```[bash]
python -m attributes.enrichment --n_neurons 243 # we also did the same with 24
python -m attributes.plotting
```


### Section 7 (Activation frequencies)

```[bash]
python freqs.py --log
```
