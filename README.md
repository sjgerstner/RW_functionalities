# Software and data for the RW functionalities paper (weakening neurons)

## Structure

* ```TransformerLens/``` (submodule): a fork of TransformerLens that supports OLMo.
* ```neuroscope/``` (submodule): Code and results for activation-based neuron case studies. Contains the dataset ```neuroscope/datasets/dolma-small/```.
* ```interactive.ipynb```: interactive vector visualisations of RW classes (may help to understand section 2.2 with the definitions of RW functionalities).
* Statistics of RW functionalities (section 3 in the submission):
  * Code is in the top-level directory:
    * ```utils.py``` and ```plotting.py``` contain helper functions
    * ```main.py```: Run this to reproduce the section. There are options for each sub-experiment (see the argparse part of the code).
  * Results are in ```results/```.
* Ablation experiments (section 4):
  * attributes rate:
    * main code in ```attributes/```
    * Wikipedia data for subject-attribute mappings: code in ```wiki/``` and data in ```wiki_data/```
    * data to run the model on: ```knowns/```
    * machine-readable results in ```se/logitlens.pickle```
    * plots in ```plots/ablations/``` (all the ```attributes.pdf``` files)
  * other metrics (entropy etc.), incl. conditional ablations:
    * bash script that runs all the modified models: ```entropy_interventions.sh```. It calls the code in ```entropy/```.
    * machine-readable results in ```intervention_results/```
    * plots in ```plots/ablations/``` (files named ```entropy.pdf``` etc.)
* Activation frequencies (section 5):
  * code in ```freqs.py```
  * plots in ```plots/freq/```

## Steps to reproduce

### Environment

First, create your environment and install requirements:

```[bash]
conda create -n wcos --file environment.yml
conda activate wcos
git submodule init --recursive
pip install -e TransformerLens
```

The last line will change some of the package versions
(no combination was perfectly compatible),
but the behaviour is fine anyway.

### Section 3 (RW functionalities by layer)

```[bash]
python main.py --refactor_glu
```

(see the argparse part of the code for more options)

### Section 4 (Ablation experiments)

#### Attributes rate

##### Preparation (may be skipped)

In this section we explain how to reproduce the data
that we then use for the ablation experiment on attributes rate.
You can skip this because we provide the resulting data.

Get the online datasets:

```[bash]
cd knowns
wget https://rome.baulab.info/data/dsets/known_1000.json
cd ../wiki_data
wget https://archive.org/download/enwiki-20211020/enwiki-20211020-pages-articles-multistream-index.txt.bz2
wget https://archive.org/download/enwiki-20211020/enwiki-20211020-pages-articles-multistream.xml.bz2
cd ..
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

#### Entropy etc

Including conditional ablations.

To run the interventions:

```[bash]
bash entropy_interventions.sh
```

The above command will run up to 5 different ablations in parallel on different GPUs.
If you don't have that many GPUs available (but more time),
you can call the python commands by hand one by one.
Each ablation run needs approximately 7-8 GPU hours,
and there are 23 different ablations to run.

To get the plots:

```[bash]
python -m entropy.compare --experiment_name 24 --neurons strengthening24 "conditional strengthening24" "proportional change24" "conditional weakening24" weakening24
python -m entropy.compare --experiment_name 243 --neurons "conditional strengthening243" "proportional change243" "conditional weakening243" weakening
python -m entropy.compare --experiment_name weakening_complete --neurons weakening weakening_gate+_post+ weakening_gate+_post- weakening_gate-_post+ weakening_gate-_post-
```

### Section 5 (Activation frequencies)

```[bash]
python freqs.py --log
```
