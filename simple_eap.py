import torch
from torch.utils.data import (
    #Dataset,
    DataLoader,
)

import datasets
from transformer_lens import HookedTransformer

from eap import attribute
from eap.evaluate import evaluate_graph
from eap.graph import Graph
from eap.utils import get_logit_positions#, tokenize_plus

from examples import create_args, find_neurons, list_ablation_hooks

#TODO:
# Modify Graph to allow for finding position-sensitive circuits.
# Don't change the structure of the graph, just
# the shape of graph.scores
# and the methods.

#model and graph
model = HookedTransformer.from_pretrained_no_processing(#TODO do we need preprocessing here?
    'allenai/OLMo-7B-0424-hf',
    #refactor_glu=True,
    device='cpu',#TODO comment out and run on beta
)
graph = Graph.from_model(model)
is_fancy_graph=False#TODO change once ready

def collate_fn(batch):
    sequences = [item["sequences"] for item in batch]
    labels = [item["labels"] for item in batch]
    return (sequences, sequences, labels)

sequence = "Yesterday (21 December) the Government announced a package of support for hospitality and leisure businesses that are losing trade because of the O"
dataset = datasets.Dataset.from_dict(mapping = {
    "sequences": [sequence],
    "labels": ["mic"]})
dataloader = DataLoader(dataset=dataset, collate_fn=collate_fn)

#metrics
def mic_score(logits:torch.Tensor, clean_logits:torch.Tensor, input_lengths:torch.Tensor, label:torch.Tensor|None=None)->torch.Tensor:
    logits = get_logit_positions(logits, input_lengths)
    return logits[...,model.to_single_token('mic')]

def entropy(logits:torch.Tensor, clean_logits:torch.Tensor, input_lengths:torch.Tensor, label:torch.Tensor|None=None)->torch.Tensor:
    logits = get_logit_positions(logits, input_lengths)
    probs = torch.softmax(logits, dim=-1)#TODO here and in loss: what is the right temperature?
    return (probs*torch.log(probs)).sum(-1)

def loss(logits:torch.Tensor, clean_logits:torch.Tensor, input_lengths:torch.Tensor, label:torch.Tensor|None=None)->torch.Tensor:
    logits = get_logit_positions(logits, input_lengths)
    probs = torch.softmax(logits, dim=-1)
    return -torch.log(probs[...,label])

my_metric=mic_score #TODO also try with entropy and loss

args = create_args(#note that this function is hard-coding the model name
    neuron_subset_name="weakening",
    intervention_type="zero_ablation",
    metric="entropy",
    topk=1,
    device=model.cfg.device,
    gate='-',
    post='+',
)
neuron_list = find_neurons(args)
corruption_hooks = list_ablation_hooks(args, neuron_list)

if is_fancy_graph:
#attribute
#running this modifies graph.scores:
    attribute.attribute(
        model,
        graph,
        dataloader,
        metric=my_metric,
        method='EAP',
        # corruption_hooks=corruption_hooks,
        # intervention='custom',
        # keep_pos_dim=True,
    )

    #circuit finding
    n=16
    graph.apply_topn(n)#TODO define n, and/or apply another method

    #TODO evaluation: wich method do we want?
    results = evaluate_graph(model, graph, dataloader, metrics=my_metric)

else:#minimalist alternative without fancy graph
    scores = attribute.get_scores_eap(
        model,
        graph,
        metric=my_metric,
        dataloader=dataloader,
        keep_pos_dims=True,
        corruption_hooks=corruption_hooks,
    )
    #print(torch.topk(scores, k=16))
    #TODO circuit finding, e.g. topn. Make sure we keep only real edges!
    #TODO evaluation: how similar are the results if we just ablate the relevant edges?
    # and if we just ablate the relevant weakening neurons?
