from argparse import ArgumentParser
from os.path import exists

import torch
from torch.utils.data import (
    #Dataset,
    DataLoader,
)

import datasets
from transformer_lens import HookedTransformer

from eap import attribute
from eap.evaluate import evaluate_graph#may be needed in the future
from eap.graph import Graph
from eap.utils import get_logit_positions#, tokenize_plus

from examples import create_args, find_neurons, list_ablation_hooks

#TODO everywhere this appears:
# - pt instead of json
# - more fine-grained naming (positional vs not, ...)
GRAPH_FILE = "../RW_functionalities_results/full_graph.json"

def collate_fn(batch):
    sequences = [item["sequences"] for item in batch]
    labels = [item["labels"] for item in batch]
    return (sequences, sequences, labels)

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

parser = ArgumentParser()
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--n_edges", default=32, type=int)
args = parser.parse_args()

if exists(GRAPH_FILE):
    graph = Graph.from_json(GRAPH_FILE)
else:
    #TODO:
    # Modify Graph to allow for finding position-sensitive circuits.
    # Don't change the structure of the graph, just
    # the shape of graph.scores
    # and the methods.

    #model and graph
    model = HookedTransformer.from_pretrained_no_processing(#we don't need processing because EAP is implementation invariant
        'allenai/OLMo-7B-0424-hf',
        #refactor_glu=True,
        device=args.device,
    )
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True

    graph = Graph.from_model(model)
    # is_fancy_graph=True

    sequence = "Yesterday (21 December) the Government announced a package of support for hospitality and leisure businesses that are losing trade because of the O"
    dataset = datasets.Dataset.from_dict(mapping = {
        "sequences": [sequence],
        "labels": ["mic"]})
    dataloader = DataLoader(dataset=dataset, collate_fn=collate_fn)

    my_metric=mic_score #TODO also try with entropy and loss (and put minus in front)

    args = create_args(#NOTE that this function is hard-coding the model name
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

    #if is_fancy_graph:
    #attribute
    #running this modifies graph.scores:
    attribute.attribute(
        model,
        graph,
        dataloader,
        metric=my_metric,
        method='EAP',
        corruption_hooks=corruption_hooks,
        # keep_pos_dim=True,
    )

    graph.to_json(GRAPH_FILE)

#circuit finding
graph.apply_greedy(
    n_edges=args.n_edges,
    absolute=False,
    prune=False,
    #prune_parentless=False,
)

#image
graph.to_image("../RW_functionalities_results/graph.png")

#TODO evaluation: how similar are the results if we just ablate the relevant edges?
# and if we just ablate the relevant weakening neurons?
#results = evaluate_graph(model, graph, dataloader, metrics=my_metric)

# else:#minimalist alternative without fancy graph
#     scores = attribute.get_scores_eap(
#         model,
#         graph,
#         metric=my_metric,
#         dataloader=dataloader,
#         keep_pos_dims=True,
#         corruption_hooks=corruption_hooks,
#     )
#     #print(torch.topk(scores, k=16))
