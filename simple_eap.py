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
parser.add_argument("--positional", action='store_true')
parser.add_argument("--graph_method", choices=["topn", "threshold", "greedy"], default="greedy")
parser.add_argument("--force_recompute", action='store_true')
parser.add_argument("--test", action='store_true')
args = parser.parse_args()

POSITIONAL = args.positional or args.test
GRAPH_FILE = f"../RW_functionalities_results/full{"_positional" if POSITIONAL else ""}_graph.pt"

if exists(GRAPH_FILE) and not args.force_recompute:
    print("loading previously computed graph")
    graph = Graph.from_pt(GRAPH_FILE)
else:
    #model and graph
    #we don't need processing because EAP is implementation invariant
    model = HookedTransformer.from_pretrained_no_processing(
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

    my_args = create_args(#NOTE: this function is hard-coding the model name
        neuron_subset_name="weakening",
        intervention_type="zero_ablation",
        metric="entropy",
        topk=1,
        device=model.cfg.device,
        gate='-',
        post='+',
    )
    neuron_list = find_neurons(my_args)
    corruption_hooks = list_ablation_hooks(my_args, neuron_list)

    print("computing scores...")
    print("metric:", my_metric)
    print("lower is better:", my_metric is not mic_score)
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
        keep_pos_dims=POSITIONAL,
        lower_is_better= my_metric is not mic_score,
    )

    graph.to_pt(GRAPH_FILE)

if POSITIONAL:
    print(graph.positional_scores[:,-1,-1])
else:
    print(graph.scores[-1,-1])

print("circuit finding...")
if args.test:
    graph.apply_threshold(
        0.1,
        absolute=False,
        prune=False,
        positional=True,
    )
    graph.to_image(
        f"../RW_functionalities_results/threshold_graph.png",
        positional=True,
    )
    graph.prune(prune_parentless=False, positional=True)
elif args.graph_method=="greedy":
    graph.apply_greedy(
        n_edges=args.n_edges,
        absolute=False,
        prune=False,
        positional=POSITIONAL,
    )
elif args.graph_method=="topn":
    graph.apply_topn(
        n=args.n_edges,
        absolute=False,
        prune=True,
        prune_parentless=False,
        positional=POSITIONAL,
    )
else:
    raise NotImplementedError

print("making image...")
if args.test:
    graph.to_image(
        f"../RW_functionalities_results/threshold_graph_pruned.png",
        positional=True,
    )
else:
    graph.to_image(
        f"../RW_functionalities_results/{"positional_" if POSITIONAL else ""}graph.png",
        positional=POSITIONAL,
    )

#TODO evaluation:
# - does patching corrupted activations into any of the edges reduce mic score?
#   - or any combination of the edges?
# - how about all the final edges together?
# - how about all the initial edges together?
# - how about patching clean activations into the corrupted run: does it restore mic score?

#baseline = evaluate_baseline(model, dataloader, metrics=my_metric) -> we already did that
#results = evaluate_graph(model, graph, dataloader, metrics=my_metric)

# TODO which are the relevant weakening neurons, and what if we ablate just them?
