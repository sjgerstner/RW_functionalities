import torch
from torch.utils.data import Dataset, DataLoader

from transformer_lens import HookedTransformer

from eap import attribute
from eap.evaluate import evaluate_graph
from eap.graph import Graph
from eap.utils import get_logit_positions

#TODO:
# Modify Graph to allow for finding position-sensitive circuits.
# Don't change the structure of the graph, just
# the shape of graph.scores
# and the methods.

#model and graph
model = HookedTransformer.from_pretrained(
    'allenai/OLMo-7B-0424-hf',
    refactor_glu=True,#TODO do we need this here?
)
graph = Graph.from_model(model)
is_fancy_graph=False#TODO change once ready

# TODO dataloader with single datapoint.
def my_collate(sequences_and_labels):
    sequences, labels = zip(*sequences_and_labels)
    clean = list(sequences)
    corrupted = None
    return clean, corrupted, labels

dataset = Dataset()#TODO
dataloader = DataLoader(dataset=dataset, collate_fn=my_collate)

#metrics
def mic_score(logits:torch.Tensor, clean_logits:torch.Tensor, input_lengths:torch.Tensor, label:torch.Tensor|None=None)->torch.Tensor:
    logits = get_logit_positions(logits, input_lengths)
    return logits @ model.W_U[:,model.to_single_token('mic')]

def entropy(logits:torch.Tensor, clean_logits:torch.Tensor, input_lengths:torch.Tensor, label:torch.Tensor|None=None)->torch.Tensor:
    logits = get_logit_positions(logits, input_lengths)
    probs = torch.softmax(logits, dim=-1)
    return (probs*torch.log(probs)).sum(-1)

def loss(logits:torch.Tensor, clean_logits:torch.Tensor, input_lengths:torch.Tensor, label:torch.Tensor|None=None)->torch.Tensor:
    logits = get_logit_positions(logits, input_lengths)
    probs = torch.softmax(logits, dim=-1)
    return -torch.log(probs[...,label])

my_metric=mic_score #TODO also try with entropy and loss

if is_fancy_graph:
#attribute
#running this modifies graph.scores:
    attribute.attribute(
        model,
        graph,
        dataloader,
        metric=my_metric,
        method='EAP',
        # intervention='custom',
        # keep_pos_dim=True,
    )

    #circuit finding
    graph.apply_topn(n)#TODO define n, and/or apply another method

    #TODO evaluation: wich method do we want?
    results = evaluate_graph(model, graph, dataloader, metrics=my_metric)

else:#minimalist alternative without fancy graph
    scores = attribute.get_scores_eap(
        model,
        graph,
        metric=my_metric,
        dataloader=dataloader,
        intervention='custom',
        keep_pos_dims=True,
    )
    #TODO circuit finding, e.g. topn. Make sure we keep only real edges!
    #TODO evaluation: how similar are the results if we just ablate the relevant edges?
    # and if we just ablate the relevant weakening neurons?
