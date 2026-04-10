import torch

from transformer_lens import HookedTransformer

from eap import attribute
from eap.evaluate import evaluate_graph
from eap.graph import Graph

#TODO:
# Modify Graph to allow for finding position-sensitive circuits.
# Don't change the structure of the graph, just
# the shape of graph.scores
# and the methods.

# TODO dataloader with single datapoint.
dataloader = ...

#TODO metric
def my_metric(x:torch.Tensor)->torch.Tensor:
    return x.sum()

#model and graph
model = HookedTransformer.from_pretrained(
    'allenai/OLMo-7B-0424-hf',
    refactor_glu=True,#TODO do we need this here?
)
graph = Graph.from_model(model)
is_fancy_graph=False#TODO change once ready

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

else:#minimalist alternative:
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
