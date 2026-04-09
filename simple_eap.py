from tqdm import tqdm
from typing import Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from transformer_lens import HookedTransformer

from eap.graph import Graph
from eap.utils import make_hooks_and_matrices, tokenize_plus

#TODO:
# - get rid of graph or make one (it's just for number of forward and backward nodes)
# - possibly get rid of dataloader, as I'm interested in a single example. Alternative: dataloader with single datapoint.
# - modify gradient_hook (in eap) and shape of scores tensor
# - new intervention keyword for "self-defined corrupted activations"

def get_scores_eap_var(
    model: HookedTransformer, graph: Graph, dataloader:DataLoader, metric: Callable[[Tensor], Tensor], 
                   intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', 
                   #intervention_dataloader: Optional[DataLoader]=None,
                   quiet=False
):
    """Gets edge attribution scores using EAP. Variant of the function in EAP-IG.

    Args:
        model (HookedTransformer): The model to attribute
        graph (Graph): Graph to attribute
        dataloader (DataLoader): The data over which to attribute
        metric (Callable[[Tensor], Tensor]): metric to attribute with respect to
        quiet (bool, optional): suppress tqdm output. Defaults to False.

    Returns:
        Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
    """
    scores = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)    

    # if 'mean' in intervention:
    #     assert intervention_dataloader is not None, "Intervention dataloader must be provided for mean interventions"
    #     per_position = 'positional' in intervention
    #     means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position)
    #     means = means.unsqueeze(0)
    #     if not per_position:
    #         means = means.unsqueeze(0)


    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

        with torch.inference_mode():
            if intervention == 'patching':
                # We intervene by subtracting out clean and adding in corrupted activations
                with model.hooks(fwd_hooks_corrupted):
                    _ = model(corrupted_tokens, attention_mask=attention_mask)
            # elif 'mean' in intervention:
            #     # In the case of zero or mean ablation, we skip the adding in corrupted activations
            #     # but in mean ablations, we need to add the mean in
            #     activation_difference += means

            # For some metrics (e.g. accuracy or KL), we need the clean logits
            clean_logits = model(clean_tokens, attention_mask=attention_mask)

        with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
            logits = model(clean_tokens, attention_mask=attention_mask)
            metric_value = metric(logits, clean_logits, input_lengths, label)
            metric_value.backward()

    scores /= total_items

    return scores
