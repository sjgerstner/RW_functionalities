"""Utilities for the RW functionality paper,
especially for classification"""

from math import ceil, floor
from tqdm import tqdm

import pandas as pd
from scipy.stats import beta
import torch
from torch import Tensor
from torch.linalg import vector_norm
import einops

STRENGTHENING = "strengthening"
WEAKENING = "weakening"

COMBO_TO_NAME = {
    #keys are triples each of which represents, in order:
    #. approx(cos(w_in,w_out)) (1 or 0 or -1),
    #. |approx(cos(w_gate,w_out))| (1 or 0),
    #. approx(cos(w_gate,w_in))==approx(cos(w_gate,out))*approx(cos(w_in,w_out)) (1 or 0),
    #   but 1 for orthogonal output.
    (1,1,1): STRENGTHENING,
    (1,1,0): f"atypical {STRENGTHENING}",
    (1,0,1): f"conditional {STRENGTHENING}",
    (1,0,0): f"atypical conditional {STRENGTHENING}",
    (0,1,1): "proportional change",
    (0,1,0): "atypical proportional change",
    (0,0,1): "orthogonal output",
    (-1,1,1): WEAKENING,
    (-1,1,0): f"atypical {WEAKENING}",
    (-1,0,1): f"conditional {WEAKENING}",
    (-1,0,0): f"atypical conditional {WEAKENING}",
}
NAME_TO_COMBO = {v:k for k,v in COMBO_TO_NAME.items()}#TODO equivalent for vanilla

VANILLA_CATEGORIES = {
    1: STRENGTHENING,
    0: "orthogonal output",
    -1: WEAKENING,
}

def pretty_print(mydict):
    for key, value in mydict.items():
        assert key in COMBO_TO_NAME
        print(f'{COMBO_TO_NAME[key]}: {value}')

#the following function is largely copied from
# https://github.com/pytorch/pytorch/issues/157431#issuecomment-3026856373
def torch_quantile(  # noqa: PLR0913 (too many arguments)
    tensor: Tensor,
    q: float | Tensor,
    dim: int | None = None,
    *,
    keepdim: bool = False,
    interpolation: str = "linear",
    out: Tensor | None = None,
) -> Tensor:
    r"""Improved ``torch.quantile`` for one scalar quantile.

    Arguments
    ---------
    tensor: ``Tensor``
        See ``torch.quantile``.
    q: ``float``
        See ``torch.quantile``. Supports only scalar values currently.
    dim: ``int``, optional
        See ``torch.quantile``.
    keepdim: ``bool``
        See ``torch.quantile``. Supports only ``False`` currently.
        Defaults to ``False``.
    interpolation: ``{"linear", "lower", "higher", "midpoint", "nearest"}``
        See ``torch.quantile``. Defaults to ``"linear"``.
    out: ``Tensor``, optional
        See ``torch.quantile``. Currently not supported.

    Notes
    -----
    Uses ``torch.kthvalue``. Better than ``torch.quantile`` since:

    #. it has no :math:`2^{24}` tensor `size limit
    #   <https://github.com/pytorch/pytorch/issues/64947#issuecomment-2304371451>`_;
    #. it is much faster, at least on big tensor sizes.

    """
    # Sanitization of: q
    q_float = float(q)  # May raise an (unpredictible) error
    if not 0 <= q_float <= 1:
        msg = f"Only values 0<=q<=1 are supported (got {q_float!r})"
        raise ValueError(msg)

    # Sanitization of: dim
    # Because one cannot pass  `dim=None` to `squeeze()` or `kthvalue()`
    if dim_was_none := dim is None:
        dim = 0
        tensor = tensor.reshape((-1, *(1,) * (tensor.ndim - 1)))

    # Sanitization of: inteporlation
    idx_float = q_float * (tensor.shape[dim] - 1)
    if interpolation == "nearest":
        idxs = [round(idx_float)]
    elif interpolation == "lower":
        idxs = [floor(idx_float)]
    elif interpolation == "higher":
        idxs = [ceil(idx_float)]
    elif interpolation in {"linear", "midpoint"}:
        low = floor(idx_float)
        idxs = [low] if idx_float == low else [low, low + 1]
        weight = idx_float - low if interpolation == "linear" else 0.5
    else:
        msg = (
            "Currently supported interpolations are {'linear', 'lower', 'higher', "
            f"'midpoint', 'nearest'}} (got {interpolation!r})"
        )
        raise ValueError(msg)

    # Sanitization of: out
    if out is not None:
        msg = f"Only None value is currently supported for out (got {out!r})"
        raise ValueError(msg)

    # Logic
    outs = [torch.kthvalue(tensor, idx + 1, dim, keepdim=True)[0] for idx in idxs]
    out = outs[0] if len(outs) == 1 else outs[0].lerp(outs[1], weight)

    # Rectification of: keepdim
    if keepdim:
        return out
    return out.squeeze() if dim_was_none else out.squeeze(dim)

def topk_df(vec, model, emb=None, k=64, nonneg=True):
    #TODO refactor: shouldn't need whole model
    """
    model used for to_string fct and emb
    """
    if emb is None:
        emb=model.W_U.detach()
    logits = torch.matmul(vec, emb)
    if not nonneg:
        logits_abs = torch.abs(logits)
        _, indices = torch.topk(logits_abs, k=k)
        logits = logits.cpu()
        values = [logits[i] for i in indices]
    else:
        values, indices = torch.topk(logits, k=k)
        values=values.cpu()
    str_tokens = [model.to_string(i) for i in indices]
    df = pd.DataFrame(values, index=str_tokens, columns=["dot product"])
    return df

def neuron_analysis(model, layer, neuron, emb=None, k=64, verbose=True):
    """
    emb: if None (default), unembedding matrix of model.
    Otherwise set explicit matrix of shape d_model, d_vocab.
    """
    out = model.W_out[layer,neuron,:].detach()
    lin = model.W_in[layer,:,neuron].detach()
    linout = cos(lin, out).item()
    out_pos = topk_df(out, model, emb=emb, k=k)
    out_neg = topk_df(-out, model, emb=emb, k=k)
    lin_pos = topk_df(lin, model, emb=emb, k=k)
    lin_neg = topk_df(-lin, model, emb=emb, k=k)
    if hasattr(model, "W_gate") and model.W_gate is not None:
        gate = model.W_gate[layer,:,neuron].detach()
        gatelin = cos(gate, lin).item()
        gateout = cos(gate, out).item()
        gate_pos = topk_df(gate, model, emb=emb, k=k)
        gate_neg = topk_df(-gate, model, emb=emb, k=k)

    if verbose:
        if hasattr(model, "W_gate") and model.W_gate is not None:
            print("gate vs. linear similarity:", gatelin)
            print("gate vs. out similarity:", gateout)
        print("lin vs. out similarity:", linout)
        print("================================")
        print("most similar tokens for w_out:")
        print(out_pos)
        print("================================")
        print("most similar tokens for -w_out:")
        print(out_neg)
        print("================================")
        print("most similar tokens for w_in:")
        print(lin_pos)
        print("================================")
        print("most similar tokens for -w_in:")
        print(lin_neg)
        if hasattr(model, "W_gate") and model.W_gate is not None:
            print("================================")
            print("most similar tokens for w_gate:")
            print(gate_pos)
            print("================================")
            print("most similar tokens for -w_gate:")
            print(gate_neg)

    if hasattr(model, "W_gate") and model.W_gate is not None:
        return gatelin, gateout, linout, out_pos, out_neg, lin_pos, lin_neg, gate_pos, gate_neg
    else:
        return linout, out_pos, out_neg, lin_pos, lin_neg

def cos(v1,v2, pattern='... d, ... d -> ...'):
    "batched cosine similarities"
    v1 /= vector_norm(v1, dim=-1, keepdim=True)
    v2 /= vector_norm(v2, dim=-1, keepdim=True)
    dot = einops.einsum(v1, v2, pattern)
    return dot

def randomness_region(v1, v2, p=0.05, absolute=False):
    """randomness region based on mismatched cosines, used in randomness_regions()"""
    l,n,_d = v1.shape
    if (l==1) or (n<8200):#TODO don't hardcode, this is a heuristic to avoid subsequent OOM errors
        mismatched_cos = cos(v1, v2, '... n1 d, ... n2 d -> ... n1 n2')
    else:#compute separately layer by layer
        quantiles = []
        for layer in tqdm(range(l)):
            quantiles.append(randomness_region(
                v1[layer:layer+1,:], v2[layer:layer+1,:], p=p, absolute=absolute
            )#1 n_quantiles?
            #the weird slicing is for keeping the dimension (as if it were a 1-layer model)
            )
        return torch.cat(quantiles, dim=0)#n_layers n_quantiles
    mismatched_cos = einops.rearrange(mismatched_cos, '... n1 n2 -> ... (n1 n2)')
    #get rid of matched entries
    mask = torch.ones(mismatched_cos.shape[-1], dtype=bool)
    delete_list = [i*n+i for i in range(n)]
    for k in delete_list:
        mask[k]=0
    try:
        mismatched_cos = mismatched_cos[...,mask]
        return _randomness_region(mismatched_cos, p=p, absolute=absolute)
    except torch.cuda.OutOfMemoryError:
        #layer by layer: compute quantile and delete data
        quantiles = []
        for _layer in tqdm(range(v1.shape[0])):
            quantiles.append(
                _randomness_region(mismatched_cos[0,mask], p=p, absolute=absolute)#n_quantiles
            )
            mismatched_cos = mismatched_cos[1:]
        return torch.stack(quantiles, dim=0)#n_layers n_quantiles
        # layer = v1.shape[0]//2
        # batch1 = mismatched_cos[:layer,mask]
        # first_quantiles = _randomness_region(batch1, p=p, absolute=absolute)
        # mismatched_cos = mismatched_cos[layer:,mask]
        # second_quantiles = _randomness_region(mismatched_cos, p=p, absolute=absolute)
        # return torch.stack((first_quantiles,second_quantiles), dim=1)#n_quantiles n_layers

def _randomness_region(mismatched_cos, p=0.05, absolute=False):
    if absolute:
        mismatched_cos = torch.abs(mismatched_cos)
        high_quantile = torch_quantile(mismatched_cos, q=1-p, dim=-1)
        #return torch.unsqueeze(high_quantile, dim=1)#l 1
        return high_quantile #l
    low_quantile = torch_quantile(mismatched_cos, q=p/2, dim=-1)#l
    high_quantile = torch_quantile(mismatched_cos, q=1-(p/2), dim=-1)#l
    return torch.stack((low_quantile, high_quantile), dim=-1)#l 2

def randomness_regions(mlp_weights, p=0.05):
    """Returns a symmetrical randomness region for weight cosines
    specific to the given MLP weights,
    based on cosine similarities of mismatched weight vectors

    Args:
        mlp_weights (dict):
            Keys are "W_gate", "W_in", "W_out",
            values are the corresponding weight matrices (all in the same format)
        p (float, optional): Significance level (between 0 and 1). Defaults to 0.05.

    Returns:
        dict with
            keys "gatelin", "gateout", "linout"
            and values tensors of shape (n_layers, 2),
                where the 2 represents low and high quantiles (floats)
    """
    assert 0<=p<=1
    answer = {
        "linout": randomness_region(mlp_weights["W_in"], mlp_weights["W_out"], p),
    }
    if "W_gate" in mlp_weights:
        answer["gatelin"] = randomness_region(
            mlp_weights["W_gate"], mlp_weights["W_in"], p, absolute=True
        )
        answer["gateout"] = randomness_region(
            mlp_weights["W_gate"], mlp_weights["W_out"], p
        )
    return answer

def beta_randomness_region(d, p=0.05):
    """Returns a symmetrical randomness region
    for cosine similarities of d-dimensional i.i.d Gaussian vectors,
    for the given significance level p.

    Args:
        d (int): dimension (i.e., d_model)
        p (float, optional): Significance level (between 0 and 1). Defaults to 0.05.

    Returns:
        float: low quantile
        float: high quantile
    """
    assert 0<=p<=1
    rv = beta(a=(d-1)/2, b=(d-1)/2, loc=-1., scale=2.)
    low_quantile = rv.ppf(q=p/2).item()
    high_quantile = rv.ppf(q=1-p/2).item()
    #absolute_quantile = rv.ppf(q=1-p)
    return low_quantile, high_quantile#, absolute_quantile

def _approx(x, threshold=.5):
    ans = torch.where(x>threshold, 1,0)
    ans = torch.where(x<-threshold, -1, ans)
    if isinstance(x, float):
        return ans.item()
    return ans

def compute_category(data, threshold=.5, device=None):
    """Computes the category of the given neurons,
    in the triple format of COMBO_TO_NAME.
    If args are floats: returns a triple
    If args are tensors (of shape (layer,neuron)): Returns tensor of shape (layer, neuron, 3)
    """
    if device:
        moved_data = {
            key:
            (value.to(device) if isinstance(value, torch.Tensor) else value)
            for key,value in data.items()
        }
        data = moved_data
    approx_linout = _approx(data["linout"], threshold)
    if "gateout" in data:
        approx_gateout = _approx(data["gateout"], threshold)
        typical = torch.where(
            _approx(data["gatelin"], threshold)==approx_linout*approx_gateout, 1, 0
        )
        typical = torch.where(
            (approx_linout==0) & (approx_gateout==0), 1, typical
        )
        answer = (approx_linout, torch.abs(approx_gateout), typical)
        if isinstance(data["linout"], torch.Tensor):
            return torch.stack(answer, dim=-1)
        return answer
    return approx_linout

#the functions count_categories and category_lists are not used anymore
# def count_categories(indices, gatelin, gateout, linout, threshold=.5):
#     """
#     Use this only for a small list of indices.
#     To count categories on the whole model, use count_categories_all().
#     If you have already done categories(),
#     read everything off from the output tensor of that function.
#     """
#     answer = {key:0 for key in COMBO_TO_NAME}

#     for l,n in indices:
#         category = compute_category(linout[l][n], gateout[l][n], gatelin[l][n], threshold)
#         answer[category] +=1
#     return answer

# def category_lists(indices, gatelin, gateout, linout, threshold=.5):
#     """
#     Use this only for a small list of indices.
#     To list all categories, use categories().
#     If you have already done that,
#     you can just read everything off from the output tensor of categories().
#     """
#     answer = {key:[] for key in COMBO_TO_NAME}
#     for l,n in indices:
#         category = compute_category(linout[l][n], gateout[l][n], gatelin[l][n], threshold)
#         answer[category].append((l,n))
#     return answer

def is_in_category(category_tensor, category_key):
    """Return a tensor of booleans indicating for each neuron if it belongs to the given category

    Args:
        category_tensor (tensor): output of compute_category,
            shape (layer, neuron, 3) for GLU variants, or (layer, neuron) for vanilla
        category_key (tuple):
            one of the keys of COMBO_TO_NAME (for GLU variants), or an int (for vanilla)

    Returns:
        tensor: tensor of booleans of shape (layer, neuron)
    """
    key_tensor = torch.tensor(category_key).to(category_tensor.device)
    eq = category_tensor==key_tensor
    if category_tensor.dim()==3 and category_tensor.shape[-1]==3:
        return eq.all(dim=-1)
    return eq

def layerwise_count(category_tensor:torch.Tensor):
    """count how often each category appears per layer

    Args:
        category_tensor (tensor): output of compute_category,
            shape (layer, neuron, 3) for GLU variants, or (layer, neuron) for vanilla

    Returns:
        dict with
            keys: as in COMBO_TO_NAME (for GLU variants), or ints (for vanilla)
            values: tensors of shape (layer),
            indicating the number of times the category appears in each layer
    """
    if category_tensor.dim()==3 and category_tensor.shape[-1]==3:
        combo_name_dict = COMBO_TO_NAME
    else:
        combo_name_dict = {key[0]:COMBO_TO_NAME[key] for key in [(1,1,1),(0,0,1),(-1,1,1)]}
    results = {key:torch.zeros(category_tensor.shape[0]) for key in combo_name_dict}
    for key in combo_name_dict:
        results[key] = torch.count_nonzero(is_in_category(category_tensor, key), dim=1)
    return results

# def count_categories_all(category_tensor):
#     """Output:
#     dict with
#     keys: strings corresponding to layers: '0', '1', etc.
#     values: list of number of neurons (in that layer) for each class,
#     where classes are ordered as in CATEGORY_NAMES
#     """
#     results = {str(l):[] for l in range(category_tensor.shape[0])}
#     for i in range(len(CATEGORY_NAMES)):
#         entry = torch.count_nonzero(category_tensor==i, dim=1)
#         for l in range(category_tensor.shape[0]):
#             results[str(l)].append(entry[l])
#     return results

# def gather_class_changes(all_data, checkpoint_names=None):
#     """
#     returns: dict
#     with key (orig_class, new_class)
#     and value list of tuples (new_checkpoint, layer, neuron)
#     """
#     class_changes = {}
#     for i in range(11):
#         for j in range(11):
#             class_changes[(i,j)] = torch.empty((0,3)).cuda()
#     if checkpoint_names is None:
#         checkpoint_names = list(all_data.keys())
#     for checkpoint_nr, checkpoint_name in enumerate(checkpoint_names):
#         if checkpoint_nr == len(checkpoint_names)-1:
#             break
#         new_checkpoint_name = checkpoint_names[checkpoint_nr+1]
#         old_classes = all_data[checkpoint_name]['categories'].cuda()
#         new_classes = all_data[new_checkpoint_name]['categories'].cuda()
#         for t in class_changes:
#             indices = torch.nonzero((old_classes==t[0]) & (new_classes==t[1]))
#             #each row of indices is an index indicating: layer, neuron
#             indices_var = torch.cat(
#                 [
#                     torch.full(
#                         (indices.shape[0],1), checkpoint_nr+1
#                     ).cuda(),
#                     indices
#                 ],
#                 dim=1
#             )
#             #each row of indices_var indicates: new_checkpoint_nr, layer, neuron
#             class_changes[t] = torch.cat([class_changes[t], indices_var], dim=0)
#     for t in class_changes:
#         class_changes[t] = class_changes[t].cpu()
#     return class_changes
