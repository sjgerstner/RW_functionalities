import pandas as pd
import torch

from transformer_lens import HookedTransformer

from src.weight_analysis_utils.utils import cos

def topk_df(vec, model:HookedTransformer, emb=None, k=64, nonneg=True):
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

def neuron_analysis(model:HookedTransformer, layer, neuron, emb=None, k=64, verbose=True):
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
    return linout, out_pos, out_neg, lin_pos, lin_neg
