import gc
from os.path import exists
import pickle

from torch import load
from torch.cuda import empty_cache

import einops
from transformer_lens import HookedTransformer, HookedEncoderDecoder
from transformer_lens.loading_from_pretrained import OLMO_CHECKPOINTS_1B, OLMO_CHECKPOINTS_7B

MODEL_TO_CHECKPOINTS = {
    'allenai/OLMo-1B-hf': OLMO_CHECKPOINTS_1B,
    'allenai/OLMo-7B-0424-hf': OLMO_CHECKPOINTS_7B,
}

def load_model(
    model_name,
    processing=True,
    **kwargs,
):
    if model_name.startswith("bert") or not processing:
        model = HookedTransformer.from_pretrained_no_processing(#TODO change to HookedEncoder?
            model_name, **kwargs,
        )
    elif model_name.startswith("t5"):
        model = HookedEncoderDecoder.from_pretrained(
            model_name, **kwargs,
        )
    else:
        model = HookedTransformer.from_pretrained(
            model_name,
            **kwargs,
        )
    return model

def load_model_data(
    model_name,
    refactor_glu=True, device='cuda:0',
    **kwargs
):
    model_kwargs = kwargs.copy()
    model_kwargs["device"] = 'cpu'
    if not model_name.startswith("t5"):
        model_kwargs["refactor_glu"]=refactor_glu
    try:
        model = load_model(model_name, local_files_only=True, **model_kwargs)
    except Exception as e:
        print(
            f"Need to fetch remote files for model {model_name}. Ignored the following error: {e}"
        )
        model = load_model(model_name, local_files_only=False, **model_kwargs)

    out_dict = {}
    #new shape: layer neuron model_dim
    if hasattr(model, "W_gate") and model.W_gate is not None:
        out_dict["W_gate"] = einops.rearrange(model.W_gate.detach(), 'l d n -> l n d').to(device)
    out_dict["W_in"] = einops.rearrange(model.W_in.detach(), 'l d n -> l n d').to(device)
    out_dict["W_out"] = model.W_out.detach().to(device) #already has the shape we want

    out_dict["d_model"] = model.cfg.d_model

    del model
    gc.collect()
    empty_cache()

    return out_dict

def load_data_if_exists(path):
    data_file = f"{path}/data.pt"
    if exists(data_file):
        data = load(
            data_file, map_location='cpu',
            #weights_only=False,
        )
    elif exists(f"{path}/data.pickle"):
        with open(f"{path}/data.pickle", 'rb') as f:
            data = pickle.load(f)
    else:
        data={}
    return data
