import gc
from os.path import exists
import pickle

from torch import load
from torch.cuda import empty_cache

import einops
from transformer_lens.model_bridge import TransformerBridge

def load_model(
    model_name: str,
    processing_kwargs: dict[str,bool]|None=None,
    **load_kwargs,
):
    model = TransformerBridge.boot_transformers(model_name, **load_kwargs)
    if processing_kwargs is not None:
        model.enable_compatibility_mode(**processing_kwargs)
    return model

def load_model_legacy(
    model_name,
    processing=True,
    **kwargs,
    ):
    from transformer_lens import HookedTransformer, HookedEncoderDecoder
    if model_name.startswith("bert") or not processing:
        model = HookedTransformer.from_pretrained_no_processing(
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
    processing_kwargs = {}
    model_kwargs = kwargs.copy()
    model_kwargs["device"] = 'cpu'
    if not model_name.startswith("t5"):
        processing_kwargs["refactor_glu"]=refactor_glu
    # try:
    model = load_model(
        model_name,
        #local_files_only=True,
        processing_kwargs=processing_kwargs,
        **model_kwargs
    )
    # except Exception as e:
    #     print(
    #         f"Need to fetch remote files for model {model_name}. Ignored the following error: {e}"
    #     )
    #     model = load_model(model_name, local_files_only=False, processing_kwargs=processing_kwargs, **model_kwargs)

    out_dict = {}
    if hasattr(model, "W_gate") and model.W_gate is not None:
        out_dict["W_gate"] = model.W_gate.detach().to(device)
    out_dict["W_in"] = model.W_in.detach().to(device)
    out_dict["W_out"] = model.W_out.detach().to(device)
    #ensure shape: layer neuron model_dim
    for key,value in out_dict.items():
        if value.shape[1]==model.cfg.d_model:
            out_dict[key] = einops.rearrange(value, 'l d n -> l n d')
            assert out_dict[key].shape[2]==model.cfg.d_model, f"tensor {key} has a weird shape: {value.shape}"

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

def legacy_checkpoint_list():
    from transformer_lens.loading_from_pretrained import OLMO_CHECKPOINTS_1B, OLMO_CHECKPOINTS_7B
    return {
        'allenai/OLMo-1B-hf': OLMO_CHECKPOINTS_1B,
        'allenai/OLMo-7B-0424-hf': OLMO_CHECKPOINTS_7B,
    }
