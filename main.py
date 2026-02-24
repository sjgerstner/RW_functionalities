"""Main code for the RW functionalities paper"""

import argparse
import gc
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

import torch
import einops
from transformer_lens import HookedTransformer, HookedEncoderDecoder
from transformer_lens.loading_from_pretrained import OLMO_CHECKPOINTS_1B, OLMO_CHECKPOINTS_7B

import src.plotting as plotting
import src.utils as utils

torch.set_grad_enabled(False)

DEVICE='cuda:0'

EXPERIMENT_LIST = [
    "beta",
    "randomness", #compute 95 percent randomness regions (95 percent of 'mismatched' weight cosines are in this region)
    "norms", #norms of weight vectors
    "categories", #categorize the neurons
    "category_stats",#compute statistics of RW classes by layer
    #"quartiles",#compute quartiles of cosine similarities (by layer)
    "plot_fine",#create fine-grained plot
    "plot_selected",
    "plot_coarse",#create coarse-grained plot (categories by layer)
    "plot_boxplots",#make boxplots of cosine similarities by layer
    "plot_all_medians",#make one plot with the median cos(w_in,w_out) similarities (y) across layers (x) of all models (one line per model)
    "plot_selected_medians",
    "plot_norms",
    "plots_cosines_vs_norms",
    "plot_norm_in_norm_out",
]
MODEL_LIST = [
    "allenai/OLMo-7B-0424-hf",
    "allenai/OLMo-1B-hf",
    "gemma-2-2b",
    "gemma-2-9b",
    "Llama-2-7b",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "mistral-7b",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-7B",
    "yi-6b",
]

MODEL_TO_CHECKPOINTS = {
    'allenai/OLMo-1B-hf': OLMO_CHECKPOINTS_1B,
    'allenai/OLMo-7B-0424-hf': OLMO_CHECKPOINTS_7B,
}

ARRANGEMENT_NEEDED_LIST = ["plot_fine", "plots_norms", "plot_cosines_vs_norms", "plot_norm_in_norm_out"]

def _load_model(model_name, **kwargs):
    if model_name.startswith("bert"):
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

def _load_model_data(model_name, cache_dir=None, checkpoint_value=None, refactor_glu=True):
    model_kwargs = {
        "checkpoint_value":checkpoint_value,
        "cache_dir":cache_dir,
        "device":"cpu",
    }
    if not model_name.startswith("t5"):
        model_kwargs["refactor_glu"]=refactor_glu
    try:
        model = _load_model(model_name, local_files_only=True, **model_kwargs)
    except Exception as e:
        print(
            f"Need to fetch remote files for model {model_name}. Ignored the following error: {e}"
        )
        model = _load_model(model_name, local_files_only=False, **model_kwargs)

    out_dict = {}
    #new shape: layer neuron model_dim
    if hasattr(model, "W_gate") and model.W_gate is not None:
        out_dict["W_gate"] = einops.rearrange(model.W_gate.detach(), 'l d n -> l n d').to(DEVICE)
    out_dict["W_in"] = einops.rearrange(model.W_in.detach(), 'l d n -> l n d').to(DEVICE)
    out_dict["W_out"] = model.W_out.detach().to(DEVICE) #already has the shape we want
    #sanity check,comment out
    #print(W_gate.shape, W_in.shape, W_out.shape)#should all be the same

    out_dict["d_model"] = model.cfg.d_model

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return out_dict

def cosines(mlp_weights):
    """Compute weight cosines within each neuron of the given model
    and return result as a dict of three tensors (gatelin, linout, gateout),
    each of which is of shape (layer neuron)."""
    data = {"d_model":mlp_weights["d_model"]}
    data["linout"] = utils.cos(mlp_weights["W_in"], mlp_weights["W_out"]).cpu()
    if "W_gate" in mlp_weights:
        data["gatelin"] = utils.cos(mlp_weights["W_gate"], mlp_weights["W_in"]).cpu() #don't overload the gpu
        data["gateout"] = utils.cos(mlp_weights["W_gate"], mlp_weights["W_out"]).cpu()
    return data

def _load_data_if_exists(path):
    data_file = f"{path}/data.pt"
    if os.path.exists(data_file):
        #print(torch.serialization.get_unsafe_globals_in_checkpoint(data_file))
        data = torch.load(
            data_file, map_location='cpu',
            #weights_only=False,
        )
    elif os.path.exists(f"{path}/data.pickle"):
        #try:
        with open(f"{path}/data.pickle", 'rb') as f:
            data = pickle.load(f)
        # except RuntimeError as e:
        #     print(f"Ignoring error when loading pickle, recomputing data: {e}")
    else:
        data={}
    return data

# def _get_cosine_data(path, model_data):
#     """load and/or compute cosine data"""
#     data_file = f"{path}/data.pt"
#     if os.path.exists(data_file):
#         #print(torch.serialization.get_unsafe_globals_in_checkpoint(data_file))
#         data = torch.load(
#             data_file, map_location='cpu',
#             #weights_only=False,
#         )
#     elif os.path.exists(f"{path}/data.pickle"):
#         #try:
#         with open(f"{path}/data.pickle", 'rb') as f:
#             data = pickle.load(f)
#         # except RuntimeError as e:
#         #     print(f"Ignoring error when loading pickle, recomputing data: {e}")
#     else:
#         print("computing cosines")
#         data = cosines(model_data)
#     return data

def _norm_data(model_data, data_to_write):
    data_to_write["norm_gate"] = torch.linalg.vector_norm(model_data["W_gate"], dim=-1)
    data_to_write["norm_in"] = torch.linalg.vector_norm(model_data["W_in"], dim=-1)
    data_to_write["norm_out"] = torch.linalg.vector_norm(model_data["W_out"], dim=-1)
    data_to_write["norm_in_out"] = data_to_write["norm_in"]*data_to_write["norm_out"]
    return data_to_write    

def _get_basic_data(args, data, model_name, cache_dir=None, checkpoint_value=None):
    model_data = _load_model_data(
        model_name,
        cache_dir=cache_dir, checkpoint_value=checkpoint_value,
        refactor_glu=args.refactor_glu,
    )
    if "linout" not in data:
        print("computing cosines")
        data = cosines(model_data)
    if "beta" in args.experiments and "beta" not in data:
        print("computing beta randomness regions")
        data["beta"] = utils.beta_randomness_region(d=data["d_model"])
    if "randomness" in args.experiments and "randomness" not in data:
        print("computing layer-specific randomness regions")
        data["randomness"] = utils.randomness_regions(model_data)
    if "norms" in args.experiments and "norm_gate" not in data:
        print("computing norms of weight vectors")
        data = _norm_data(model_data=model_data, data_to_write=data)
    return data

def _get_advanced_data(args, data, model_name, path, checkpoint_value=None):
    """load and/or compute advanced data"""
    if checkpoint_value is not None:
        model_name=f"{model_name}/{checkpoint_value}"
    #categories and category statistics
    if "categories" in args.experiments:# and 'categories' not in data:
        print("classifying neurons")
        data['categories'] = utils.compute_category(
            # gatelin=data['gatelin'].to(DEVICE),
            # gateout=data['gateout'].to(DEVICE),
            # linout=data['linout'].to(DEVICE)
            data=data, device=DEVICE,
            ) #layer neuron
    if "category_stats" in args.experiments:# and 'category_stats' not in data:
        print("category statistics")
        data['category_stats'] = utils.layerwise_count(data['categories'])
    torch.save(data, f"{path}/data.pt")
    return data

def _make_plots(args, data, model_name, path):
    """make plots"""
    layers = data['linout'].shape[0]
    arrangement_needeed = any(s in args.experiments for s in ARRANGEMENT_NEEDED_LIST)
    if arrangement_needeed:
        aggregated_data = {
            key:einops.rearrange(value, 'l n -> 1 (l n)') for key,value in data.items()
            if isinstance(value, torch.Tensor) and value.dim()==2
        }
        ncols = 4
        arrangement = (int(np.ceil(layers/ncols)), ncols)
        #fine-grained / cosines
        if "plot_fine" in args.experiments:# and not os.path.exists(f"{path}/fine.pdf"):
            fig, _ax = plotting.wcos_plot(
                data,
                range(layers),
                arrangement = arrangement,
                # model_name=model_name
            )
            fig.savefig(
                f"{path}/fine.pdf",
                bbox_inches='tight',
                #dpi=400
            )
            plt.close()
        #norms of weight vectors
        if "plot_norms" in args.experiments:
            fig, _ax = plotting.plot_norms(data, arrangement=arrangement)
            fig.savefig(f"{path}/norms.pdf", bbox_inches="tight")
            plt.close()
            fig, _ax = plotting.plot_norms(aggregated_data, arrangement=(1,1))
            fig.savefig(f"{path}/norms_all_layers.pdf")
            plt.close()
        if "plot_norm_in_norm_out" in args.experiments:
            fig, _ax = plotting.plot_norms(
                data, arrangement=arrangement, keys=("norm_in", "norm_out")
            )
            fig.savefig(f"{path}/norm_in_norm_out.pdf", bbox_inches="tight")
            plt.close()
            fig, _ax = plotting.plot_norms(
                aggregated_data, arrangement=(1,1), keys=("norm_in", "norm_out")
            )
            fig.savefig(f"{path}/norm_in_norm_out_all_layers.pdf", bbox_inches="tight")
            plt.close()
        #cosines vs norms
        if "plot_cosines_vs_norms" in args.experiments:
            for cosine_key in ["linout", "gateout", "gatelin"]:
                for norms in ["norm_gate", "norm_in_out"]:
                    fig, _ax = plotting.plot_cosines_vs_norms(
                        data, arrangement=arrangement, keys=(cosine_key, norms)
                    )
                    fig.savefig(f"{path}/{cosine_key}_{norms}.pdf", bbox_inches="tight")
                    plt.close()
                    fig, _ax = plotting.plot_cosines_vs_norms(
                        aggregated_data, arrangement=(1,1), keys=(cosine_key, norms)
                    )
                    fig.savefig(f"{path}/{cosine_key}_{norms}_all_layers.pdf", bbox_inches="tight")
                    plt.close()
    #fine-grained / cosines for selected layers of selected model
    if "plot_selected" in args.experiments and model_name==args.selected_model:
        fig, _ax = plotting.wcos_plot(
            data,
            args.selected_layers,
            arrangement= (1, len(args.selected_layers)),
            # model_name=model_name,
        )
        fig.savefig(
            f"{path}/selected.pdf",
            bbox_inches="tight"
        )
        plt.close()
    #coarse-grained / category stats
    if "plot_coarse" in args.experiments:# and not os.path.exists(f"{path}/coarse.pdf"):
        fig, _ax = plotting.my_survey(data['category_stats'], model_name)
        fig.savefig(f"{path}/coarse.pdf", bbox_inches='tight')
        plt.close()
    #quartiles
    if "plot_boxplots" in args.experiments:# and not os.path.exists(f"{path}/quartiles.pdf")
        fig, _ax = plotting.plot_boxplots(data, model_name)
        fig.savefig(f"{path}/boxplot.pdf", bbox_inches='tight')
        plt.close()

def analysis(args, model_name, cache_dir=None, checkpoint_value=None):
    """General function
    that computes weight cosines of the given model
    and then does the analyses specified in the args"""

    #path
    path = f"{args.work_dir}/results/{model_name}"
    if args.refactor_glu:
        path += '/refactored'
    if checkpoint_value is not None:
        path += f"/checkpoints/{checkpoint_value}"
    if not os.path.exists(path):
        os.makedirs(path)

    #load and/or compute data and save it
    data = _load_data_if_exists(
        path
    )
    #cosines etc.
    if (
        ("linout" not in data)
        or (("randomness" in args.experiments) and "randomness" not in data)
        or (("norms" in args.experiments) and "norm_gate" not in data)
    ):
        data = _get_basic_data(
            args, data, model_name, cache_dir=cache_dir, checkpoint_value=checkpoint_value
        )
        torch.save(data, f"{path}/data.pt")
    #advanced
    data = _get_advanced_data(
        args, data, model_name, path, checkpoint_value=checkpoint_value
    )
    #create various plots if requested and save them
    print("plots")
    _make_plots(args, data, model_name, path)

    return data

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', default='.')
    parser.add_argument(
        '--refactor_glu',
        action='store_true',
        help='whether to refactor the weights such that cos(w_gate,w_in)>=0'
    )
    parser.add_argument(
        '--experiments',
        nargs='+',
        default=EXPERIMENT_LIST,
        help="which experiment(s) to perform, see EXPERIMENT_LIST for details"
    )
    parser.add_argument(
        "--model",
        nargs='+',
        default=MODEL_LIST,
        help='one or several TransformerLens models',
        )
    parser.add_argument(
        "--checkpoints",
        nargs='+',
        default=[None],
        type=int,
        help="""which training checkpoints of the given model to analyse
        (type: int representing the index in the official checkpoint list)"""
    )
    parser.add_argument(
        "--selected_model",
        default="meta-llama/Llama-3.2-3B",
        help="model to plot selected layers from",
    )
    parser.add_argument(
        "--selected_layers",
        nargs='+',
        default=[0,14,27],
        help="selected layers for main paper plot"
    )
    args = parser.parse_args()
    if "plot_all_medians" in args.experiments or "plot_selected_medians" in args.experiments:
        model_to_medians_dict = {}
    for model_name in args.model:
        print(model_name)
        if args.checkpoints==[None]:
            data = analysis(args, model_name)
            if "plot_all_medians" in args.experiments or "plot_selected_medians" in args.experiments:
                #model_to_medians_dict[model_name] = data["linout_quartiles"][2,:].flatten().cpu()
                model_to_medians_dict[model_name] = utils.torch_quantile(data["linout"], q=.5, dim=1)
            del data
        else:
            for checkpoint_index in args.checkpoints:
                data = analysis(
                    args, model_name,
                    checkpoint_value=MODEL_TO_CHECKPOINTS[model_name][checkpoint_index],
                    cache_dir='/nfs/datz/olmo_models',
                )
                del data
    if "plot_all_medians" in args.experiments:
        fig, ax = plotting.plot_all_medians(model_to_medians_dict)
        fig.savefig(f'{args.work_dir}/results/medians.pdf', bbox_inches='tight')
        plt.close()
    if "plot_selected_medians" in args.experiments:
        tiny_models = [
            model_name for model_name in MODEL_LIST
            if "1b" in model_name.lower() or "0.5b" in model_name.lower()
        ]
        filtered_dict = {
            key:value for key,value in model_to_medians_dict.items() if key not in tiny_models
        }
        fig, ax = plotting.plot_all_medians(filtered_dict)
        fig. savefig(f'{args.work_dir}/results/selected_medians.pdf', bbox_inches='tight')
        plt.close()
