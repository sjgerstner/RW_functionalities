"""Main code for the RW functionalities paper"""

import argparse
import os

import matplotlib.pyplot as plt

import torch

from weight_analysis_utils import utils, plotting, loading

torch.set_grad_enabled(False)

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

def _get_basic_data(args, data, model_name, cache_dir=None, checkpoint_value=None):
    model_data = loading.load_model_data(
        model_name,
        cache_dir=cache_dir, checkpoint_value=checkpoint_value,
        refactor_glu=args.refactor_glu,
        device=args.device,
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
        data = utils.norm_data(model_data=model_data, data_to_write=data)
    return data

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
    data = loading.load_data_if_exists(
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
    data = utils.get_advanced_data(
        experiments=args.experiments, data=data,
        #model_name,
        path=path,
        #checkpoint_value=checkpoint_value,
    )
    #create various plots if requested and save them
    print("plots")
    plotting.make_all_weight_based_plots(
        experiments=args.experiments,
        data=data,
        model_name=model_name,
        path=path,
        selected_model=args.selected_model,
        selected_layers=args.selected_layers,
    )

    return data

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', default='../RW_functionalities_results')
    parser.add_argument('--device', default='cuda:0')
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
                    checkpoint_value=loading.MODEL_TO_CHECKPOINTS[model_name][checkpoint_index],
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