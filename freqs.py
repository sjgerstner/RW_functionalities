from argparse import ArgumentParser
from itertools import chain
import os

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler
import torch
from transformer_lens import HookedTransformer

import neuron_choice
from src.plotting import SHORT_TO_LONG, _short_to_long, aligned_histograms, freq_sim_scatter, plot_any_vs_any
from src.utils import COMBO_TO_NAME

def load_wout_norms(model_name:str)->torch.Tensor:
    path = f"results/{model_name}/vector_lengths.pt"
    if os.path.exists(path):
        return torch.load(path)
    #load model, refactor_glu doesn't change anything
    model = HookedTransformer.from_pretrained(model_name, refactor_glu=False)
    norm_wout = torch.linalg.vector_norm(model.W_out, dim=-1)
    del model
    torch.cuda.empty_cache()
    #with open(path, "w", encoding="utf-8") as f:
    torch.save(norm_wout, path)
    return norm_wout

def process_activation_data(summary_dict, combo, activation_location):
    return torch.abs(torch.nan_to_num(summary_dict[(combo, activation_location, 'sum')]))

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--work_dir', default='.')
    parser.add_argument('--neuroscope_dir', default='OLMo-7B-0424')
    parser.add_argument(
        '--wcos_dir',
        default='.',#'wcos_casestudies',
    )
    parser.add_argument('--model', default='allenai/OLMo-7B-0424-hf')
    parser.add_argument('--refactor_glu', action='store_true')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--combo', default='summary')
    parser.add_argument('--metric_type', default='freq')
    parser.add_argument('--activation_location', default='hook_post')
    parser.add_argument('--subexperiments', nargs='+', default=['all'])
    parser.add_argument('--layer_list', nargs='+', default=[0, 15, 31], type=int)
    args = parser.parse_args()

    if 'all' in args.subexperiments:
        subexps = ["layer_plots", "category_plots", "scatter_plots", "table", "selected", "all_layers"]
    else:
        subexps = args.subexperiments

    #tensor of frequency by neuron
    SUMMARY_PATH = f'{args.work_dir}/neuroscope/results/{args.neuroscope_dir}/summary{"_refactored" if args.refactor_glu else ""}.pt'
    summary_dict = torch.load(SUMMARY_PATH, map_location="cuda:0")
    if args.metric_type=='freq':
        if args.combo=='summary':
            data_tensor = summary_dict[('gate+_in+', 'freq')]+summary_dict[('gate+_in-', 'freq')]
        else:
            data_tensor = summary_dict[(args.combo, 'freq')]#layer neuron
        #max_output=None#dummy
    elif args.metric_type=='sum':#actually means mean
        norm_wout = load_wout_norms(args.model)
        if args.combo=='summary':
            data_tensor = torch.sum(torch.stack(
                [
                    summary_dict[(combo, 'freq')]*process_activation_data(
                        summary_dict, combo, args.activation_location
                    )
                    for combo in ["gate+_in+", "gate+_in-", "gate-_in+", "gate-_in-"]
                ],
            dim=0), dim=0)
        else:
            data_tensor = process_activation_data(summary_dict, args.combo, args.activation_location)
        data_tensor *= norm_wout #to make activations comparable across neurons
        #max_output = torch.max(data_tensor).item()
    else:
        raise NotImplementedError(
            f"args.metric_type has to be one of 'freq' or 'sum', but was {args.metric_type}"
        )
    data_tensor = data_tensor.cpu()

    n_layers = data_tensor.shape[0]
    d_mlp = data_tensor.shape[1]

    PLOT_DIR = f'{args.work_dir}/plots/{args.combo}_{args.metric_type}/{args.model}'
    os.makedirs(PLOT_DIR, exist_ok=True)

    #plotting by layer
    LAYER_PATH = f'{PLOT_DIR}/layers{"_log" if args.log else ""}.pdf'
    if "layer_plots" in subexps:#not os.path.exists(LAYER_PATH):
        aligned_histograms(
            [list(data_tensor[layer]) for layer in range(n_layers)],
            [f"Layer {i}" for i in range(n_layers)],
            savefile = LAYER_PATH,
            #suptitle = f"{SHORT_TO_LONG[args.combo]} by layer in {args.model}",
            xlabel=_short_to_long(args.combo),
            ylabel="proportion of neurons",
            ncols=4,
            log=args.log,
            weighted=True,
        )

    #plotting by category
    CATEGORY_PATH = f'{PLOT_DIR}/categories_{args.log}.pdf'
    if "category_plots" in subexps or "table" in subexps:#not os.path.exists(CATEGORY_PATH):
        my_data = []
        for i,key in enumerate(COMBO_TO_NAME.keys()):
            #lists of (layer neuron) tuples
            neuron_list, baseline_list = neuron_choice.neuron_choice(args, key)
            if neuron_list is None:#too few neurons in category
                continue
            my_data.append([data_tensor[index].item() for index in neuron_list])
            my_data.append([data_tensor[index].item() for index in baseline_list])

        if "category_plots" in subexps:
            aligned_histograms(
                my_data,
                list(chain.from_iterable(
                    (name, name+'_baseline') for name in COMBO_TO_NAME.values()
                )),
                savefile=CATEGORY_PATH,
                #suptitle=f"{SHORT_TO_LONG[args.combo]} of neurons in {args.model}",
                xlabel=SHORT_TO_LONG[args.combo],
                ylabel="proportion of neurons",
                ncols=2,
                log=args.log,
                weighted=True
            )
        if "table" in subexps:
            PICKLE_ALL = f'{PLOT_DIR}/{args.combo}_all.pickle'
            if not os.path.exists(PICKLE_ALL):
                #tensor of category by neuron
                PATH = f"{args.work_dir}/{args.wcos_dir}/results/{args.model}"
                DATA_PATH = f"{PATH}/data.pt"
                if not os.path.exists(DATA_PATH):
                    DATA_PATH = f"{PATH}/refactored/data.pt"
                data = torch.load(DATA_PATH)
                # with open(f"{PATH}/data.pickle", 'rb') as f:
                #     data = pickle.load(f)
                category_tensor = data['categories']#layer neuron
                #we have a tensor of frequencies (already above) a tensor of categories
                #later we want to choose subsets based on layer only (done above), category only,
                # or layer and category
                #for hist plots (first two cases) and for means (all cases)
                # and counts (last case, should be done by the wcos code)
                #TODO save means as separate dataframes
            else:
                pass#TODO
            PICKLE_MEANS = f'{PLOT_DIR}/{args.combo}_means.pickle'
            if not os.path.exists(PICKLE_MEANS):
                means = [np.mean(np.array(subset)) for subset in my_data]
                means = np.array(means).reshape((len(COMBO_TO_NAME),2))
                table = pd.DataFrame(
                    means, index=COMBO_TO_NAME.values(), columns=["true", "baseline"]
                )
                table.to_pickle(PICKLE_MEANS)
            else:
                table = pd.read_pickle(PICKLE_MEANS)
            styler = Styler(table, precision=3)
            styler.to_latex(
                f'{PLOT_DIR}/{args.combo}.tex', label=f"tab:{args.combo}",
                caption="""
                Activation frequencies by RW class.
                The second column represents random neurons taken from the same layers.
                """,
            )

    #scatter plots
    if any(s in subexps for s in ["scatter_plots", "selected", "all_layers", "norms"]):
        #tensor of category by neuron
        PATH = f"{args.work_dir}/{args.wcos_dir}/results/{args.model}/refactored"
        data = torch.load(f"{PATH}/data.pt")
        # with open(f"{PATH}/data.pt", 'rb') as f:
        #     data = pickle.load(f)
        # linout = data["linout"]
        # gateout = data["gateout"]
        # gatelin = data["gatelin"]
        data[f'{args.combo}_{args.metric_type}'] = data_tensor

        #scatter plots
        SIMS = []
        if any(s in subexps for s in ["scatter_plots", "selected", "all_layers"]):
            SIMS += ["linout", "gateout", "gatelin"]
        if "norms" in subexps:
            SIMS += ["norm_gate", "norm_in_out"]
        if "all_layers" in subexps:
            flattened_data = {
                key:torch.flatten(value)
                for key,value in data.items()
                if key in SIMS or key==f'{args.combo}_{args.metric_type}'
            }
        NCOLS = 4
        for sim in SIMS:
            #absolute = (sim=="gatelin", None)
            if "scatter_plots" in subexps or "norms" in subexps:
                plot_any_vs_any(
                    data,
                    keys=(sim, f'{args.combo}_{args.metric_type}'),
                    arrangement=(int(np.ceil(n_layers/NCOLS)), NCOLS),
                    bounded=(not sim.startswith('norm'), args.metric_type=='freq'),
                    #fit_line=True,
                    savefile=f'{PLOT_DIR}/{sim}.pdf',
                )
            if "all_layers" in subexps:
                plot_any_vs_any(
                    flattened_data,
                    keys=(sim, f'{args.combo}_{args.metric_type}'),
                    arrangement=(1,1),
                    bounded=(not sim.startswith('norm'), args.metric_type=='freq'),
                    #fit_line=True,
                    savefile=f'{PLOT_DIR}/{sim}_all_layers.pdf'
                )
        if "selected" in subexps:
            keys=("linout", f'{args.combo}_{args.metric_type}')
            data_by_layer = [
                {
                    key:data[key][layer] for key in keys
                }
                for layer in range(n_layers)
            ]
            max_output = (
                None,
                None if args.metric_type=='freq' else torch.max(data[keys[1]]).item()
            )
            min_output = (
                None,
                None if args.metric_type=='freq' else torch.min(data[keys[1]]).item()
            )
            # if args.metric_type=='sum':
            #     print(max_output[1], type(max_output[1]))
            #     print(min_output[1], type(min_output[1]))
            freq_sim_scatter(
                data_by_layer,
                keys=keys,
                arrangement = (1, len(args.layer_list)),
                #suptitle = f"{SHORT_TO_LONG[args.combo]} vs. {SHORT_TO_LONG["linout"]} in {args.model}",
                savefile = f'{PLOT_DIR}/linout_{args.layer_list}.pdf',
                layer_list=args.layer_list if isinstance(args.layer_list, list) else [args.layer_list],
                max_output=max_output,
                min_output=min_output,
            )
