from argparse import ArgumentParser
import os

import numpy as np
import torch

from weight_analysis_utils.plotting import freq_sim_scatter, plot_any_vs_any
from freqs import load_wout_norms, process_activation_data

SIGN_COMBOS = ["gate+_in+", "gate+_in-", "gate-_in+", "gate-_in-"]

parser = ArgumentParser()
parser.add_argument('--work_dir', default='.')
parser.add_argument('--neuroscope_dir', default='OLMo-7B-0424')
parser.add_argument(
    '--wcos_dir',
    default='.',#'wcos_casestudies',
)
parser.add_argument('--model', default='allenai/OLMo-7B-0424-hf')
parser.add_argument('--refactor_glu', action='store_true')
parser.add_argument('--combos', nargs='+', default=SIGN_COMBOS+["summary"])
parser.add_argument('--metric_type', default='freq')
parser.add_argument('--activation_location', default='hook_post')
parser.add_argument('--subexperiments', nargs='+', default=["separate", "aggregated", "selected"])
parser.add_argument('--layer_list', nargs='+', default=[0, 15, 31], type=int)
args = parser.parse_args()

PLOT_DIR = f'{args.work_dir}/plots/freq_vs_mean/{args.model}'
os.makedirs(PLOT_DIR, exist_ok=True)

NCOLS=4

#tensor of frequency by neuron
SUMMARY_PATH = f'{args.work_dir}/neuroscope/results/{args.neuroscope_dir}/summary{"_refactored" if args.refactor_glu else ""}.pt'
summary_dict = torch.load(SUMMARY_PATH, map_location="cuda:0")
n_layers = summary_dict[('gate+_in+', 'freq')].shape[0]
d_mlp = summary_dict[('gate+_in+', 'freq')].shape[1]
norm_wout = load_wout_norms(args.model)
for argscombo in args.combos:
    SUB_DIR = f'{PLOT_DIR}/{argscombo}'
    os.makedirs(SUB_DIR, exist_ok=True)
    data = {}
    if argscombo=='summary':
        data[f'{argscombo}_freq'] = (summary_dict[('gate+_in+', 'freq')]+summary_dict[('gate+_in-', 'freq')]).cpu()
        data[f'{argscombo}_sum'] = torch.sum(torch.stack(
            [
                summary_dict[(combo, 'freq')]*process_activation_data(
                    summary_dict, combo, args.activation_location
                )
                for combo in SIGN_COMBOS
            ],
        dim=0), dim=0)
    elif argscombo in SIGN_COMBOS:
        data[f'{argscombo}_freq'] = summary_dict[(argscombo, 'freq')].cpu()#layer neuron
        data[f'{argscombo}_sum'] = process_activation_data(
            summary_dict, argscombo, args.activation_location
        )
    data[f'{argscombo}_sum'] *= norm_wout #to make activations comparable across neurons
    data[f'{argscombo}_sum'] = data[f'{argscombo}_sum'].cpu()

    keys=(f'{argscombo}_freq', f'{argscombo}_sum')
    #plots
    if "separate" in args.subexperiments:
        plot_any_vs_any(
            data,
            keys=keys,
            arrangement=(int(np.ceil(n_layers/NCOLS)), NCOLS),
            bounded=(True, False),
            savefile=f'{SUB_DIR}/separate.pdf',
        )
    if "aggregated" in args.subexperiments:
        flattened_data = {
            key:torch.flatten(value)
            for key,value in data.items()
        }
        plot_any_vs_any(
            flattened_data,
            keys=keys,
            arrangement=(1,1),
            bounded=(True,False),
            #fit_line=True,
            savefile=f'{SUB_DIR}/aggregated.pdf'
        )
    if "selected" in args.subexperiments:
        data_by_layer = [
            {
                key:data[key][layer] for key in keys
            }
            for layer in range(n_layers)
        ]
        max_output = (
            None,
            torch.max(data[keys[1]]).item()
        )
        min_output = (
            None,
            torch.min(data[keys[1]]).item()
        )
        freq_sim_scatter(
            data_by_layer,
            keys=keys,
            arrangement = (1, len(args.layer_list)),
            savefile = f'{SUB_DIR}/{args.layer_list}.pdf',
            layer_list=args.layer_list if isinstance(args.layer_list, list) else [args.layer_list],
            max_output=max_output,
            min_output=min_output,
        )
