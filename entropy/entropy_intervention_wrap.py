import argparse
import os

import torch
import datasets
from transformer_lens import HookedTransformer

from entropy.entropy_intervention import run_intervention_experiment
from neuron_choice import neuron_choice
from utils import NAME_TO_COMBO

def get_mean_values(args):
    summary_dict = torch.load(args.means_path, map_location=args.device)
    relevant_gate_signs=[args.gate] if args.gate else ['+','-']
    relevant_post_signs=[args.post] if args.post and args.gate else ['+','-']
    relevant_cases_post = [
        f'gate{gate_sign}_post{post_sign}'
        for gate_sign in relevant_gate_signs for post_sign in relevant_post_signs
    ]
    relevant_cases_in = [
        post_string.replace('post', 'in') if post_string.startswith('gate+')
        else post_string.replace('post+', 'in-').replace('post-', 'in+')
        for post_string in relevant_cases_post
    ]
    mean_values = torch.sum(torch.stack([
            summary_dict[(case_key,'freq')]*torch.nan_to_num(summary_dict[(case_key, 'hook_post', 'sum')])
            for case_key in relevant_cases_in
        ], dim=0), dim=0) / torch.sum(torch.stack([
            summary_dict[case_key,'freq'] for case_key in relevant_cases_in
        ], dim=0), dim=0)
    return mean_values

def run_with_baseline(
    args,
    model,
    tokenized_dataset,
    device,
    neuron_list=None,
    random_baseline=None,
    subset=None,
):
    # if neuron_list is None:
    #     neuron_list = []
    if neuron_list:
        neuron_subset_name=f'{args.neuron_subset_name}{subset if subset else ""}'
        intervention_type=args.intervention_type
    else:
        neuron_subset_name='baseline'
        random_baseline=None
        intervention_type=None
    if intervention_type=='mean_ablation':
        mean_values = get_mean_values(args)
    else:
        mean_values = None

    run_if_necessary(
        args,
        model,
        tokenized_dataset,
        device,
        neuron_subset=neuron_list,
        neuron_subset_name=neuron_subset_name,
        intervention_type=intervention_type,
        mean_values=mean_values,
    )
    if random_baseline is not None and args.gate is None and args.post is None:
        run_if_necessary(
            args,
            model,
            tokenized_dataset,
            device,
            neuron_subset=random_baseline,
            neuron_subset_name=f'{neuron_subset_name}_baseline',
            intervention_type=intervention_type,
            mean_values=mean_values,
        )

def run_if_necessary(
    args,
    model,
    tokenized_dataset,
    device,
    neuron_subset,
    neuron_subset_name=None,
    intervention_type=None,
    mean_values:torch.Tensor|None=None,
):
    if not neuron_subset_name:
        neuron_subset_name = '_'.join([f'{l}.{n}' for l, n in neuron_subset])
    if args.gate or args.post:
        neuron_subset_name = f'{neuron_subset_name}_gate{args.gate}_post{args.post}'
    save_path = os.path.join(
        args.data_dir,
        args.output_dir,
        args.model,
        args.token_dataset.split('/')[-1],
        neuron_subset_name,
        str(intervention_type)+'_'+str(args.intervention_param),
    )

    print(neuron_subset_name)
    if not os.path.exists(save_path):
        run_intervention_experiment(
            args,
            model,
            tokenized_dataset,
            device,
            neuron_subset,
            neuron_subset_name,
            save_path=save_path,
            mean_values=mean_values,
        )
    else:
        print("already computed, skipping")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # general arguments
    parser.add_argument('--work_dir', default='.')
    parser.add_argument('--data_dir', default='.')
    parser.add_argument('--wcos_dir', default='.')
    parser.add_argument(
        '--output_dir', default='intervention_results')
    parser.add_argument('--means_path', default='neuroscope/results/OLMo-7B-0424/summary_refactored.pt')
    parser.add_argument(
        '--model',
        default='allenai/OLMo-7B-0424-hf',
        help='Name of model from TransformerLens')
    parser.add_argument(
        '--token_dataset',
        default='neuroscope/datasets/dolma-small',
        help='Name of cached feature dataset')
    parser.add_argument(
        '--activation_location', default='mlp.hook_post',
        help='Model component to save')

    # activation processing/subsetting arguments
    parser.add_argument(
        '--batch_size', default=32, type=int)
    parser.add_argument(
        '--device', default=torch.device('cuda' if torch.cuda.is_available() else (
            'mps' if torch.backends.mps.is_available() else 'cpu')), type=str,
    )

    parser.add_argument(
        '--neuron_subset_name', default='baseline',
        help="'baseline' or one of the names in NAME_TO_COMBO"
    )
    parser.add_argument(
        '--n_neurons', default=None,
        help="""(None or float or int, default None): how many neurons to choose from the category.
            If None (default): take all neurons from the category.
            If float (should be between 0. and 1.): proportion from the category
            If int: absolute numbers"""
    )
    parser.add_argument(
        '--gate', default=None,
        help="ablate only when x_gate has this sign ('+' or '-')"
    )
    parser.add_argument(
        '--post', default=None,
        help="ablate only when activation*cos(w_gate,w_in) has this sign ('+' or '-')"
    )

    parser.add_argument(
        '--intervention_type',
        choices=["zero_ablation", "threshold_ablation", "fixed_activation", "relu_ablation", "mean_ablation"],
        default="zero_ablation",
    )
    parser.add_argument(
        '--intervention_param', type=float, default=None,
        help='Parameter for intervention type (eg, threshold or fixed activation)')

    # saving arguments
    parser.add_argument(
        '--save_precision', default=16, type=int)
    # parser.add_argument(
    #     '--separate', action='store_true',
    #     help='also do the ablation analysis for each neuron separately'
    # )

    args = parser.parse_args()

    device = args.device

    model = HookedTransformer.from_pretrained(args.model, device=device, refactor_glu=True)
    #model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    tokenized_dataset = datasets.load_from_disk(
        f'{args.data_dir}/{args.token_dataset}'
    )
    tokenized_dataset = tokenized_dataset.select_columns('input_ids')
    if args.neuron_subset_name=='baseline':
        neuron_list = []
        random_baseline=None
        subset=None
    else:
        if (args.n_neurons is None) or (args.n_neurons=='None'):
            subset = None
        else:
            subset = float(args.n_neurons) if '.' in args.n_neurons else int(args.n_neurons)
        neuron_list, random_baseline = neuron_choice(
            args,
            category_key=NAME_TO_COMBO[args.neuron_subset_name],
            subset=subset,
        )

    run_with_baseline(
        args,
        model,
        tokenized_dataset,
        device,
        neuron_list,
        random_baseline,
        subset=subset,
    )
