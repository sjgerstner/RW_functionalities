from argparse import ArgumentParser
import os
# Scientific packages
import torch
from transformer_lens import HookedTransformer
# Utilities
from neuron_choice import neuron_choice, get_n_neurons
from utils import NAME_TO_COMBO
from entropy.entropy_intervention_wrap import get_mean_values
from attributes.utils import make_neuron_hooks

torch.set_grad_enabled(False)

def generate_ablated(
    args,
    model:HookedTransformer,
    neuron_subset:list,
    mean_values:torch.Tensor|None=None,
    **generate_kwargs,
    )->str:
    hooks = make_neuron_hooks(
            args, neuron_subset, mean_values,
        )
    for hook in hooks:
        model.add_hook(*hook)
    text = model.generate(
        "<|endoftext|>",
        return_type="str",
        **generate_kwargs
    )
    assert isinstance(text, str)
    model.remove_all_hook_fns()
    return text

def generate_and_save(save_path, file_name, **kwargs):
    text = generate_ablated(**kwargs)
    with open(f"{save_path}/{file_name}", "x", encoding="utf-8") as f:
        f.write(text)

def _get_args():
    parser = ArgumentParser()
    parser.add_argument('--work_dir', default='.')
    parser.add_argument('--wcos_dir', default='.')
    parser.add_argument('--means_path', default='neuroscope/results/7B_new/summary_refactored.pt')
    parser.add_argument('--model', default='allenai/OLMo-7B-0424-hf')
    parser.add_argument(
        '--intervention_type',
        choices=["zero_ablation", "threshold_ablation", "fixed_activation", "relu_ablation", "mean_ablation"],
        default="zero_ablation",
    )
    parser.add_argument('--activation_location', default='mlp.hook_post')
    parser.add_argument(
        '--n_neurons', default=None,
        help="""(None or float or int, default None): how many neurons to choose from the category.
            If None (default): take all neurons from the category.
            If float (should be between 0. and 1.): proportion from the category
            If int: absolute numbers"""
    )
    parser.add_argument(
        '--by_freq', default=None,
        help="""a metric for activation frequency, to adapt the number of neurons to ablate
        (ablate more neurons for less frequently activated classes)"""
    )
    parser.add_argument(
        '--constant_class', default='weakening',
        help="The class from which to ablate n_neurons"
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
        '--subsets',
        nargs='+',
        default=[
            "strengthening",
            "conditional strengthening",
            "proportional change",
            "weakening",
            "conditional weakening",
        ]
    )
    parser.add_argument('--max_new_tokens', default=1024, type=int)
    parser.add_argument(
        '--do_sample', action='store_true',
        help="""
        If true, sample from the full model distribution; otherwise, always select the top next token.
        ATTENTION: NO SEED IS SET YET, SO THE RESULTS ARE NON DETERMINISTIC!
        """
    )
    return parser.parse_args()

if __name__=="__main__":
    args = _get_args()
    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,#TODO set a seed for this case!
    }
    save_path = os.path.join(
        "generations",
        args.model,
    )
    if args.intervention_type=='mean_ablation':
        mean_values = get_mean_values(args)
    else:
        mean_values = None

    model = HookedTransformer.from_pretrained(args.model)
    short_model_name = args.model.split('/')[-1]

    N_NEURONS, _constant = get_n_neurons(args)
    temperature_str = "full" if args.do_sample else "greedy"

    #baseline (no ablations)
    save_path_none = os.path.join(
            save_path,
            "baseline",
        )
    os.makedirs(save_path_none, exist_ok=True)
    if not os.path.exists(f"{save_path_none}/generated_{temperature_str}.txt"):
        generate_and_save(
            save_path=save_path_none,
            file_name=f"generated_{temperature_str}.txt",
            args=args,
            model=model,
            neuron_subset=[],
            mean_values=mean_values,
            **generate_kwargs,
        )

    for subset_name in args.subsets:
        clear_subset_name = f'{subset_name}{N_NEURONS}' if N_NEURONS else subset_name
        baseline_name=f'{clear_subset_name}_baseline'
        save_path_clear = os.path.join(
            save_path,
            clear_subset_name,
            str(args.intervention_type),
        )
        save_path_baseline = os.path.join(
            save_path,
            baseline_name,
            str(args.intervention_type),
        )
        os.makedirs(save_path_clear, exist_ok=True)
        os.makedirs(save_path_baseline, exist_ok=True)
        #TODO number of neurons to ablate from this class (if args.by_freq)
        #create neuron subsets
        nice_subset, random_subset = neuron_choice(
            args, NAME_TO_COMBO[subset_name], subset=N_NEURONS
        )
        if nice_subset is None:#too few neurons in class
            continue
        if not os.path.exists(f"{save_path_clear}/generated_{temperature_str}.txt"):
            generate_and_save(
                save_path=save_path_clear,
                file_name=f"generated_{temperature_str}.txt",
                args=args,
                model=model,
                neuron_subset=nice_subset,
                mean_values=mean_values,
                **generate_kwargs,
            )
        if not os.path.exists(f"{save_path_baseline}/generated_{temperature_str}.txt"):
            generate_and_save(
                save_path=save_path_baseline,
                file_name=f"generated_{temperature_str}.txt",
                args=args,
                model=model,
                neuron_subset=random_subset,
                mean_values=mean_values,
                **generate_kwargs,
            )
