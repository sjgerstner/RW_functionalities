from argparse import Namespace
from functools import partial
import os

import torch
import datasets
from transformer_lens import HookedTransformer, ActivationCache

from ablation import generate_and_save
from entropy.compare import unflattened_data
from entropy.entropy_intervention import make_hooks
from neuron_choice import neuron_choice
from utils import NAME_TO_COMBO

def store_activation_hook(activation, hook, cache_dict):
    cache_dict[hook.name] = activation.clone()

def find_text(
    intervention_type = "zero_ablation",
    metric = "entropy",
    neuron_subset_name = "weakening",
    data_path = "intervention_results/allenai/OLMo-7B-0424-hf/dolma-small",
    extremum = "min",
)-> tuple[int,int]:
    diff_data = unflattened_data(data_path, metric, neuron_subset_name, intervention_type)
    if extremum=="min":
        return torch.argmin(diff_data, dim=0).item(), torch.argmin(diff_data, dim=1).item()
    if extremum=="max":
        return torch.argmax(diff_data, dim=0).item(), torch.argmax(diff_data, dim=1).item()
    raise NotImplementedError

def create_args(**kwargs):
    return Namespace(
        work_dir='.', wcos_dir='.',
        model='allenai/OLMo-7B-0424-hf',
        #gate='-', post='+',
        activation_location='mlp.hook_post',
        **kwargs,
    )

def find_neurons(args:Namespace)->list[tuple]:
    return neuron_choice(
        args,
        category_key=NAME_TO_COMBO[args.neuron_subset_name],
        subset=None,
        baseline=False,
    )

def run_ablated_and_cache(args:Namespace, model:HookedTransformer, input_ids:torch.Tensor, neuron_list:list[tuple]):
    cache_ablated = {}
    hooks = []
    #conditioning_values = {}
    for lix, nix in neuron_list:
        #conditioning_values[(lix,nix)]={}
        hooks.extend(make_hooks(
            args,
            lix, nix,
            #conditioning_value=conditioning_values[(lix,nix)],
            #sign=(model.W_gate[lix,:,nix]@model.W_in[lix,:,nix]).item(),
            mean_value=0.0,
        ))
    for layer in range(model.cfg.n_layers):
        for sublayer in ['mid', 'post']:
            hooks.append((
                f'blocks.{layer}.hook_resid_{sublayer}',
                partial(store_activation_hook, cache_dict=cache_ablated)
            ))
    #print(hooks)

    logits_ablated = model.run_with_hooks(input_ids, fwd_hooks=hooks)
    return logits_ablated, cache_ablated

def inspect_logit_diff(model, logits_clean, logits_ablated):

    logit_diff = logits_clean - logits_ablated

    relevant_logit_diff = logit_diff[0,pos,:]

    top = torch.topk(relevant_logit_diff, k=16)
    bottom = torch.topk(relevant_logit_diff, k=16, largest=False)

    print("top tokens promoted by the neurons (overall effect):")
    print(model.to_str_tokens(top.indices))

    print("top tokens suppressed by the neurons (overall effect):")
    print(model.to_str_tokens(bottom.indices))

def analyze_hidden_states(model:HookedTransformer, cache:ActivationCache|dict[str,torch.Tensor]):
    for layer in range(model.cfg.n_layers):
        for sublayer in ['mid', 'post']:
            print('===========================')
            print(f'Layer {layer}, {sublayer}:')
            top_token_vi = torch.topk(
                cache[f'blocks.{layer}.hook_resid_{sublayer}'][0,25]@model.W_U,
                k=4
            )
            for j in range(4):
                print((model.to_str_tokens(top_token_vi.indices[j]), top_token_vi.values[j]))

def inspect_text(
    model:HookedTransformer,
    text_dataset,
    **kwargs
):
    ds_index, pos = find_text(**kwargs)
    my_input_ids = text_dataset['input_ids'][ds_index]

    print("Input tokens:")
    print(model.to_str_tokens(my_input_ids[:pos+1]))
    print("Correct output token:")
    print(model.to_single_str_token(my_input_ids[pos+1].item()))

    # ## List of relevant neurons
    args = create_args(**kwargs)
    neuron_list = find_neurons(args)

    # ## Running the model on the example

    logits_clean, cache_clean = model.run_with_cache(my_input_ids)
    logits_ablated, cache_ablated = run_ablated_and_cache(args, model, my_input_ids, neuron_list)

    # ## Finding relevant tokens
    inspect_logit_diff(model, logits_clean, logits_ablated)
    #analyze hidden states
    analyze_hidden_states(model, cache_clean)
    analyze_hidden_states(model, cache_ablated)

def inspect_generations(
    args:Namespace,
    model:HookedTransformer,
    save_path:str|os.PathLike,
    mean_values,
    **generate_kwargs
):
    temperature_str = "full" if args.do_sample else "greedy"
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

#%%
def main(list_of_kwargs_dicts:list[dict]=[]):
    model = HookedTransformer.from_pretrained(
        'allenai/OLMo-7B-0424-hf',
        #refactor_glu=True
    )
    text_dataset = datasets.load_from_disk('neuroscope/datasets/dolma-small')
    for kwargs in list_of_kwargs_dicts:
        inspect_text(model, text_dataset, **kwargs)

#%%
if __name__=="__main__":
    main()#TODO list of kwargs dicts