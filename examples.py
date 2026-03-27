from argparse import Namespace
from functools import partial
import os

import torch
import datasets
from transformer_lens import HookedTransformer, ActivationCache

from entropy.compare import unflattened_data
from ablation_utils.utils import make_hooks
from ablation_utils.generate import generate_and_save, generate_ablated
from neuron_choice import neuron_choice
from src.weight_analysis_utils.utils import NAME_TO_COMBO

DATA_DIR = "../RW_functionalities_results/intervention_results/allenai/OLMo-7B-0424-hf/dolma-small"

def store_activation_hook(activation, hook, cache_dict):
    cache_dict[hook.name] = activation.clone()

def _find_single_extremum(diff_data:torch.Tensor, extremum="min")->torch.Tensor:
    if extremum=="min":
        return torch.stack([torch.argmin(diff_data, dim=0), torch.argmin(diff_data, dim=1)])
    if extremum=="max":
        return torch.stack([torch.argmax(diff_data, dim=0), torch.argmax(diff_data, dim=1)])
    raise NotImplementedError

def _multidimensional_topk(x:torch.Tensor, topk:int, largest=True)->torch.Tensor:
    """_summary_

    Args:
        x (torch.Tensor): _description_
        topk (int): _description_
        largest (bool, optional): _description_. Defaults to True.

    Returns:
        torch.Tensor: of shape (k, *x.shape), each row contains the complete index of one of the top-k entries
    """
    raw_topk = torch.topk(x.flatten(), k=topk, largest=largest).indices
    out=[]
    r = raw_topk
    for dim in range(x.ndim):
        q, r = r//(x.shape[dim]), r%(x.shape[dim])
        out.append(q)
    return torch.stack(out, dim=-1)

def find_texts(
    intervention_type = "zero_ablation",
    metric = "entropy",
    neuron_subset_name = "weakening",
    data_dir = DATA_DIR,
    extremum = "min",
    topk = 1,
)-> torch.Tensor:
    diff_data = unflattened_data(data_dir, metric, neuron_subset_name, intervention_type)
    if topk==1:
        return _find_single_extremum(diff_data, extremum)
    if extremum=="min":
        return _multidimensional_topk(diff_data, topk, largest=False)
    if extremum=="max":
        return _multidimensional_topk(diff_data, topk, largest=True)
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
            mean_value=0.0,#TODO mean values
        ))
    for layer in range(model.cfg.n_layers):
        for sublayer in ['mid', 'post']:
            hooks.append((
                f'blocks.{layer}.hook_resid_{sublayer}',
                partial(store_activation_hook, cache_dict=cache_ablated)
            ))
    #results.append(hooks)

    logits_ablated = model.run_with_hooks(input_ids, fwd_hooks=hooks)
    return logits_ablated, cache_ablated

def inspect_logit_diff(model, logits_clean, logits_ablated, pos):

    logit_diff = logits_clean - logits_ablated

    relevant_logit_diff = logit_diff[0,pos,:]

    top = torch.topk(relevant_logit_diff, k=16)
    bottom = torch.topk(relevant_logit_diff, k=16, largest=False)

    results = []
    results.append("top tokens promoted by the neurons (overall effect):")
    results.append(model.to_str_tokens(top.indices))

    results.append("top tokens suppressed by the neurons (overall effect):")
    results.append(model.to_str_tokens(bottom.indices))
    return "\n".join(results)

def analyze_hidden_states(model:HookedTransformer, cache:ActivationCache|dict[str,torch.Tensor]):
    results = []
    for layer in range(model.cfg.n_layers):
        for sublayer in ['mid', 'post']:
            results.append('===========================')
            results.append(f'Layer {layer}, {sublayer}:')
            top_token_vi = torch.topk(
                cache[f'blocks.{layer}.hook_resid_{sublayer}'][0,25]@model.W_U,
                k=4
            )
            for j in range(4):
                results.append((model.to_str_tokens(top_token_vi.indices[j]), top_token_vi.values[j]))
    return "\n".join(results)

def show_single_text(
    args:Namespace,
    model:HookedTransformer,
    input_ids:torch.Tensor,
    pos:int,
    neuron_list:list,
    return_cache=False,
    max_new_tokens=1,
):
    results = []
    results.append("Input tokens:")
    results.append(model.to_str_tokens(input_ids[:pos+1]))
    results.append("Ground-truth output token:")
    results.append(model.to_single_str_token(input_ids[pos+1].item()))
    if max_new_tokens>1:
        results.append("Ground-truth continuation:")
        results.append(model.to_str(input_ids[pos+1:pos+1+max_new_tokens]))

    # ## Running the model on the example
    logits_clean, cache_clean = model.run_with_cache(input_ids[:pos+1])
    results.append("\nThe clean model outputs:")
    argmax_token_clean = torch.argmax(logits_clean[0,pos])
    results.append(model.to_single_str_token(argmax_token_clean))
    results.append("with score", logits_clean[0,pos,argmax_token_clean])
    if max_new_tokens>1:
        results.append("The clean model generates (greedily):")
        results.append(model.generate(input_ids[:pos+1], max_new_tokens=max_new_tokens, do_sample=False))

    #same with ablated model
    logits_ablated, cache_ablated = run_ablated_and_cache(args, model, input_ids, neuron_list)
    results.append("\nThe ablated model outputs:")
    argmax_token_ablated = torch.argmax(logits_ablated[0,pos])
    results.append(model.to_single_str_token(argmax_token_ablated))
    results.append("with score", logits_ablated[0,pos,argmax_token_ablated])
    if max_new_tokens>1:
        results.append("The ablated model generates (greedily):")
        results.append(generate_ablated(
            args=args,
            model=model,
            neuron_subset=neuron_list,
            prompt=input_ids[:pos+1],
            do_sample=False,
            mean_values=None,#TODO
        ))

    # ## Which tokens were boosted or suppressed?
    results.append(inspect_logit_diff(model, logits_clean, logits_ablated, pos=pos))
    result_str = "\n".join(results)
    if return_cache:
        return result_str, cache_clean, cache_ablated
    return result_str, None, None

def show_texts(
    model:HookedTransformer,
    text_dataset,
    output_path,
    **kwargs
):
    indices = find_texts(**kwargs)#first column is index in dataset, second column is position
    my_input_ids = text_dataset['input_ids'][indices[:,0]]

    # ## List of relevant neurons
    #TODO this is super ugly, so change the find_neurons and run_ablated_and_cache functions to admit **kwargs
    args = create_args(**kwargs)
    neuron_list = find_neurons(args)

    for i in range(kwargs["topk"]):
        result_str, _cache_clean, _cache_ablated = show_single_text(
            args=args,
            model=model,
            input_ids=my_input_ids,
            pos=indices[i,1].item(),
            neuron_list=neuron_list,
            return_cache=i==0,
        )
        result_str+="\n\n"
        if _cache_clean:
            cache_clean, cache_ablated = _cache_clean, _cache_ablated
    return result_str, cache_clean, cache_ablated

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
        data_dir = os.path.join(
            DATA_DIR,
            kwargs["neuron_subset_name"],
            f"{kwargs["intervention_type"]}_None",
        )
        output_path = os.path.join(data_dir, f"{kwargs["metric"]}.txt")
        results = []
        #First inspect top-k (say 4) texts with clean and ablated predictions and logit diff
        result_str, cache_clean, cache_ablated = show_texts(model, text_dataset, **kwargs)
        results.append(result_str)
        #Then look at the top text in more detail by analyzing hidden states
        results.append(analyze_hidden_states(model, cache_clean))
        results.append(analyze_hidden_states(model, cache_ablated))
        #TODO (for top text)
        #definitely:
        # - finding relevant neurons and/or token positions for a given effect:
        # use EAP or similar to find neuron-position pairs that are likely to be relevant,
        # and then ablate the top-n ones to check
        # (fine-grained ablations as in 2025-12-17-final_layer.ipynb,
        #       including ablating several neurons/positions together).
        #for the record (probably not necessary):
        # - is there a single neuron *directly* responsible for a given effect? -> logit lens on w_out of neurons as in entropy_example.ipynb
        # - inspecting direct path (are the neurons jointly directly responsible) as in 2025-12-18-paths.ipynb
        with open(output_path, 'a', encoding='utf-8') as output_file:
            output_file.write("\n".join(results))

#%%
if __name__=="__main__":
    main()#TODO list of kwargs dicts
    # kwargs should contain at least:
    # neuron_subset_name, intervention_type, other args of find_texts, topk (i.e. top-k texts), metric (e.g. entropy),
    # and the args properties (other than directories, model_name, activation_location) of find_neurons and show_single_text
