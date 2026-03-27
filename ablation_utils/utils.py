from functools import partial

import torch

from .ablation_hooks import (
    threshold_ablation_hook,
    relu_ablation_hook,
    fixed_activation_hook,
    multiply_activation_hook,
)

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

def make_hooks(args, layer, neuron, conditioning_value=None, sign=1, mean_value:float=0.0, positions:torch.Tensor|None=None):
    """Adapted from Gurnee et al.

    Args:
        args (_type_): _description_
        layer (_type_): _description_
        neuron (_type_): _description_
        conditioning_value (_type_, optional): _description_. Defaults to None.
        sign (int, optional): _description_. Defaults to 1.
        mean_value (float, optional): _description_. Defaults to 0.0.
        positions (torch.Tensor | None, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if conditioning_value is None:
        conditioning_value = {}
    out = []

    if args.gate is not None:
        hook_loc = f'blocks.{layer}.mlp.hook_pre'
        hook_fn = partial(
            store_conditioning_hook,
            conditioning_values=conditioning_value, neuron=neuron
        )
        out.append((hook_loc, hook_fn))
    if args.post is not None:
        hook_loc = f'blocks.{layer}.mlp.hook_post'
        hook_fn = partial(
            store_conditioning_hook,
            conditioning_values=conditioning_value, neuron=neuron,
            sign=sign,
        )
        out.append((hook_loc, hook_fn))

    hook_fn = partial(
        conditional_ablation_hook,
        conditioning_value=conditioning_value,
        args=args, neuron=neuron, mean_value=mean_value,
        positions=positions)

    hook_loc = f'blocks.{layer}.{args.activation_location}'
    out.append((hook_loc, hook_fn))

    return out


def store_conditioning_hook(activation, hook, conditioning_values, neuron=None, sign=1):
    """Store the conditioning activation"""
    if neuron is not None:
        conditioning_values[hook.name] = activation[:,:,neuron].clone()
        if sign<0:
            conditioning_values[hook.name] *=-1
    else:
        conditioning_values[hook.name] = activation.clone()


def conditional_ablation_hook(
    activation:torch.Tensor,
    hook,
    args,
    conditioning_value:dict[str, torch.Tensor],
    neuron=None,
    mean_value:float=0.0,
    positions:torch.Tensor|None=None
):
    """Ablate based on conditioning activation"""
    hook_loc = '.'.join(hook.name.split('.')[:-1])
    gate_loc = hook_loc+'.hook_pre'
    post_loc = hook_loc+'.hook_post'
    condition = torch.ones(activation.shape[:-1], dtype=torch.bool, device=activation.device)
    if gate_loc in conditioning_value:
        if args.gate=='+':
            condition = conditioning_value[gate_loc]>0
        elif args.gate=='-':
            condition = conditioning_value[gate_loc]<0
        else:
            raise NotImplementedError("argument 'gate' has to be + or -")
    if post_loc in conditioning_value:
        # if condition is None:
        #     condition = torch.ones_like(conditioning_value[post_loc])
        if args.post=='+':
            condition = torch.logical_and(condition, conditioning_value[post_loc]>0)
        elif args.post=='-':
            condition = torch.logical_and(condition, conditioning_value[post_loc]<0)
        else:
            raise NotImplementedError("argument 'post' has to be + or -")
    #optionally, ablate only at given positions
    if positions is not None:
        position_mask = torch.zeros(activation.shape[:-1],dtype=torch.bool, device=activation.device)
        position_mask[:,positions]=True
        condition = torch.logical_and(condition, position_mask)
    activation = ablation_hook(activation, hook, args, neuron=neuron, mask=condition, mean_value=mean_value)
    return activation


def ablation_hook(activation, hook, args, neuron=None, mask=None, mean_value:float=0.0):
    """Branch to the right intervention type based on args.intervention_type"""
    if mask is None:
        mask=torch.ones(size=activation.shape[:-1], dtype=torch.bool)
    if args.intervention_type == 'zero_ablation':
        # if mask is not None:
        #     activation = masked_zero_ablation_hook(activation,hook,neuron,mask)
        #else:
        activation = fixed_activation_hook(activation, hook, neuron=neuron, mask=mask, fixed_act=0)
    elif args.intervention_type == 'mean_ablation':
        activation = fixed_activation_hook(activation, hook, neuron=neuron, mask=mask, fixed_act=mean_value)
    elif args.intervention_type == 'fixed_activation':
        activation = fixed_activation_hook(
            activation, hook,
            neuron=neuron,
            fixed_act=args.intervention_param
        )
    #TODO implement mask for the other intervention types
    elif args.intervention_type == 'threshold_ablation':
        activation = threshold_ablation_hook(
            activation, hook, neuron=neuron,
            threshold=args.intervention_param
        )
    elif args.intervention_type == 'relu_ablation':
        activation = relu_ablation_hook(activation, hook, neuron=neuron)

    elif args.intervention_type == 'multiply_activation':
        activation = multiply_activation_hook(
            activation, hook,
            neuron=neuron,
            multiplier=args.intervention_param)
    else:
        raise ValueError(
            f'Unknown intervention type: {args.intervention_type}')
    return activation

#inspired by Gurnee et al., Universal Neurons
#new but boring:
# def make_layer_hooks(args, layer_list):
#     hooks = []
#     for layer in layer_list:
#         hooks += make_hooks(args, layer)
#     return hooks
def make_neuron_hooks(args, neuron_list, mean_values:torch.Tensor|None=None):
    hooks = []
    if neuron_list is not None:
        for layer,neuron in neuron_list:
            hooks.extend(make_hooks(
                args, layer, neuron,
                mean_value=mean_values[layer,neuron].item() if mean_values is not None else 0,
            ))
    return hooks
