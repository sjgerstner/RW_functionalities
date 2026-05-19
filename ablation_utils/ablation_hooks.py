"""
All of these functions are copied from the file intervention.py in https://github.com/wesg52/universal-neurons,
except fixed_activation_hook() which is largely inspired by the other functions.

Accordingly, the copyright and license from the original repo apply:

MIT License

Copyright (c) 2023 Wes Gurnee
"""

#TODO (low prio): extend other hooks similarly to fixed_activation_hook

import torch

def multiply_activation_hook(activations, hook, neuron, multiplier=1):
    activations[:, :, neuron] = activations[:, :, neuron] * multiplier
    return activations

def threshold_ablation_hook(activations, hook, neuron, threshold=0):
    activations[:, :, neuron] = torch.min(
        activations[:, :, neuron],
        threshold * torch.ones_like(activations[:, :, neuron])
    )
    return activations

def relu_ablation_hook(activations, hook, neuron):
    activations[:, :, neuron] = torch.relu(activations[:, :, neuron])
    return activations

def fixed_activation_hook(activations, hook, neuron=None, fixed_act:float=0.0, mask=None):
    try:
        if mask is None:
            mask = torch.ones(
                size=activations[...,neuron].shape if neuron is not None else activations.shape,
                dtype=torch.bool
            )
        if isinstance(neuron, int):
            activations[:,:, neuron][mask] = fixed_act
        elif isinstance(neuron, (list, torch.Tensor)):
            for i,n in enumerate(list(neuron)):
                if isinstance(fixed_act, (list, torch.Tensor)):
                    fixed_act_item=fixed_act[i].item()
                else:
                    fixed_act_item = fixed_act
                activations[:,:,n][mask[...,i]] = fixed_act_item
        else:
            activations[:][mask] = fixed_act
        return activations
    except RuntimeError as e:
        raise RuntimeError(
            f"""{e}
            We are in hook {hook.name}, ablated neurons are {neuron}.
            Activations have shape {activations.shape}.
            Mask is of shape {mask.shape} with {mask.sum()} True entries.
            Fixed act is {fixed_act}."""
        ) from e
