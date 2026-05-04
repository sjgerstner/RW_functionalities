"""
All of these functions are copied from the file intervention.py in https://github.com/wesg52/universal-neurons,
except fixed_activation_hook() which is largely inspired by the other functions.

Accordingly, the copyright and license from the original repo apply:

MIT License

Copyright (c) 2023 Wes Gurnee
"""

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
    if mask is None:
        mask = torch.ones(size=activations.shape[:-1], dtype=torch.bool)
    if neuron is not None:
        activations[:,:, neuron][mask] = fixed_act
    else:
        activations[:][mask] = fixed_act
    return activations
