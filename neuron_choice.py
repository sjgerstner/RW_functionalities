#TODO save neuron choices to avoid recomputing

from os.path import exists
#import pickle
import random
import torch

from utils import COMBO_TO_NAME, is_in_category, VANILLA_CATEGORIES

def _key_to_name(category_key):
    if isinstance(category_key, tuple):
        return COMBO_TO_NAME[category_key]
    else:
        return VANILLA_CATEGORIES[category_key]

def neuron_choice(args, category_key, subset=None, baseline=True):
    """
    Create a list of (some) neurons of the given category,
    plus possibly a random baseline.

    Args:
        args: arguments from argument parser
        category_key (tuple or int): key for COMBO_TO_NAME or for the vanilla version
        d_mlp (int):
            number of neurons per layer of the model. Necessary when creating a random baseline.
        subset (None or float or int, default None): how many neurons to choose from the category.
            If None (default): take all neurons from the category.
            If float (should be between 0. and 1.): proportion from the category
            If int: absolute numbers
        random_baseline (bool):
            create a random list of neurons from the same layers but not in the given category.
            Default True.

    Returns:
        neuron_list: list of neurons formatted as (layer,neuron) tuples
        other_neuron_list (optional): list of baseline neurons in the same format
    """
    #category_index = CATEGORY_NAMES.index(category_name)
    random.seed(2512800)
    path = f"{args.work_dir}/{args.wcos_dir}/results/{args.model}"
    # with open(f"{path}/data.pickle", 'rb') as f:
    #     data = pickle.load(f)
    data_path = f"{path}/data.pt"
    if not exists(data_path):
        data_path = f"{path}/refactored/data.pt"
    data = torch.load(data_path)
    neuron_tensor = torch.nonzero(is_in_category(data['categories'],category_key))
    neuron_list = [tuple(line) for line in neuron_tensor]
    if subset is not None:
        if isinstance(subset, float):
            assert 0<subset<=1
            subset = int(subset*len(neuron_list))
        if subset<len(neuron_list):
            neuron_list = random.sample(neuron_list, subset)
        elif subset>len(neuron_list):
            category_name = _key_to_name(category_key)
            print(f"Warning: category {category_name} only contains {len(neuron_list)} neurons.")
            return None, None
    if baseline:
        #TODO adapt baseline to activation frequencies
        baseline_list = random_baseline(neuron_list, data['categories'], category_key)
        return neuron_list, baseline_list
    return neuron_list

def random_baseline(neuron_list, data_categories, category_key):
    """Generate a random list of neurons with the following requirements:
    - no overlap with neuron_list
    - no overlap with the RW class that neuron_list is taken from
    - from each layer take the same number of neurons as neuron_list has in that layer;
    if neuron_list contains more than half of all neurons of a given layer, just take all the others
    
    Alternative: take neurons from neighbouring layers? But this may not work either...

    Args:
        neuron_list (list of tuples):
            the neurons, represented as (layer, neuron), to generate a baseline for
        data_categories (tensor of ints):
            tensor of ints, of shape (layer, neuron),
            with entries representing the category of each neuron (see CATEGORY_NAMES)
        category_key (tuple): key of the current category

    Returns:
        list of tuples: the baseline neurons
    """
    n_layers = data_categories.shape[0]
    d_mlp = data_categories.shape[1]
    layer_counts = [0]*n_layers
    for layer,_neuron in neuron_list:
        layer_counts[layer]+=1
    baseline_list = []
    for layer,number in enumerate(layer_counts):
        if number==0:
            continue
        baseline_sublist = list(torch.nonzero(~is_in_category(data_categories[layer],category_key)))
        if number<d_mlp/2:
            baseline_sublist = random.sample(baseline_sublist, number)
        else:
            category_name = _key_to_name(category_key)
            print(
                f"Warning: category {category_name} covers more than half of neurons",
                f"in layer {layer} ({number} of {d_mlp})"
            )
        baseline_list.extend([(layer,neuron.item()) for neuron in baseline_sublist])
    return baseline_list

# def random_baseline_old(layer, category_index, data, other_neurons, d_mlp):
#     """Unused!"""
#     other_neuron = random.randint(0, d_mlp-1)
#     if (data['categories'][layer,other_neuron]==category_index) or ((layer,other_neuron) in other_neurons):
#         other_neuron = random_baseline_old(layer, category_index, data, other_neurons, d_mlp)
#     return other_neuron
