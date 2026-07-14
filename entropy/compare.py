import os

import torch

def unflattened_data(data_path, metric, neuron_subset_name, intervention_type='zero_ablation')->torch.Tensor:
    #print('loading data...')
    baseline = torch.load(
        f'{data_path}/baseline/None_None/{metric}.pt',
        weights_only=True,
        map_location='cuda:0' if torch.cuda.is_available() else 'cpu',
    )
    ablated = torch.load(
        f'{data_path}/{neuron_subset_name}/{intervention_type}_None/{metric}.pt',
        weights_only=True,
        map_location='cuda:0' if torch.cuda.is_available() else 'cpu',
    )#sample pos
    if baseline.shape[0]!=ablated.shape[0]:
        #for some reason an earlier version of dolma-small had 45734 rows
        # while the new version has 45736.
        # But (I'm pretty sure) the rows are in the same order,
        # so it should be possible to just remove the last two from the longer version
        min_shape = min(baseline.shape[0], ablated.shape[0])
        baseline = baseline[:min_shape]
        ablated = ablated[:min_shape]
    #print('computing difference...')
    if metric=='scale':
        baseline = torch.log(baseline)
        ablated = torch.log(ablated)
        #diff = baseline / ablated
    #else:
    diff = baseline - ablated
    return diff

def compute_data(data_path, metric, neuron_subset_name, intervention_type='zero_ablation'):
    diff = unflattened_data(data_path, metric, neuron_subset_name, intervention_type)
    diff_flattened = diff.flatten()
    #remove zeros, corresponding to padding
    diff_nonzero = diff_flattened[diff_flattened.nonzero()].cpu().numpy()
    return diff_nonzero

def compare(args, metric, neuron_subset_names, intervention_type='zero_ablation'):
    data_dir = args.data_dir if args.data_dir is not None else os.environ["WORK"]+'/RW_functionalities_results'
    data_path = f'{data_dir}/intervention_results/{args.model}/{args.dataset}'
    print('computing data...')
    diffs = {}
    baseline_names=[]
    for neuron_subset_name in neuron_subset_names:
        if not os.path.exists(os.path.join(data_path, neuron_subset_name)):
            continue
        print(neuron_subset_name)
        diffs[neuron_subset_name] = compute_data(
            data_path, metric, neuron_subset_name, intervention_type
        )
        baseline_exists = os.path.exists(f'{data_path}/{neuron_subset_name}_baseline')
        if baseline_exists:
            baseline_name = neuron_subset_name+'_baseline'
            baseline_names.append(baseline_name)
            print(baseline_name)
            diffs[baseline_name] = compute_data(
                data_path, metric, baseline_name, intervention_type
            )
    return diffs, data_dir
