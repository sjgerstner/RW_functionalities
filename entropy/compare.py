import os
import torch

#%%
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
        # print("number of nan entries in 'baseline' tensor:", baseline.isnan().sum().item())
        # print("number of inf entries in 'baseline' tensor:", baseline.isinf().sum().item())
        ablated = torch.log(ablated)
        # print("number of nan entries in 'ablated' tensor:", ablated.isnan().sum().item())
        # print("number of inf entries in 'ablated' tensor:", ablated.isinf().sum().item())
        #diff = baseline / ablated
    #else:
    diff = baseline - ablated
    # print("number of nan entries in 'diff' tensor:", diff.isnan().sum())
    return diff

def compute_data(data_path, metric, neuron_subset_name, intervention_type='zero_ablation'):
    diff = unflattened_data(data_path, metric, neuron_subset_name, intervention_type)
    diff_flattened = diff.flatten()
    #remove zeros (or nans), corresponding to padding
    keep_or_not = (diff_flattened!=0) & ~(diff_flattened.isnan())
    diff_nonzero = diff_flattened[keep_or_not.nonzero()]
    #print("number of nan values in diff_nonzero:", diff_nonzero.isnan().sum().item())
    return diff_nonzero.cpu().numpy()

def compare(args, metric, neuron_subset_names, intervention_type='zero_ablation'):
    data_dir = args.data_dir if args.data_dir is not None else os.environ["WORK"]+'/RW_functionalities_results'
    data_path = f'{data_dir}/intervention_results/{args.model}/{args.dataset}'
    print('computing data...')
    diffs = {}
    baseline_names=[]
    for neuron_subset_name in neuron_subset_names:
        if not os.path.exists(os.path.join(data_path, neuron_subset_name)):
            print(neuron_subset_name, "not found")
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
