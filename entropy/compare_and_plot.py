#%%
from argparse import ArgumentParser
import os

from weight_analysis_utils import plotting
from .compare import compare

def compare_and_plot(args, metric, neuron_subset_names, intervention_type='zero_ablation', **kwargs):
    diffs, data_dir = compare(args, metric=metric, neuron_subset_names=neuron_subset_names, intervention_type=intervention_type)
    list_data = list(diffs.values())
    subtitles = list(k.replace('_', ' ', 1) for k in diffs)
    experiment_dir = f'{data_dir}/{args.plot_dir}/{args.experiment_name}'
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir, exist_ok=True)
    if args.log:
        kwargs["log"]=True
    metric = "log_scale" if metric=='scale' else metric
    plotting.aligned_histograms(
        list_data,
        subtitles=subtitles,
        savefile=f'{experiment_dir}/{metric}{"_log" if args.log else ""}.pdf',
        suptitle = None,
        xlabel=f'{metric}(clean) - {metric}(ablated)',
        ncols = 2,
        **kwargs
    )

#%%
if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir',
        default=None,#'../RW_functionalities_results',
    )
    parser.add_argument('--plot_dir', default='plots/ablations')
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--model', default='allenai/OLMo-7B-0424-hf')
    parser.add_argument('--dataset', default='dolma-small')
    parser.add_argument('--metric', default='all')
    parser.add_argument(
        '--intervention_type',
        choices=[
            "zero_ablation",
            "threshold_ablation",
            "fixed_activation",
            "relu_ablation",
            "mean_ablation"
        ],
        default="zero_ablation",
    )
    parser.add_argument(
        '--log', type=bool, default=True, help="logarithmic y-axis in the histograms"
    )
    parser.add_argument('--neurons', nargs='+', default=['weakening'])
    args = parser.parse_args()
    #neuron_subset_names = [s.replace('_', ' ') for s in args.neurons]
    if "WORK" not in os.environ:
        os.environ["WORK"]='..'
    if args.metric=='all':
        for metric in ['entropy', 'loss', 'rank', 'scale']:
            print(metric)
            compare_and_plot(
                args, metric=metric, neuron_subset_names=args.neurons,
                intervention_type=args.intervention_type,
                #log=True
            )
    else:
        compare_and_plot(
            args, metric=args.metric, neuron_subset_names=args.neurons,
            intervention_type=args.intervention_type,
            #log=True
        )
