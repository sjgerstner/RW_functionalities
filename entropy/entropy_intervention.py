"""
Largely copied from Gurnee et al. 2024, Universal neurons:
https://github.com/wesg52/universal-neurons

MIT License

Copyright (c) 2023 Wes Gurnee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import os
#import time
import tqdm
import torch
#import einops
import datasets
import argparse
# import numpy as np
# import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import DataCollatorWithPadding
from transformer_lens import HookedTransformer
from transformer_lens.utils import lm_cross_entropy_loss

from ablation_utils.utils import make_hooks

from .utils import get_model_family
from .activations import get_correct_token_rank

def save_layer_norm_scale_hook(activations, hook):
    hook.ctx['activation'] = activations.detach().cpu()

def run_intervention_experiment(
    args,
    model,
    dataset,
    device,
    neuron_subset=None,
    neuron_subset_name=None,
    save_path=None,
    mean_values:torch.Tensor|None=None,
    ):

    if neuron_subset is None:
        neuron_subset = args.neuron_subset
        # if neuron_subset is None:
        #     return
    if args.intervention_type!='mean_ablation':
        mean_values = torch.zeros((model.cfg.n_layers, model.cfg.d_mlp))
    else:
        assert mean_values is not None
    print("> preparing hooks...")
    conditioning_values = {}
    hooks = []
    for lix, nix in neuron_subset:
        conditioning_values[(lix,nix)]={}
        hooks += make_hooks(
            args,
            lix, nix,
            conditioning_values[(lix,nix)],
            sign=(model.W_gate[lix,:,nix]@model.W_in[lix,:,nix]).item(),
            mean_value=mean_values[(lix,nix)].item(),
        )

    hooks.append(('ln_final.hook_scale', save_layer_norm_scale_hook))
    print(">done")

    #n, d = dataset['tokens'].shape
    n = dataset.num_rows
    d = max(len(example["input_ids"]) for example in dataset)
    #print(d)
    loss_tensor = torch.zeros(n, d, dtype=torch.float16)
    entropy_tensor = torch.zeros(n, d, dtype=torch.float16)
    rank_tensor = torch.zeros(n, d, dtype=torch.int32)
    scale_tensor = torch.zeros(n, d, dtype=torch.float16)

    print(">preparing data loading...")
    data_collator = DataCollatorWithPadding(
        tokenizer=model.tokenizer,
        # padding='max_length',
        # max_length=d,
    )
    dataloader = DataLoader(
        dataset,#dataset['tokens'],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    print(">done")

    offset = 0
    for step, batch in enumerate(tqdm.tqdm(dataloader)):
        batch = batch.to(device)
        logits = model.run_with_hooks(
            input=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            fwd_hooks=hooks
        )
        #print(logits.shape)
        bs, local_seq_len = batch["input_ids"].shape
        token_loss = lm_cross_entropy_loss(
            logits,
            batch["input_ids"],
            batch["attention_mask"],
            per_token=True
        ).cpu()
        probs = F.softmax(logits, dim=-1)
        entropies = (
            - torch.sum(probs * torch.log(probs + 1e-8), dim=-1)*batch["attention_mask"]
        ).cpu()
        token_ranks = (
            get_correct_token_rank(logits, batch["input_ids"])*batch["attention_mask"][:,:-1]
        ).cpu()

        loss_tensor[offset:offset+bs, :local_seq_len-1] = token_loss
        entropy_tensor[offset:offset+bs, :local_seq_len] = entropies
        rank_tensor[offset:offset+bs, :local_seq_len-1] = token_ranks

        scale = model.hook_dict['ln_final.hook_scale'].ctx['activation'].squeeze()
        scale_tensor[offset:offset+bs, :local_seq_len] = scale

        offset += bs

        model.reset_hooks()
    if not save_path:
        if not neuron_subset_name:
            neuron_subset_name = '_'.join([f'{l}.{n}' for l, n in neuron_subset])
        save_path = os.path.join(
            args.output_dir,
            args.model,
            args.token_dataset,
            neuron_subset_name,
            str(args.intervention_type)+'_'+str(args.intervention_param),
        )
    os.makedirs(save_path, exist_ok=True)
    torch.save(loss_tensor, os.path.join(save_path, f'loss.pt'))
    torch.save(entropy_tensor, os.path.join(
        save_path, f'entropy.pt'))
    torch.save(rank_tensor, os.path.join(save_path, f'rank.pt'))
    torch.save(scale_tensor, os.path.join(save_path, f'scale.pt'))


def parse_neuron_str(neuron_str: str):
    neurons = []
    for group in neuron_str.split(','):
        lix, nix = group.split('.')
        neurons.append((int(lix), int(nix)))
    return neurons


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # general arguments
    parser.add_argument(
        '--model', default='stanford-gpt2-small-a',
        help='Name of model from TransformerLens')
    parser.add_argument(
        '--token_dataset',
        help='Name of cached feature dataset')
    parser.add_argument(
        '--activation_location', default='mlp.hook_post',
        help='Model component to save')

    # activation processing/subsetting arguments
    parser.add_argument(
        '--batch_size', default=32, type=int)
    parser.add_argument(
        '--device', default=torch.device('cuda' if torch.cuda.is_available() else (
            'mps' if torch.backends.mps.is_available() else 'cpu')), type=str,
    )

    parser.add_argument(
        '--neuron_subset', type=parse_neuron_str, default=None,
        help='list of neurons')

    parser.add_argument(
        '--intervention_type', type=str, default='fixed_activation',
        help='Type of intervention to perform')
    parser.add_argument(
        '--intervention_param', type=float, default=None,
        help='Parameter for intervention type (eg, threshold or fixed activation)')

    # saving arguments
    parser.add_argument(
        '--save_precision', default=16, type=int)
    parser.add_argument(
        '--output_dir', default='../RW_functionalities_results/intervention_results')

    args = parser.parse_args()

    device = args.device

    model = HookedTransformer.from_pretrained(args.model, device='cpu')
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    model_family = get_model_family(args.model)

    tokenized_dataset = datasets.load_from_disk(
        os.path.join(
            os.getenv('DATASET_DIR', 'token_datasets'),
            model_family,
            args.token_dataset
        )
    )

    run_intervention_experiment(
        args, model, tokenized_dataset, device)
