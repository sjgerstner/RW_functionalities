"""
Modified from Geva et al. 2023, Dissecting recall:
https://github.com/google-research/google-research/tree/master/dissecting_factual_predictions

Copyright 2023 Google LLC.
Licensed under the Apache License, Version 2.0 (the "License").

Modifications:
Jupyter to Python script,
only subject enrichment code and only for final subject position,
using transformer_lens,
more fine-grained ablations.

Run from the repo's main dir: python -m attributes.enrichment
"""

from argparse import ArgumentParser
from math import isnan
import os
# Scientific packages
import numpy as np
import nltk
from nltk.corpus import stopwords
import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
# Utilities
from attributes.utils import (
    find_token_range,
    record_logitlens,
    decode_tokens,
)
from neuron_choice import neuron_choice, get_n_neurons
from utils import NAME_TO_COMBO
from entropy.entropy_intervention_wrap import get_mean_values

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('--work_dir', default='.')
    parser.add_argument('--wcos_dir', default='.')
    parser.add_argument('--wiki_dir', default='wiki_data')
    parser.add_argument('--means_path', default='neuroscope/results/OLMo-7B-0424/summary_refactored.pt')
    parser.add_argument('--model', default='allenai/OLMo-7B-0424-hf')
    #parser.add_argument('--subject_repr_layer', default=40)
    #parser.add_argument('--num_block_layers', default=10)
    #parser.add_argument('--paragraphs_data_path', default=None)
    parser.add_argument('--topk', default=50)
    parser.add_argument(
        '--intervention_type',
        choices=["zero_ablation", "threshold_ablation", "fixed_activation", "relu_ablation", "mean_ablation"],
        default="zero_ablation",
    )
    parser.add_argument('--activation_location', default='mlp.hook_post')
    parser.add_argument(
        '--n_neurons', default=None,
        help="""(None or float or int, default None): how many neurons to choose from the category.
            If None (default): take all neurons from the category.
            If float (should be between 0. and 1.): proportion from the category
            If int: absolute numbers"""
    )
    parser.add_argument(
        '--by_freq', default=None,
        help="""a metric for activation frequency, to adapt the number of neurons to ablate
        (ablate more neurons for less frequently activated classes)"""
    )
    parser.add_argument(
        '--constant_class', default='weakening',
        help="The class from which to ablate n_neurons"
    )
    parser.add_argument(
        '--gate', default=None,
        help="ablate only when x_gate has this sign ('+' or '-')"
    )
    parser.add_argument(
        '--post', default=None,
        help="ablate only when activation*cos(w_gate,w_in) has this sign ('+' or '-')"
    )
    parser.add_argument(
        '--device', default=torch.device('cuda' if torch.cuda.is_available() else (
            'mps' if torch.backends.mps.is_available() else 'cpu')), type=str,
    )
    parser.add_argument(
        '--subsets',
        nargs='+',
        default=[
            "strengthening",
            "conditional strengthening",
            "proportional change",
            "weakening",
            "conditional weakening",
        ]
    )
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    tqdm.pandas()

    # List of stopwords from NLTK, needed only for the attributes rate evaluation.
    nltk.download('stopwords')
    stopwords0_ = stopwords.words('english')
    stopwords0_ = {word: "" for word in stopwords0_}

    # %%
    #get mean activations for each neuron if we're using them later
    if args.intervention_type=='mean_ablation':
        mean_values = get_mean_values(args)
    else:
        mean_values = None

    # %%
    model = HookedTransformer.from_pretrained(args.model)
    short_model_name = args.model.split('/')[-1]
    knowns_df = pd.read_json(f'{args.work_dir}/knowns/known_{short_model_name}.json')

    # %%
    OUT_DIR = f'{args.work_dir}/se'

    # %% [markdown]
    # ## Subject enrichment
    # ### Get token representations' projections

    #%%
    #get embeddings and unembeddings
    E = model.W_E.detach()
    U = model.W_U.detach()

    # %%
    # Projection of token embeddings
    DF_PATH = f'{OUT_DIR}/logitlens.pickle'
    if not os.path.exists(DF_PATH):
        records = []
        for row_i, row in tqdm(knowns_df.iterrows()):
            subject = row.subject
            prompt = row.prompt
            #prompt = "<|endoftext|> " + prompt  # fix first-position bias

            inp = model.to_tokens(prompt)[0]
            try:
                e_range = find_token_range(model.tokenizer, inp, subject)
            except ValueError as e:
                print(inp)
                print(decode_tokens(model.tokenizer, inp))
                print(subject)
                raise e
            # e_range = list(range(e_range[0], e_range[1]))
            # subject_tok = [inp["input_ids"][0][i].item() for i in e_range]
            subject_tok = inp[e_range[0]:e_range[1]]
            subject_tok_str = model.to_str_tokens(subject_tok)

            vec = E[subject_tok, :].mean(axis=0)
            proj = vec.matmul(U).cpu().numpy()
            ind = np.argsort(-proj)
            record = {
                "example_index": row_i,
                "prompt": row.prompt,
                "subject": subject,
                "subject_tok": subject_tok,
                "subject_tok_str": str(subject_tok_str),
                "top_k_preds_str": model.to_str_tokens(ind[:args.topk]),
                "intervention_type": None,
            }
            records.append(record)
        tmp = pd.DataFrame.from_records(records)
        tmp.to_pickle(DF_PATH)
    else:
        tmp = pd.read_pickle(DF_PATH)

    # %%
    if "layer" not in tmp.columns:
        baseline_df = record_logitlens(
            args, knowns_df, model,
        )
        tmp = pd.concat([tmp, baseline_df], ignore_index=True)
        tmp.to_pickle(DF_PATH)

    N_NEURONS, CONSTANT = get_n_neurons(args)
    for subset_name in args.subsets:
        clear_subset_name = f'{subset_name}{N_NEURONS}' if N_NEURONS else subset_name
        baseline_name=f'{clear_subset_name}_baseline'
        test1 = ((tmp["neuron_subset_name"]!=clear_subset_name) | (tmp["intervention_type"]!=args.intervention_type)).all()
        test2 = ((tmp["neuron_subset_name"]!=baseline_name) | (tmp["intervention_type"]!=args.intervention_type)).all()
        if test1 or test2:
            #TODO number of neurons to ablate from this class
            #create neuron subsets
            nice_subset, random_subset = neuron_choice(
                args, NAME_TO_COMBO[subset_name], subset=N_NEURONS
            )
            if nice_subset is None:#too few neurons in class
                continue
            # Projection of token representations while applying knockouts to, say, strengthening neurons
            if test1:
                nice_df = record_logitlens(
                    args, knowns_df, model,
                    neuron_subset=nice_subset, neuron_subset_name=clear_subset_name,
                    mean_values = mean_values,
                )
                tmp = pd.concat([tmp, nice_df], ignore_index=True)
                tmp.to_pickle(DF_PATH)
            # Projection of token representations while applying knockouts to random neurons
            if test2:
                random_baseline_df = record_logitlens(
                    args, knowns_df, model,
                    neuron_subset=random_subset, neuron_subset_name=baseline_name,
                    mean_values = mean_values,
                )
                tmp = pd.concat([tmp, random_baseline_df], ignore_index=True)
                tmp.to_pickle(DF_PATH)

    # #%%
    # #Full dataframe
    # tmp = pd.concat(
    #     df_list,
    #     ignore_index=True
    # )

    # %% [markdown]
    # ### Prepare attributes rate evaluation

    # %%
    # Processing of Wikipedia paragraphs for automatic attribute rate evaluation.

    WIKI_CLEANED = f'{args.work_dir}/{args.wiki_dir}/wiki_cleaned.pickle'
    if not os.path.exists(WIKI_CLEANED):
        # This should be a path to a csv file
        # with 2 columns and a header of column names "subject" and "paragraphs".
        # Each entry should have (a) a subject (string) from the "knowns" data (knowns_df)
        # and (b) paragraphs concatenated with space about the subject (a single string).
        df_wiki = pd.read_csv(f'{args.work_dir}/{args.wiki_dir}/wiki.csv')
        df_wiki = df_wiki.fillna('')
        # Tokenize, remove duplicate tokens, stopwords, and subwords.
        df_wiki["context_tokenized_dedup"] = df_wiki["paragraphs"].progress_apply(
            lambda x: list(set(model.to_str_tokens(x)))
        )
        df_wiki["context_tokenized_dedup_len"] = df_wiki.context_tokenized_dedup.apply(
            len#lambda x: len(x)
        )
        df_wiki["context_tokenized_dedup_no-stopwords"] = df_wiki.context_tokenized_dedup.apply(
            lambda x: [
                y for y in x
                if y.strip() not in stopwords0_ and len(y.strip())>2
            ]
        )
        df_wiki["context_tokenized_dedup_no-stopwords_len"] = df_wiki["context_tokenized_dedup_no-stopwords"].apply(
            len#lambda x: len(x)
        )
        df_wiki.to_pickle(WIKI_CLEANED)
    else:
        df_wiki = pd.read_pickle(WIKI_CLEANED)


    # %%
    def get_preds_wiki_overlap(subject, top_preds):
        wiki_toks = df_wiki[df_wiki.subject == subject]
        if len(wiki_toks) == 0 or len(top_preds)==0:
            return -1
        wiki_toks = wiki_toks.iloc[0]["context_tokenized_dedup_no-stopwords"]
        preds_wiki_inter = set(top_preds).intersection(set(wiki_toks))

        return len(preds_wiki_inter) * 100.0 / len(top_preds)

    # %% [markdown]
    # ### Evaluate attributes rate

    # %%
    if "top_k_preds_clean" not in tmp.columns:
        tmp["top_k_preds_clean"] = np.nan
    tmp["top_k_preds_clean"] = tmp.top_k_preds_str.progress_apply(lambda x: [
        y for y in x
        if y.strip().lower() not in stopwords0_ and len(y.strip())>2
    ])
    tmp.to_pickle(DF_PATH)
    if "num_clean_tokens" not in tmp.columns:
        tmp["num_clean_tokens"] = np.nan
    tmp["num_clean_tokens"] = tmp.top_k_preds_clean.progress_apply(
        len#lambda x: len(x)
    )
    tmp.to_pickle(DF_PATH)

    #%%
    if "top_k_preds_in_context" not in tmp.columns:
        tmp["top_k_preds_in_context"] = np.nan
    tmp["top_k_preds_in_context"] = tmp.progress_apply(
        lambda row: get_preds_wiki_overlap(
            row["subject"], row["top_k_preds_clean"][:args.topk]
        ) if isnan(row["top_k_preds_in_context"]) else row["top_k_preds_in_context"],
        axis=1
        )
    tmp.to_pickle(DF_PATH)

    print("Percentage of entries with no known attributes:")
    print(len(tmp[tmp.top_k_preds_in_context == -1]) * 100.0 / len(tmp))
    print("Number of unique subject with known attributes:")
    print(tmp[tmp.top_k_preds_in_context > -1].subject.nunique())

    # if "layer_1" not in tmp.columns:
    #     tmp["layer_1"] = tmp.layer.apply(lambda x: x+1)
    #     tmp.to_pickle(DF_PATH)
