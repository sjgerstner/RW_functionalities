#!/bin/bash
set -euox pipefail

python main.py \
    --experiments beta categories category_stats plot_fine plot_coarse \
    --model \
        "gpt2-small" \
        "gpt2-medium" \
        "gpt2-large" \
        "gpt2-xl" \
        "distilgpt2" \
        "opt-125m" \
        "opt-1.3b" \
        "opt-6.7b" \
        "opt-13b" \
        "gpt-j-6B" \
        "pythia-14m" \
        "pythia-1b" \
        "pythia-6.9b" \
        "pythia-12b" \
        "bloom-560m" \
        "bloom-1b1" \
        "bloom-1b7" \
        "bloom-7b1" \
        "bert-base-cased" \
        "bert-large-cased" \
        "t5-small" \
        "t5-base" \
        "t5-large" \
        "othello-gpt"
