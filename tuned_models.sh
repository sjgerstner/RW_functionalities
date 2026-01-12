#!/bin/bash
set -euox pipefail

python main.py \
    --refactor_glu \
    --experiments beta categories category_stats plot_fine plot_coarse \
    --model \
        "gemma-2-2b-it" \
        "gemma-2-9b-it" \
        "mistral-7b-instruct" \
        "qwen2.5-0.5b-instruct" \
        "qwen2.5-7b-instruct" \
        "yi-6b-chat"
