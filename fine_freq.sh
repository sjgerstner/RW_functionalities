#!/bin/bash
set -euox pipefail

for combo in "summary" "gate+_in+" "gate+_in-" "gate-_in+" "gate-_in-"; do
    python freqs.py \
    --refactor_glu \
    --combo $combo \
    --subexperiments scatter_plots selected all_layers norms \
    --layer_list 15 \
    "$@"
done
