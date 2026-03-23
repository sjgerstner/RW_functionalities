#!/bin/bash
set -euox pipefail

metric_type=${1:-freq}

for combo in "summary" "gate+_in+" "gate+_in-" "gate-_in+" "gate-_in-"; do
    python freqs.py \
    --refactor_glu \
    --combo $combo \
    --metric_type $metric_type \
    --subexperiments scatter_plots selected all_layers norms \
    --layer_list 15
done
