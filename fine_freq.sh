#!/bin/bash
set -euox pipefail

for metric in "gate+_in+" "gate+_in-" "gate-_in+" "gate-_in-"; do
    python freqs.py \
    --refactor_glu \
    --metric $metric \
    --subexperiments scatter_plots selected \
    --layer_list 15
done