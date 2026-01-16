#!/bin/bash
set -euox pipefail

intervention_type=${1:-zero_ablation}

names=("strengthening" "conditional strengthening" "proportional change" "conditional weakening") #don't need weakening here
signs=("+" "-")

for i in "${!names[@]}"; do
    neuron_subset_name=${names[i]}
    python -m entropy.entropy_intervention_wrap \
        --neuron_subset_name "$neuron_subset_name" \
        --batch_size 4 \
        --device cuda:$i \
        --intervention_type $intervention_type &
done
wait
