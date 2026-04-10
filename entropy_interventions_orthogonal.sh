#!/bin/bash
set -euox pipefail

n_neuron_variants=("24" "243" "None")

for intervention_type in {zero_ablation,mean_ablation}; do
    for i in "${!n_neuron_variants[@]}"; do
        n_neurons=${n_neuron_variants[i]}
        python -m entropy.entropy_intervention_wrap \
            --neuron_subset_name "orthogonal output" \
            --n_neurons $n_neurons \
            --batch_size 4 \
            --device cuda:$i \
            --intervention_type $intervention_type &
    done
    wait
done
wait
