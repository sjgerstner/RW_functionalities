#!/bin/bash

names=("strengthening" "conditional strengthening" "proportional change" "conditional weakening" "weakening")
signs=("+" "-")

python -m entropy.entropy_intervention_wrap \
    --batch_size 4 \
    --device cuda:0

for n_neurons in {24,243}; do
    for i in "${!names[@]}"; do
        neuron_subset_name=${names[i]}
        if [ $n_neurons -eq 243 ] && [ $neuron_subset_name -eq "weakening" ]; then
            n_neurons=None
        fi
        python -m entropy.entropy_intervention_wrap \
            --neuron_subset_name "$neuron_subset_name" \
            --n_neurons $n_neurons \
            --batch_size 4 \
            --device cuda:$i &
    done
    wait
done

for gate in "${!signs[@]}"; do
    for post in "${!signs[@]}"; do
        python -m entropy.entropy_intervention_wrap \
            --neuron_subset_name weakening \
            --gate $signs[$gate] \
            --post $signs[$post] \
            --batch_size 4 \
            --device cuda:$((2*$gate)+$post) &
    done
done
