#!/bin/bash

for neuron_subset_name in {"strengthening","conditional strengthening","proportional change","conditional weakening","weakening"}; do
    for n_neurons in {24,243}; do
        python -m entropy.entropy_intervention_wrap \
            --neuron_subset_name $neuron_subset_name \
            --n_neurons $n_neurons \
            --batch_size 4
    done
done

for gate in {"+","-"}; do
    for post in {"+","-"}; do
        python -m entropy.entropy_intervention_wrap \
            --neuron_subset_name weakening \
            --gate $gate \
            --post $post \
            --batch_size 4
    done
done
