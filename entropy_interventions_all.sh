#!/bin/bash
set -euox pipefail

names=("strengthening" "conditional strengthening" "proportional change" "conditional weakening" "weakening" "orthogonal output")
signs=("+" "-")
n_neuron_variants=("strengthening" "weakening" "None")

# baseline run
python -m entropy.entropy_intervention_wrap \
    --batch_size 4 \
    --device cuda:0

# Detect GPUs
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra VISIBLE_GPUS <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=${#VISIBLE_GPUS[@]}
else
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
fi
echo "Detected $NUM_GPUS GPUs. Will process all names, max $NUM_GPUS concurrent."
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No GPUs found. Exiting."
    exit 1
fi

job_counter=0
for intervention_type in {mean_ablation,zero_ablation}; do
    for n_neurons in "${n_neuron_variants[@]}"; do
        for neuron_subset_name in "${names[@]}"; do
            # Enforce Concurrency Limit
            # Wait until we have fewer than NUM_GPUS background jobs running
            while [ $(jobs -r | wc -l) -ge $NUM_GPUS ]; do
                # Wait for one to finish
                wait -n
            done
            # Assign GPU ID cyclically (0, 1, ..., NUM_GPUS-1, 0, 1...)
            # This ensures we never exceed the limit and distribute load evenly
            gpu_id=$((job_counter % NUM_GPUS))
            # Get the physical GPU ID corresponding to the logical slot
            # If VISIBLE_GPUS is "0,1,2,3", then slot 0 -> physical 0, slot 1 -> physical 1.
            # If VISIBLE_GPUS is "2,4,6", then slot 0 -> physical 2.
            physical_gpu="${VISIBLE_GPUS[$gpu_id]}"
            
            # Run the job with a restricted environment
            (
                export CUDA_VISIBLE_DEVICES="$physical_gpu"
                python -m entropy.entropy_intervention_wrap \
                    --neuron_subset_name "$neuron_subset_name" \
                    --n_neurons $n_neurons \
                    --batch_size 4 \
                    --device cuda:0 \
                    --intervention_type $intervention_type
            ) &
            ((job_counter++)) || true
        done
    done
    #conditional ablations
    for gate in "${!signs[@]}"; do
        for post in "${!signs[@]}"; do
            # Enforce Concurrency Limit
            # Wait until we have fewer than NUM_GPUS background jobs running
            while [ $(jobs -r | wc -l) -ge $NUM_GPUS ]; do
                wait -n
            done
            # Assign GPU ID cyclically (0, 1, ..., NUM_GPUS-1, 0, 1...)
            # This ensures we never exceed the limit and distribute load evenly
            gpu_id=$((job_counter % NUM_GPUS))
            # Get the physical GPU ID corresponding to the logical slot
            # If VISIBLE_GPUS is "0,1,2,3", then slot 0 -> physical 0, slot 1 -> physical 1.
            # If VISIBLE_GPUS is "2,4,6", then slot 0 -> physical 2.
            physical_gpu="${VISIBLE_GPUS[$gpu_id]}"
            
            # Run the job with a restricted environment
            (
                export CUDA_VISIBLE_DEVICES="$physical_gpu"
                python -m entropy.entropy_intervention_wrap \
                    --neuron_subset_name weakening \
                    --gate "${signs[$gate]}" \
                    --post "${signs[$post]}" \
                    --batch_size 4 \
                    --device cuda:0 \
                    --intervention_type $intervention_type
            ) &
            ((job_counter++)) || true
        done
    done
done
wait

