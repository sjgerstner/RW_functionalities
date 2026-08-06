#!/bin/bash
set -euox pipefail

names=("strengthening" "conditional strengthening" "proportional change" "conditional weakening" "weakening" "orthogonal output")
signs=("+" "-")
n_neuron_variants=("strengthening" "weakening" "None")

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

declare -a GPU_JOB_PIDS
for i in $(seq 0 $((NUM_GPUS-1))); do
    GPU_JOB_PIDS[$i]=0
done

get_free_gpu() {
    for i in $(seq 0 $((NUM_GPUS-1))); do
        local pid=${GPU_JOB_PIDS[$i]}
        # Slot is free if it's 0 or the process has died
        if [ "$pid" = "0" ] || ! kill -0 "$pid" 2>/dev/null; then
            GPU_JOB_PIDS[$i]=0
            echo "$i"
            return
        fi
    done
    # All busy — signal caller to wait
    echo "-1"
}

while true; do
    gpu_id=$(get_free_gpu)
    if [ "$gpu_id" != "-1" ]; then
        break
    fi
    wait -n  # wait for any one job to finish, then retry
done
physical_gpu="${VISIBLE_GPUS[$gpu_id]}"

# baseline run
(
    export CUDA_VISIBLE_DEVICES="$physical_gpu"
    python -m entropy.entropy_intervention_wrap \
        --batch_size 4 \
        --device cuda:0 \
        "$@"
) &
GPU_JOB_PIDS[$gpu_id]=$!

for intervention_type in {mean_ablation,zero_ablation}; do
    for n_neurons in "${n_neuron_variants[@]}"; do
        for neuron_subset_name in "${names[@]}"; do
            while true; do
                gpu_id=$(get_free_gpu)
                if [ "$gpu_id" != "-1" ]; then
                    break
                fi
                wait -n  # wait for any one job to finish, then retry
            done
            physical_gpu="${VISIBLE_GPUS[$gpu_id]}"
            
            # Run the job with a restricted environment
            (
                export CUDA_VISIBLE_DEVICES="$physical_gpu"
                python -m entropy.entropy_intervention_wrap \
                    --neuron_subset_name "$neuron_subset_name" \
                    --n_neurons $n_neurons \
                    --batch_size 4 \
                    --device cuda:0 \
                    --intervention_type $intervention_type \
                    "$@"
            ) &
            GPU_JOB_PIDS[$gpu_id]=$!
        done
    done
    #conditional ablations
    for gate in "${!signs[@]}"; do
        for post in "${!signs[@]}"; do
            while true; do
                gpu_id=$(get_free_gpu)
                if [ "$gpu_id" != "-1" ]; then
                    break
                fi
                wait -n  # wait for any one job to finish, then retry
            done
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
                    --intervention_type $intervention_type \
                    "$@"
            ) &
            GPU_JOB_PIDS[$gpu_id]=$!
        done
    done
done
wait

