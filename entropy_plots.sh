set -euox pipefail

model=$1
n_neuron_variants=("strengthening" "weakening" "None")
names=("strengthening" "conditional_strengthening" "proportional_change" "conditional_weakening" "weakening" "orthogonal_output")

#GPU assignment logic
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

for intervention_type in {mean_ablation,zero_ablation}; do
    for n_neurons in "${n_neuron_variants[@]}"; do

        neurons=""
        for name in "${names[@]}"; do
            if [ "$name" = "$n_neurons" ] || [ "$n_neurons" = "None" ]; then neurons+="$name ";
            else if [ "$name" = "strengthening" ] && [ "$n_neurons" = "weakening" ]; then :;
            else neurons+="${name}_${n_neurons} "
            fi fi
        done
        #neurons="$neurons% "

        #GPU assignment logic
        while true; do
            gpu_id=$(get_free_gpu)
            if [ "$gpu_id" != "-1" ]; then
                break
            fi
            wait -n  # wait for any one job to finish, then retry
        done
        physical_gpu="${VISIBLE_GPUS[$gpu_id]}"

        (
            export CUDA_VISIBLE_DEVICES="$physical_gpu"
            python -m entropy.compare_and_plot \
                --model $model \
                --experiment_name "${model}/${n_neurons}_${intervention_type}" \
                --intervention_type $intervention_type \
                --neurons $neurons \
                --table_format md
        ) &
        GPU_JOB_PIDS[$gpu_id]=$!

    done

    #GPU assignment logic
    while true; do
        gpu_id=$(get_free_gpu)
        if [ "$gpu_id" != "-1" ]; then
            break
        fi
        wait -n  # wait for any one job to finish, then retry
    done
    physical_gpu="${VISIBLE_GPUS[$gpu_id]}"

    (
        export CUDA_VISIBLE_DEVICES="$physical_gpu"
        python -m entropy.compare_and_plot \
            --model $model \
            --experiment_name "${model}/weakening_complete_${intervention_type}" \
            --intervention_type $intervention_type \
            --neurons weakening weakening_gate+_post+ weakening_gate+_post- weakening_gate-_post+ weakening_gate-_post- \
            --table_format md
    ) &
    GPU_JOB_PIDS[$gpu_id]=$!

done
