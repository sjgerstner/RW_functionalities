#!/bin/bash -l

#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --job-name=testjob_gpu
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
export HF_HUB_OFFLINE=1

srun --kill-on-bad-exit=1 \
    apptainer exec --nv $WORK/weakening4.sif \
    bash -c \
    "pip install ~/TransformerLens && python main.py --refactor_glu --model Qwen/Qwen2.5-0.5B"
