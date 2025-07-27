#!/bin/bash

#SBATCH -o slurm_uce_eval_%j.txt
#SBATCH -e slurm_uce_eval_error_%j.txt
#SBATCH --job-name=uce-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=14
#SBATCH --mem=240G
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_priority
#SBATCH --gres=gpu:2
#SBATCH -C v100_32gb
#SBATCH --time=12:00:00
#SBATCH --nice=10000

#SBATCH --mail-user=rushin.gindra@helmholtz-munich.de
#SBATCH --mail-type=BEGIN,END,FAIL  # Send email when job starts, ends, or fails

source $HOME/.bashrc

echo "Starting UCE Eval..."

chmod 600 ./slurm_uce_eval_$SLURM_JOB_ID.txt
chmod 600 ./slurm_uce_eval_error_$SLURM_JOB_ID.txt

conda activate uce

accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 inference.py \
--adata_dir "/mnt/raw/datasets/hest1k/st_processed_esm" \
--temp_dir "/mnt/raw/datasets/hest1k/st_temp" \
--final_dir "/mnt/raw/datasets/hest1k/st_uce" \
--nlayers 4 \
--model_loc model_files/4layer_model.torch \
--batch_size 128 \
--filter False \
--skip False \
--multi_gpu True
