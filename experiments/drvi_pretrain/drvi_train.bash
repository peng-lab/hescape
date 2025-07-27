#!/bin/bash

# Define the directory and files
base_dir="/p/project1/hai_spatial_clip/data/xenium/preprocessing_adatas"
h5_files=(
    "human_5k_panel_adata"
    "human_breast_panel_adata.h5ad"
    "human_colon_panel_adata.h5ad"
    "human_immuno_oncology_panel_adata.h5ad"
    "human_lung_healthy_panel_adata.h5ad"
    "human_multi_tissue_panel_adata.h5ad"
    "human_skin_panel_adata.h5ad"
    # Add your other H5 files here

)

# Submit a job for each file
for h5_file in "${h5_files[@]}"; do
    base_name=$(basename "$h5_file" .h5ad)
    sbatch << EOF
#!/bin/bash
#SBATCH --account=hai_spatial_clip
#SBATCH -o $pp_drvi_${base_name}_%j.txt
#SBATCH -e $pp_drvi_${base_name}_%j.txt
#SBATCH --job-name=pp_drvi_${base_name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --partition=booster
#SBATCH --time=5:00:00

source $HOME/.bashrc

echo "Starting drvi training..."

conda deactivate
conda activate hescape

python experiments/drvi_pretrain/train.py --h5-file $h5_file
EOF
done
