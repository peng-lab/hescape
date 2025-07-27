conda activate spatialclip

export CUDA_VISIBLE_DEVICES=0
unset SLURM_CPU_BIND
# Lung
python experiments/drvi_pretrain/train.py --organ "Lung"

# Bowel
python experiments/drvi_pretrain/train.py --organ "Bowel"

# Brain
python experiments/drvi_pretrain/train.py --organ "Brain"

# Breast
python experiments/drvi_pretrain/train.py --organ "Breast"

# Pancreas
python experiments/drvi_pretrain/train.py --organ "Pancreas"
