#!/bin/bash

#SBATCH --time=1:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=2
#SBATCH --mem-per-cpu=256000M   # 64G memory per CPU core
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --qos=cs
#SBATCH --partition=cs
#SBATCH --job-name=latexter
#SBATCH --output ./truereader.out
#SBATCH --mail-user jaden.lorenc@gmail.com

# some helpful debugging options
set -e
set -u

nvidia-smi

export PATH=$HOME/texlive/bin/x86_64-linux:$PATH
export MANPATH=$HOME/texlive/texmf-dist/doc/man:$MANPATH
export INFOPATH=$HOME/texlive/texmf-dist/doc/info:$INFOPATH

which pdflatex
which pdftoppm
echo 'has whiched'

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source /apps/miniconda3/latest/etc/profile.d/conda.sh
conda activate mmcoder
export WANDB_EXECUTABLE=$CONDA_PREFIX/bin/python
export WANDB_MODE=offline
# Update LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib


torchrun --nproc_per_node=2 --master_port=12345 truereader.py \
--model_name_or_path ~/visual-check/llama_weights/ \
--data_path ./alpaca_data.json \
--output_dir ~/visual-check/output/ \
--bf16 True \
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 2000 \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--fsdp "full_shard auto_wrap" \
--fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
--tf32 True