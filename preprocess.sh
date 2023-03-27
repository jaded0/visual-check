#!/bin/bash
#SBATCH --job-name=prep_latexter
#SBATCH --output=prep_latexter.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1G   # 1G memory per CPU core
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jaden.lorenc@gmail.com

source /apps/miniconda3/latest/etc/profile.d/conda.sh
conda activate mmcoder

python pdf_conversion_multiprocessing.py
