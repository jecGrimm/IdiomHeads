#!/bin/bash
#
#SBATCH --job-name=small_detect_idiom_gpu
#SBATCH --comment="Test the idiom scorer on a GPU"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.grimm@campus.lmu.de
#SBATCH --chdir=/home/g/grimmj/IdiomHeads
#SBATCH --output=/home/g/grimmj/IdiomHeads/slurm.%j.%N.out
#SBATCH --ntasks=1

python3 -u small_detect_idiom.py