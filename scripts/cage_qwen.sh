#!/bin/bash
#
#SBATCH --job-name=cage_qwen
#SBATCH --comment="Cage Qwen Activations"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.grimm@campus.lmu.de
#SBATCH --chdir=/home/g/grimmj/IdiomHeads
#SBATCH --output=/home/g/grimmj/IdiomHeads/scripts/output/slurm.%j.%N.out
#SBATCH --ntasks=1

python3 -u cage.py -d formal trans -m "Qwen/Qwen2-0.5B-Instruct"