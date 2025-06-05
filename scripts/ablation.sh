#!/bin/bash
#
#SBATCH --job-name=ablation
#SBATCH --comment="Compute ablation results for formal and translated idiom occurences"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.grimm@campus.lmu.de
#SBATCH --chdir=/home/g/grimmj/IdiomHeads
#SBATCH --output=/home/g/grimmj/IdiomHeads/scripts/output/slurm.%j.%N.out
#SBATCH --ntasks=1

python3 -u compute_ablation.py -d formal trans -s 0 0 -e None None -m "meta-llama/Llama-3.2-1B-Instruct" -i "llama_formal_idiom_pos.json" -a "Llama-3.2-1B-Instruct"
