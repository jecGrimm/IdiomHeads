#!/bin/bash
#
#SBATCH --job-name=ablation
#SBATCH --comment="Compute ablation results for formal and translated idiom occurences"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.grimm@campus.lmu.de
#SBATCH --chdir=/home/g/grimmj/IdiomHeads
#SBATCH --output=/home/g/grimmj/IdiomHeads/gpu_scripts/output/slurm.%j.%N.out
#SBATCH --ntasks=1

python3 -u compute_ablation.py -d formal trans -s 0 0 -e None None -m "EleutherAI/Pythia-1.4B" -i "pythia_formal_idiom_pos_ablation.json" -a "pythia-1.4b_formal_DLA"
