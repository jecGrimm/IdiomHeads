#!/bin/bash
#
#SBATCH --job-name=awareness
#SBATCH --comment="Compute idiom awareness for formal and translated idiom occurences"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.grimm@campus.lmu.de
#SBATCH --chdir=/home/g/grimmj/IdiomHeads
#SBATCH --output=/home/g/grimmj/IdiomHeads/scripts/output/slurm.%j.%N.out
#SBATCH --ntasks=1

python3 -u compute_idiom_awareness.py -d formal trans -m "EleutherAI/pythia-1.4b" -i "pythia_formal_idiom_pos_all.json" -s 0 0 -e None None
