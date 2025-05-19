#!/bin/bash
#
#SBATCH --job-name=logit
#SBATCH --comment="Compute logit attribution for formal and translated idiom occurences"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.grimm@campus.lmu.de
#SBATCH --chdir=/home/g/grimmj/IdiomHeads
#SBATCH --output=/home/g/grimmj/IdiomHeads/scripts/output/slurm.%j.%N.out
#SBATCH --ntasks=1

python3 -u compute_logit_attr.py -d static -s 0 -e 2761 -m "EleutherAI/pythia-1.4b" -i "pythia_static_idiom_pos.json"
