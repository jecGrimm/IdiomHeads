#!/bin/bash
#
#SBATCH --job-name=formal_only_idiom
#SBATCH --comment="Compute idiom only scores for formal idiom occurences"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.grimm@campus.lmu.de
#SBATCH --chdir=/home/g/grimmj/IdiomHeads
#SBATCH --output=/home/g/grimmj/IdiomHeads/scripts/output/slurm.%j.%N.out
#SBATCH --ntasks=1

python3 -u compute_idiom_only.py -d formal -s 1231 -e 1232