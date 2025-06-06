#!/bin/bash
#
#SBATCH --job-name=tiny_idiom_only
#SBATCH --comment="Compute idiom only scores for formal and translated idiom occurences"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.grimm@campus.lmu.de
#SBATCH --chdir=/home/g/grimmj/IdiomHeads
#SBATCH --output=/home/g/grimmj/IdiomHeads/scripts/output/slurm.%j.%N.out
#SBATCH --ntasks=1

python3 -u compute_idiom_only.py -d trans -m "roneneldan/TinyStories-Instruct-33M" -i "tiny_formal_idiom_pos.json"
