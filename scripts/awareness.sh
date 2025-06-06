#!/bin/bash
#
#SBATCH --job-name=awareness
#SBATCH --comment="Compute idiom awareness for formal and translated idiom occurences"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.grimm@campus.lmu.de
#SBATCH --chdir=/home/g/grimmj/IdiomHeads
#SBATCH --output=/home/g/grimmj/IdiomHeads/scripts/output/slurm.%j.%N.out
#SBATCH --ntasks=1

python3 -u compute_idiom_awareness.py -d formal trans -m "roneneldan/TinyStories-Instruct-33M" -i "tiny_formal_idiom_pos.json" -s 0 -e None
