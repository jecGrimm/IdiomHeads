#!/bin/bash
#
#SBATCH --job-name=literal_only
#SBATCH --comment="Compute literal only scores for formal and translated idiom occurences"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.grimm@campus.lmu.de
#SBATCH --chdir=/home/g/grimmj/IdiomHeads
#SBATCH --output=/home/g/grimmj/IdiomHeads/scripts/output/slurm.%j.%N.out
#SBATCH --ntasks=1

python3 -u compute_literal_only.py -d formal trans -s 225 0 -e None None -i pythia_formal_idiom_pos.json