#!/bin/bash
#
#SBATCH --job-name=formal_detect_idiom
#SBATCH --comment="Detect idiom heads for formal idiom occurences"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.grimm@campus.lmu.de
#SBATCH --chdir=/home/g/grimmj/IdiomHeads
#SBATCH --output=/home/g/grimmj/IdiomHeads/slurm.%j.%N.out
#SBATCH --ntasks=1

python3 -u detect_idiom.py