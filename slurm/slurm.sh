#!/bin/bash

#SBATCH -J process_wiki
#SBATCH -p general
#SBATCH -o process_wiki_%j.txt
#SBATCH -e process_wiki_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mapsel@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=3-00:00:00
#SBATCH --mem=180G
#SBATCH -A r00682

#module load openmpi

#Run your program
python wikitest.py 