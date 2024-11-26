#!/bin/bash

#SBATCH -J test10
#SBATCH -p general
#SBATCH -o test10_wiki_%j.txt
#SBATCH -e test10_wiki_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mapsel@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --mem=180G
#SBATCH -A r00682

#module load openmpi

#Run your program
python first_second10.py 