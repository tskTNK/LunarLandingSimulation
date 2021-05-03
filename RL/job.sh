#!/bin/bash 
#SBATCH -J tensorflow_job 
#SBATCH -o tensorflow_job.o%j 
#SBATCH -t 16:00:00 
#SBATCH -N 1 -n 24 # 24 is the maximum that starts immediately
#SBATCH --gres=gpu:0 #  4  is maximum for the same reasons
#SBATCH --mem=32GB
#SBATCH --mail-type=END # When to get mail
#SBATCH --mail-user=bump.bunny@gmail.com # Address to get email (email lines are optional)
 
module add python
python main.py 