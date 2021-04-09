#!/usr/bin/env bash

#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=16gb
#SBATCH --job-name=test_slurm
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mcbride.342@osu.edu

#run like sbatch example_analysis.sh
#need to load python3.7-conda4.5
module load python/3.7-conda4.5
#cd into the correct dir
cd /fs/project/beatty.85/mcbride.342/analysis_GCR_models/SC_python_code
#just run the code and wait
python3.7 example_analysis_run.py
