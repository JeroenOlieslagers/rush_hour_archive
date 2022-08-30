#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=40:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=tru_dist
#SBATCH --mail-type=END
#SBATCH --mail-user=zb9@nyu.edu
#SBATCH --output=tru_dist_%j.out

psiturk_file='../results/all_stages/trialdata.csv'
run_id=${SLURM_ARRAY_TASK_ID}
module purge
module load python/intel/2.7.12


#that means coping stuff
cd /scratch/$USER/rushhour/src
python analyze_real_dist.py ${psiturk_file}  
