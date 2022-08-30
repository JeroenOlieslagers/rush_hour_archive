#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=40:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=ll
#SBATCH --mail-type=END
#SBATCH --mail-user=zb9@nyu.edu
#SBATCH --output=ll_%j.out

move_file='../results/pilot/moves.c.csv'
run_id=${SLURM_ARRAY_TASK_ID}
module purge
module load python/intel/2.7.12


cd /scratch/$USER/rushhour/src
python run_loglik.py ${move_file} ${run_id} >> ../output/report_LRTA_instance_data${run_id}.txt
