#!/bin/bash
#SBATCH --job-name=lu-emb-sim
#SBATCH --gres=gpu:1       # uncomment only if/as needed
#SBATCH --time=3:00:00    
#SBATCH --cpus-per-task=2    # change as needed
#SBATCH --mem=50G
#SBATCH --account=llm_investigating
#SBATCH --qos=llm_investigating_high
#SBATCH --array=0-49
## %j is the job id, %u is the user id
#SBATCH --output=./logs/lu-emb-sim-%j.log

python lu-emb-sim.py --job_id $SLURM_ARRAY_TASK_ID

                  