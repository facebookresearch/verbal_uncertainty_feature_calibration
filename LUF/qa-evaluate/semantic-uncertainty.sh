#!/bin/bash
#SBATCH --job-name=sem-uncertainty
#SBATCH --gres=gpu:1       # uncomment only if/as needed
#SBATCH --time=12:00:00    
#SBATCH --cpus-per-task=2    # change as needed
#SBATCH --mem=64G
#SBATCH --account=llm_investigating
#SBATCH --qos=llm_investigating_high
#SBATCH --array=0-29
## %j is the job id, %u is the user id
#SBATCH --output=./logs/sem-uncertainty-%j.log

python semantic-uncertainty.py --job_id $SLURM_ARRAY_TASK_ID
                  