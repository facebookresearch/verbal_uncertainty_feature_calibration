#!/bin/bash
#SBATCH --job-name=qa-with-iti
#SBATCH --gres=gpu:2       # uncomment only if/as needed
#SBATCH --time=3:00:00    
#SBATCH --cpus-per-task=2    # change as needed
#SBATCH --mem=50G
#SBATCH --account=llm_investigating
#SBATCH --qos=llm_investigating_high
#SBATCH --array=0-29
## %j is the job id, %u is the user id
#SBATCH --output=./logs/qa-with-iti-%j.log

python qa-with-iti.py --job_id $SLURM_ARRAY_TASK_ID

                  