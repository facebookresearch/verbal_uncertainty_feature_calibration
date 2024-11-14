#!/bin/bash
#SBATCH --job-name=factuality
#SBATCH --gres=gpu:4       # uncomment only if/as needed
#SBATCH --time=12:00:00    
#SBATCH --cpus-per-task=2    # change as needed
#SBATCH --mem=100G
#SBATCH --account=llm_investigating
#SBATCH --qos=llm_investigating_high
#SBATCH --array=0-29
## %j is the job id, %u is the user id
#SBATCH --output=./logs/factuality-%j.log

python factuality.py --job_id $SLURM_ARRAY_TASK_ID

                  