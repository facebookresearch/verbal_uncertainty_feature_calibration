#!/bin/bash
#SBATCH --job-name=lu-llm-judge
#SBATCH --gres=gpu:8       # uncomment only if/as needed
#SBATCH --time=12:00:00    
#SBATCH --cpus-per-task=2    # change as needed
#SBATCH --mem=100G
#SBATCH --account=llm_investigating
#SBATCH --qos=llm_investigating_high
#SBATCH --array=15,17,27
## %j is the job id, %u is the user id
#SBATCH --output=./logs/lu-llm-judge-%j.log

python lu-llm-judge.py --job_id $SLURM_ARRAY_TASK_ID

                  