import submitit
import os
import datetime
import argparse
import yaml
import re

class Trainer:
    def __init__(self, output_dir, word_size, config):
        self.cwd = "/private/home/ziweiji/Hallu_Det/ling_uncertainty"
        self.conda_env_name = "detect"
        self.conda_path = "detect"
        self.output_dir = output_dir
        self.training_args = config.get("training_args", {})
        self.word_size = word_size

    def create_cmd(self):
        args = []
        for key, value in self.training_args.items():
            if key == 'task':
                task = value
                continue
            if type(value) == bool:
                if value:
                    args.append(f"--{key}")
            elif type(value) == str:
                if value:
                    args.append(f"--{key} {value}")
            elif type(value) in [int, float]:
                args.append(f"--{key} {value}")
            else:
                assert False, f"Unsupported type: {type(value)}"
        args = " \\\n".join(args)

        cmd = f"""
source {self.conda_path}/etc/profile.d/conda.sh
conda activate {self.conda_env_name}
export MKL_THREADING_LAYER=GNU
hash -r

echo "Using python: $(which python)"
echo "Python version: $(python --version)"
echo "Using torchrun: $(which torchrun)"
echo "Conda envs: $(conda env list)"

python /private/home/ziweiji/Hallu_Det/ling_uncertainty/lu_llm_judge.py \\
    {args}
        """
        print(cmd)
        return cmd

    def __call__(self):
        import os
        import subprocess
        os.chdir(self.cwd)
        cmd = self.create_cmd()
        subprocess.run(cmd, shell=True, check=True, executable="/bin/zsh")


def load_config(args):
    """
conda activate detect
cd /private/home/ziweiji/Hallu_Det/ling_uncertainty

for DATA in 'trivia_qa' 'nq_open' 'pop_qa'
do
for MODEL in llama-3.1-8B-grpo
do
for SPLIT in train val test
do
python /private/home/ziweiji/Hallu_Det/ling_uncertainty/lu_llm_judge.py \
--results_dir '/private/home/ziweiji/Hallu_Det/ling_uncertainty/outputs_10' \
--dataset $DATA \
--split $SPLIT \
--model_name $MODEL \
--port 'http://learnfair6023:8000/v1' &
done
done
done

for DATA in 'trivia_qa' 'nq_open' 'pop_qa'
do
for MODEL in llama-3.1-8B-grpo
do
for SPLIT in train val test
do
python /private/home/ziweiji/Hallu_Det/ling_uncertainty/scripts/submit_job_judge.py \
--dataset $DATA \
--split $SPLIT \
--model_name $MODEL
done
done
done



for DATA in 'trivia_qa'
do
for MODEL in llama-3.1-8B-grpo
do
for SPLIT in test
do
python /private/home/ziweiji/Hallu_Det/ling_uncertainty/scripts/submit_job_judge.py \
--dataset $DATA \
--split $SPLIT \
--model_name $MODEL \
--batch_size 4

done
done
done

    """
    # if os.path.exists(config_path):
    #     with open(config_path, "r") as file:
    #         return yaml.safe_load(file)
    ######################################################
    
    return {'training_args': {
        "dataset": args.dataset,
        'split': args.split,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "results_dir": '/private/home/ziweiji/Hallu_Det/ling_uncertainty/outputs_10',
        }}
    ######################################################

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Submitit Training Script")
    parser.add_argument("--dataset", type=str, default='')
    parser.add_argument("--split", type=str, default='')
    parser.add_argument("--model_name", type=str, default='')
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--results_fn", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


    
def get_run_output_dir(args):
    description = f"logs/judge_{args['dataset']}_{args['split']}"
    return description

if __name__ == "__main__":
    args = parse_args()

    # Load configuration
    config = load_config(args)
    output_dir = get_run_output_dir(config['training_args'])
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the executor
    nodes = 1
    executor = submitit.AutoExecutor(folder=output_dir)
    executor.update_parameters(
        mem_gb=700,
        gpus_per_node=8,
        cpus_per_task=80,
        nodes=nodes,
        timeout_min=4320,
        slurm_partition="learnfair",
        slurm_exclude='learnfair6000',
    )

    # Submit the job
    word_size = nodes * 8
    job = executor.submit(Trainer(output_dir, word_size, config))

    print(f'Output directory: {output_dir}')
    print(f"Submitted job with ID: {job.job_id}")