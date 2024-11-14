import submitit
import os
import datetime
import argparse
import yaml
import re

class Trainer:
    def __init__(self, output_dir, word_size, config):
        self.cwd = "/home/ziweiji/Hallu_Det/sem_uncertainty/eigen"
        self.conda_env_name = config.get("conda_env_name", "detect")
        self.conda_path = config.get("conda_path", "/home/ziweiji/miniconda3")
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

python /home/ziweiji/Hallu_Det/sem_uncertainty/eigen/{task}.py \\
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
for D in IDK
do
for T in no_refuse_sentence
do
for SPLIT in test
do
python /home/ziweiji/Hallu_Det/sem_uncertainty/eigen/scripts/submit_job_eigen2.py \
--dataset $D \
--type $T \
--dataset_splits $SPLIT &

done
done
done
"""
    # if os.path.exists(config_path):
    #     with open(config_path, "r") as file:
    #         return yaml.safe_load(file)
    ######################################################
    dataset_splits =  args.dataset_splits.replace("_", " ")
    return {'training_args': {
        "task": "get_eigen_score",
        "dataset": args.dataset,
        'dataset_splits': dataset_splits,
        'type': args.type,
        }}

        # long --metric=llm_gpt-4 --entailment_model=gpt-3.5 
        # "no-compute_accuracy_at_all_temps": True,
    ######################################################

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Submitit Training Script")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--type", type=str, default="")
    parser.add_argument("--dataset_splits", type=str, default="")
    return parser.parse_args()


    
def get_run_output_dir(args):
    description = f"logs/{args['task']}_{args['dataset']}"
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
        slurm_qos="embodiment_shared",
        slurm_account="multimodal-reasoning",
    )

    # Submit the job
    word_size = nodes * 8
    job = executor.submit(Trainer(output_dir, word_size, config))

    print(f'Output directory: {output_dir}')
    print(f"Submitted job with ID: {job.job_id}")