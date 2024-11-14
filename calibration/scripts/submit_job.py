# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import submitit
import os
import datetime
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(os.path.dirname(current_dir))
home_path = os.path.expanduser("~")

class Trainer:
    def __init__(self, output_dir, word_size, config):
        self.cwd = f"{root_path}/calibration/"
        self.conda_env_name = "vuf"
        self.conda_path = f"{home_path}/anaconda3"
        self.output_dir = output_dir
        self.training_args = config.get("training_args", {})
        self.word_size = word_size

    def create_cmd(self):
        args = []
        for key, value in self.training_args.items():
            if key == 'task':
                task = value
                continue
            if 'iti_method' in key and task not in ['causal', 'semantic_control']:
                continue
            if (key in ["max_alpha","use_predicted"]) and task != 'semantic_control':
                continue
            if ('run_' in key or key=='alpha' or key=='dataset2') and task != 'causal':
                continue
            if type(value) == bool:
                if value:
                    args.append(f"--{key}")
            elif type(value) == str:
                args.append(f"""--{key} '{value}'""")
            elif type(value) in [int, float]:
                args.append(f"--{key} {value}")
            elif value is None:
                pass
            else:
                assert False, f"Unsupported type: {key} {type(value)}"
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

python {root_path}/calibration/{task}.py \\
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
    # if os.path.exists(config_path):
    #     with open(config_path, "r") as file:
    #         return yaml.safe_load(file)
    ######################################################
    
    return {'training_args': {
        "task": args.task,
        "dataset": args.dataset,
        "dataset2": args.dataset2,
        'split': args.split,
        "run_certain": args.run_certain,
        "run_uncertain": args.run_uncertain,
        "iti_method": args.iti_method,
        "alpha": args.alpha,
        "max_alpha": args.max_alpha,
        "use_predicted": args.use_predicted,
        "prompt_type": args.prompt_type,
        "model_name": args.model_name,
        "str_process_layers": args.str_process_layers
        }}
    ######################################################

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Submitit Training Script")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--dataset2", type=str, default="trivia_qa")
    parser.add_argument("--split", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--run_certain", type=int, default=0)
    parser.add_argument("--run_uncertain", type=int, default=0)
    parser.add_argument("--iti_method", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--max_alpha", type=float, default=0.2)
    parser.add_argument("--use_predicted", type=int, default=0)
    parser.add_argument("--prompt_type", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--str_process_layers", type=str)
    return parser.parse_args()


    
def get_run_output_dir(args):
    if args['use_predicted']:
        description = f"{root_path}/logs/predicted_{args['task']}_{args['dataset']}"
    else:
        description = f"{root_path}/logs/{args['task']}_{args['dataset']}_{args['model_name']}"
    if args['task'] == 'causal':
        description += f"_run_{args['run_certain']}_{args['run_uncertain']}_{args['alpha']}"
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
        name=config['training_args']['task'],
        mem_gb=700,
        gpus_per_node=8,
        cpus_per_task=80,
        nodes=nodes,
        timeout_min=4320,
        slurm_partition="learnfair",
        # slurm_constraint='ampere80gb', #'ampere80gb',volta32gb
        slurm_exclude='learnfair6000,learnfair6001',
    )

    # Submit the job
    word_size = nodes * 8
    job = executor.submit(Trainer(output_dir, word_size, config))

    print(f'Output directory: {output_dir}')
    print(f"Submitted job with ID: {job.job_id}")