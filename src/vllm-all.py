# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import submitit
import os
import datetime
import argparse
import os
home_path = os.path.expanduser("~")
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_dir)

class Trainer:
    def __init__(self, output_dir, SCRIPT):
        self.cwd = root_path
        self.conda_env_name = "vuf"
        self.conda_path = f"{home_path}/anaconda3"
        self.output_dir = output_dir
        self.args = {}
        self.script = SCRIPT

    def create_cmd(self):
        cmd = f"""
source {self.conda_path}/etc/profile.d/conda.sh
conda activate {self.conda_env_name}
hash -r

echo "Using python: $(which python)"
echo "Python version: $(python --version)"
echo "Conda envs: $(conda env list)"

{self.script}
"""
        print(cmd)
        return cmd

    def __call__(self):
        import os
        import subprocess
        os.chdir(self.cwd)
        cmd = self.create_cmd()
        subprocess.run(cmd, shell=True, check=True, executable="/bin/zsh")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Submitting Inference Job")
    parser.add_argument(
        "--timeout", type=int, default=4320, help="Timeout in minutes (default: 72 hours)"
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.1-70B-Instruct", help="model"
    )
    return parser.parse_args()

    
def get_run_output_dir(MODEL):    
    description = f'{root_path}/slurm_servers/{MODEL}/'
    description += datetime.datetime.now().strftime("%m%d-%H%M")
    return description


agent_paths = {
    "llama3.1_70B": "meta-llama/Llama-3.1-70B-Instruct",
}
paths_to_models = { v: k for k, v in agent_paths.items() }


if __name__ == "__main__":
    '''''
    vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size=8 --max-model-len=4096 --disable-log-stats --download_dir=~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct
    '''''

    args = parse_args()
    MODEL = args.model
    print(args.model)

    output_dir = get_run_output_dir(MODEL)
    os.makedirs(output_dir, exist_ok=True)
    model_key = paths_to_models[MODEL]

    # Initialize the executor
    executor = submitit.AutoExecutor(folder=output_dir)
    executor.update_parameters(
        name=model_key,
        mem_gb=512,
        gpus_per_node=8,
        cpus_per_task=80,
        nodes=1,
        timeout_min=args.timeout,
        slurm_partition="learnfair",
        slurm_constraint='ampere80gb',
        slurm_exclude='learnfair6000',
    )

    MAX_MODEL_LEN = "--max-model-len=4096"
    PARALLEL_ARGS = "--tensor-parallel-size=8"
    SCRIPT = f"vllm serve {MODEL} {PARALLEL_ARGS} {MAX_MODEL_LEN} --disable-log-stats --download_dir={home_path}/.cache/huggingface/hub/models--{MODEL.replace('/', '--')}"

    # Submit the job
    job = executor.submit(Trainer(output_dir, SCRIPT))

    print(f"Submitted job with ID: {job.job_id}, NAME: {model_key}")
    print(f'Output directory: {output_dir}')
