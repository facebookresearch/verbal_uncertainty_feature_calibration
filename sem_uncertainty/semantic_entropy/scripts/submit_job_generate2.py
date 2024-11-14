import submitit
import os
import datetime
import argparse
import yaml
import re

class Trainer:
    def __init__(self, output_dir, word_size, config):
        self.cwd = config.get("cwd", "/home/ziweiji/Hallu_Det/sem_uncertainty")
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

python /home/ziweiji/Hallu_Det/sem_uncertainty/semantic_entropy/{task}.py \\
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
for D in 'nq_open'
do
for SPLIT in "train" 'test' 'val'
do
python /home/ziweiji/Hallu_Det/sem_uncertainty/semantic_entropy/eval_acc.py \
--dataset $D \
--split $SPLIT \
--model_name Mistral-7B-Instruct-v0.3 \
--port 'http://cr1-h100-p548xlarge-275:8000/v1' &
done
done

Qwen2.5-7B-Instruct Mistral-7B-Instruct-v0.3

tmux 0 trivia_qa Qwen2 348
tmux 1 trivia_qa Mistral 381

tmux 2 nq_open Qwen2 552
tmux 3 nq_open Mistral 275

tmux 4 pop_qa Qwen2 188
tmux 5 pop_qa Mistral 376




for D in 'trivia_qa' 'nq_open' 'pop_qa'
do
for SPLIT in "train" 'test' 'val'
do
python /home/ziweiji/Hallu_Det/sem_uncertainty/semantic_entropy/compute_semantic_entropy.py \
--dataset $D \
--prompt_type $T \
--split $SPLIT \
--port 'http://cr1-h100-p548xlarge-87:8000/v1'

done
done
done


git add /home/ziweiji/Hallu_Det/sem_uncertainty/outputs/*/sentence/*-Instruct*/* -f
git add /home/ziweiji/Hallu_Det/sem_uncertainty/outputs/*/sentence/*-Instruct/* -f


for D in 'nq_open' 'pop_qa' 'trivia_qa'
do
for SPLIT in 'train'
do
for MODEL in "Qwen2.5-7B-Instruct" "Mistral-7B-Instruct-v0.3"
do
for T in 1.0
do
python /home/ziweiji/Hallu_Det/sem_uncertainty/semantic_entropy/scripts/submit_job_generate2.py \
--dataset $D \
--prompt_type "sentence" \
--split $SPLIT \
--temperature $T \
--model_name $MODEL &

done
done
done
done
   """
    
    # Qwen2.5-7B-Instruct 'Mistral-7B-Instruct-v0.3'
    
    return {'training_args': {
        "task": "generate_answers",
        "dataset": args.dataset,
        'split': args.split,
        "model_name": args.model_name,
        "entailment_model": args.entailment_model, #"Meta-Llama-3.1-70B-Instruct",
        "prompt_type": args.prompt_type,
        "temperature": args.temperature,
        }}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Submitit Training Script")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--prompt_type", type=str, choices=["word", "sentence", "no_refuse_sentence", "no_refuse_word"])
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--entailment_model", type=str, default="Meta-Llama-3.1-70B-Instruct")
    parser.add_argument("--model_name", type=str)
    return parser.parse_args()
    
def get_run_output_dir(args):
    description = f"logs/{args['task']}_{args['dataset']}_{args['prompt_type']}_{args['model_name']}"
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