import submitit
import os
import datetime
import argparse
import yaml
import re


class Trainer:
    def __init__(self, output_dir, word_size, config):
        self.cwd = config.get("cwd", "/private/home/ziweiji/Hallu_Det/internal_information")
        self.conda_env_name = config.get("conda_env_name", "detect")
        self.conda_path = config.get("conda_path", "/private/home/ziweiji/anaconda3")
        self.output_dir = output_dir
        self.training_args = config.get("training_args", {})
        self.word_size = word_size

    def create_cmd(self):
        args = []
        for key, value in self.training_args.items():
            if type(value) == bool:
                if value:
                    args.append(f"--{key}")
            elif type(value) == str:
                value = re.sub(r"\n", r"\\n", value)
                values = value.split(" ")
                values = [f"\"{v}\"" for v in values]
                values = " ".join(values)
                args.append(f"--{key} {values}")
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

python get_hidden_state.py --save_dir_root "{self.output_dir}" {args}
        """
# -m torch.distributed.run --nproc_per_node=8 
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
  
## QUESTION

for D in trivia_qa nq_open pop_qa
do
for S in test train val
do
for MODEL in "Qwen2.5-7B-Instruct"
do
python /private/home/ziweiji/Hallu_Det/internal_information/scripts/submit_job_get_hidden_state.py \
--source_dir $D \
--splits $S \
--datasplit $MODEL \
--internal_model_name $MODEL \
--info_type only_question_last &

done
done
done


for D in trivia_qa nq_open pop_qa
do
for S in test train val
do
for MODEL in "Qwen2.5-7B-Instruct" "Mistral-7B-Instruct-v0.3"
do
python /private/home/ziweiji/Hallu_Det/internal_information/scripts/submit_job_get_hidden_state.py \
--source_dir $D \
--splits $S \
--datasplit ${MODEL}_sentence \
--internal_model_name $MODEL \
--info_type last &

done
done
done
"""
    source_dirs =  [args.source_dir] 
                    #pop_qa
                    #  "trivia_qa",
                    # "nq_open",
    source_dirs = [f"/private//home/ziweiji/Hallu_Det/datasets/{d}/{args.datasplit}/" for d in source_dirs]
    data_paths = [d+"{split}.csv" for d in source_dirs]
    data_paths = " ".join(data_paths)
    source_dirs = " ".join(source_dirs)
    if args.internal_model_name == "Qwen2.5-7B-Instruct":
        layers_to_process = "range(0,29)"
    else:
        layers_to_process = "range(0,33)"

    info_type = args.info_type
    splits = args.splits.replace("_", " ")
    return {
        'training_args': {
        "model_type": "P_LSTM2",
        "layers_to_process": layers_to_process,
        "source_dirs": source_dirs,
        "data_paths": data_paths,
        "internal_model_name": args.internal_model_name,
        "splits": splits,
        "hidden_batch_size": 16,
        "info_type": info_type,
        "save_hidden_state": True,
        "save_cache": f"{info_type}_activations",
        }
    }
#   paired_data_rate_0.5_1000
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Submitit Training Script")
    parser.add_argument("--source_dir", type=str, default="")
    parser.add_argument("--splits", type=str, default="")
    parser.add_argument("--datasplit", type=str, default="sampled")
    parser.add_argument("--info_type", type=str, default="last", choices=["last", "only_question_last"])
    parser.add_argument("--internal_model_name", type=str, default="")
    return parser.parse_args()


    
def get_run_output_dir(args):
    print("get_run_output_dir args", args)
    info_type = args['info_type']
    # "/private/home/ziweiji/Hallu_Det/datasets/"$DATA"/"$DATASPLIT
    datatset, datasplit = args['source_dirs'].split("/")[-2:]
    internal_model_name = args['internal_model_name']
    description = f"get_hidden_state/{datatset}_{datasplit}_{info_type}_{internal_model_name}/"
    # description += datetime.datetime.now().strftime("%m%d-%H%M")
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
        # slurm_constraint='ampere80gb', #'ampere80gb',volta32gb
        slurm_exclude='learnfair6000,learnfair6001',
    )

    # Submit the job
    word_size = nodes * 4
    job = executor.submit(Trainer(output_dir, word_size, config))

    print(f'Output directory: {output_dir}')
    print(f"Submitted job with ID: {job.job_id}")