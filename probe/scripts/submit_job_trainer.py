import submitit
import os
import argparse
import re
home_path = os.path.expanduser("~")
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(os.path.dirname(current_dir))

class Trainer:
    def __init__(self, output_dir, word_size, config):
        self.cwd = current_dir
        self.conda_env_name = "vuf"
        self.conda_path =  f"{home_path}/anaconda3"
        self.output_dir = output_dir
        self.training_args = config.get("training_args", {})
        self.word_size = word_size

    def create_cmd(self):
        args = []
        for key, value in self.training_args.items():
            if key == 'fsdp':
                args.append(f"--{key} \"{value}\"")
                continue
            if type(value) == bool:
                args.append(f"--{key} {str(value)}")
            elif type(value) == str:
                if key in ['learning_rate']:
                    args.append(f"--{key} {value}")
                else:
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

echo "Using NVCC:"
which nvcc
echo "Using python:"
which python
echo "Python version:"
python --version
echo "Using torchrun:"
which torchrun

export WANDB_PROJECT=probe
OMP_NUM_THREADS=16 \\
python -m torch.distributed.run --nproc_per_node=8 {root_path}/probe/train_ff_multilayers_regressor_trainer.py \\
--output_dir "{self.output_dir}" {args} 
"""
        # 
        print(cmd)
        return cmd

    def __call__(self):
        import os
        import subprocess
        os.chdir(self.cwd)
        cmd = self.create_cmd()
        subprocess.run(cmd, shell=True, check=True, executable="/bin/zsh")


def load_config(args):
    source_dirs = [f"{root_path}/datasets/{args.dataset}/{args.internal_model_name}/"]
    data_paths = []
    for d in source_dirs:
        data_paths.append(d+"{split}.csv")
    data_paths = " ".join(data_paths)
    source_dirs = " ".join(source_dirs)

    if len(args.layers_to_process) < 3:
        if args.model_type == 'Linear':
            per_device_train_batch_size = per_device_eval_batch_size = 160
        else:
            per_device_train_batch_size = per_device_eval_batch_size = 160
    else:
        if args.model_type == 'Linear':
           per_device_train_batch_size = per_device_eval_batch_size = 80
        else:
            per_device_train_batch_size = per_device_eval_batch_size = 80

    return {'training_args': {
        "model_type": args.model_type,
        "info_type": "last", # don't add only question!!
        "save_cache": args.save_cache,
        "label_name":  args.label_name, # "verbal_uncertainty", "sematic_entropy" 
        "layers_to_process": args.layers_to_process,
        "internal_model_name": args.internal_model_name,
        "source_dirs": source_dirs,
        "data_paths": data_paths,
        "use_val_to_train": bool(args.use_val_to_train),
        "pair_differ": False,
        "hidden_batch_size": 64,
        "save_hidden_state": True,
        "p_num_latents_list": 1,
        "share_perceiver": True,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": 8,
        "do_train": True,
        "do_eval": True,
        "learning_rate": args.learning_rate,
        "warmup_ratio": 0.1,
        "max_grad_norm": 0.3,
        "weight_decay": 0.001,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "fp16": True,
        "num_train_epochs": 30,
        "metric_for_best_model": "mse",
        "greater_is_better": False,
        "save_total_limit": 1,
        "logging_strategy": "steps",
        "logging_steps": 100,
        "report_to": "wandb",
        "max_seq_length": 512,
        'lora': False,
        "overwrite_output_dir": True,
        }}

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Submitit Training Script")
    parser.add_argument("--layers_to_process", type=str, default="",)
    parser.add_argument("--dataset", type=str, default="",)
    parser.add_argument("--model_type", type=str, default="",)
    parser.add_argument("--label_name", type=str, default="",)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--use_val_to_train", type=int, default=0)
    parser.add_argument("--save_cache", type=str, default="only_question_last_activations")
    parser.add_argument("--internal_model_name", type=str, default="")
    return parser.parse_args()


def get_run_output_dir(args):
    print("get_run_output_dir args", args)
    model_type = args['model_type']
    label_name = args['label_name']
    lr = args['learning_rate']
    layers_to_process = args['layers_to_process']
    # "datasets/"$DATA"/"$DATASPLIT
    source_dirs = args['source_dirs'].split(" ")
    datatset = []
    for source_dir in source_dirs:
        datatset.append(source_dir.split("/")[-3])
    datatset = '_'.join(datatset)
    datasplit = source_dirs[0].split("/")[-2]
    description = f"{root_path}/probe/outputs/{model_type}_{label_name}/{datatset}_{datasplit}/{lr}_{layers_to_process}"
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
        # slurm_constraint=args.slurm_constraint, 
        # slurm_mail_user="zjiad@connect.ust.hk",
        # slurm_mail_type="END,FAIL,BEGIN",
        slurm_exclude='learnfair6000,learnfair6001',
    )
    # Submit the job
    word_size = nodes * 8
    job = executor.submit(Trainer(output_dir, word_size, config))

    print(f'Output directory: {output_dir}')
    print(f"Submitted job with ID: {job.job_id}")