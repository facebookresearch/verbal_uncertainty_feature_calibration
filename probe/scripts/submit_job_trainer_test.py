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

python -m torch.distributed.run --nproc_per_node=8 {root_path}/probe/train_ff_multilayers_regressor_trainer.py \\
{args} 
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
    """
    
    # 0.001 0.005 0.0001 0.0005 1e-05 5e-05
for D in 'trivia_qa' 'nq_open' 'pop_qa'
do
for U in sentence_semantic_entropy
do
for L in "range(10,15)" "range(5,15)" "range(10,20)" "range(5,25)"
do
for LR in 0.01 0.05
do
for MODEL in "Qwen2.5-7B-Instruct"
do
python ~/Hallu_Det/probe/scripts/submit_job_trainer_test.py \
--dataset $D \
--label_name $U \
--model_path f"{root_path}/probe/outputs/LinearRegressor_"$U"/"$D"_"$MODEL"/"$LR"_"$L \
--model_type "LinearRegressor" \
--internal_model_name $MODEL \
--predict_split 'test' &

done
done
done
done
done
    
"Mistral-7B-Instruct-v0.3"
"Qwen2.5-7B-Instruct" 
"Meta-Llama-3.1-8B-Instruct"
    """ 
    # with open(config_path, "r") as file:
    #     return yaml.safe_load(file) 
    
    VERBAL_MODEL_PATHS = {
        "trivia_qa": f"{root_path}/probe/outputs/LinearRegressor_verbal_uncertainty/trivia_qa_sampled/0.01_range(11,13)", ###
        "nq_open": f"{root_path}/probe/outputs/LinearRegressor_verbal_uncertainty/nq_open_sampled/0.005_range(11,13)",
        "pop_qa": f"{root_path}/probe/outputs/LinearRegressor_verbal_uncertainty/pop_qa_sampled/0.005_range(11,13)",
    }

    WORD_SEMANTIC_MODEL_PATHS = {
        "trivia_qa": f"{root_path}/probe/outputs/LinearRegressor_word_semantic_entropy/trivia_qa_sampled/0.005_range(12,14)", ###
        "nq_open": f"{root_path}/probe/outputs/LinearRegressor_word_semantic_entropy/nq_open_sampled/0.01_range(12,14)",
        "pop_qa": f"{root_path}/probe/outputs/LinearRegressor_word_semantic_entropy/pop_qa_sampled/0.01_range(12,14)",
    }

    WORD_EIGEN_MODEL_PATHS = {
        "trivia_qa": f"{root_path}/probe/outputs/LinearRegressor_word_eigen/trivia_qa_sampled/0.01_range(12,14)",
        "nq_open": f"{root_path}/probe/outputs/LinearRegressor_word_eigen/nq_open_sampled/0.005_range(12,14)", #
        "pop_qa": f"{root_path}/probe/outputs/LinearRegressor_word_eigen/pop_qa_sampled/0.05_range(12,14)", #
    }

    SENT_SEMANTIC_MODEL_PATHS = {
        "trivia_qa": f"{root_path}/probe/outputs/LinearRegressor_sentence_semantic_entropy/trivia_qa_sampled/0.01_range(12,14)", ###
        "nq_open": f"{root_path}/probe/outputs/LinearRegressor_sentence_semantic_entropy/nq_open_sampled/0.01_range(12,14)",
        "pop_qa": f"{root_path}/probe/outputs/LinearRegressor_sentence_semantic_entropy/pop_qa_sampled/0.01_range(12,14)",
    }

    SENT_EIGEN_MODEL_PATHS = {
        "trivia_qa": f"{root_path}/probe/outputs/LinearRegressor_sentence_eigen/trivia_qa_sampled/0.01_range(12,14)",
        "nq_open": f"{root_path}/probe/outputs/LinearRegressor_sentence_eigen/nq_open_sampled/0.001_range(12,14)", #
        "pop_qa": f"{root_path}/probe/outputs/LinearRegressor_sentence_eigen/pop_qa_sampled/0.001_range(12,14)", #
    }

    if args.model_path:
        model_path = args.model_path
    else:
        if args.label_name == 'verbal_uncertainty':
            model_path = VERBAL_MODEL_PATHS[args.dataset]
        elif args.label_name == 'word_semantic_entropy':
            model_path = WORD_SEMANTIC_MODEL_PATHS[args.dataset]
        elif args.label_name == 'word_eigen':
            model_path = WORD_EIGEN_MODEL_PATHS[args.dataset]
        elif args.label_name == 'sentence_semantic_entropy':
            model_path = SENT_SEMANTIC_MODEL_PATHS[args.dataset]
        elif args.label_name == 'sentence_eigen':
            model_path = SENT_EIGEN_MODEL_PATHS[args.dataset]
        
    layers_to_process = model_path.split('/')[-1].split("_")[-1]
    source_dirs = [f"{root_path}/datasets/{args.dataset}/{args.internal_model_name}/"]
    data_paths = []
    for d in source_dirs:
        data_paths.append(d+"{split}.csv")
    data_paths = " ".join(data_paths)
    source_dirs = " ".join(source_dirs)
    return {
        "training_args": {
            "predict_split": args.predict_split,
            "output_dir": model_path,
            "model_type": args.model_type,
            "internal_model_name": args.internal_model_name,
            "model_path": model_path,
            "info_type": "last", # don't add only question!!
            "save_cache": args.save_cache,
            "label_name":  args.label_name, # "verbal_uncertainty", "word_sematic_entropy" refuse_sematic_entropy eigen
            "layers_to_process": layers_to_process,
            "pair_differ": False,
            "source_dirs": source_dirs,
            "data_paths": data_paths,
            "save_hidden_state": True,
            "p_num_latents_list": 1,
            "share_perceiver": True,
            "per_device_eval_batch_size": 8,
            "gradient_accumulation_steps": 1,
            "do_train": False,
            "do_eval": False,
            "do_predict": True,
            "eval_strategy": "epoch",
            "fp16": True,
            "metric_for_best_model": "mse",
            "max_seq_length": 500
        }
    }

# 314452
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Submitit Training Script")
    parser.add_argument("--dataset", type=str, default="",)
    parser.add_argument("--model_type", type=str, default="",)
    parser.add_argument("--label_name", type=str, default="",)
    parser.add_argument("--model_path", type=str, default="",)
    parser.add_argument("--predict_split", type=str, default='test',)
    parser.add_argument("--save_cache", type=str, default='only_question_last_activations',)
    parser.add_argument("--internal_model_name", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load configuration
    config = load_config(args)
    output_dir = config['training_args']['output_dir']
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