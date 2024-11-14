import os
import argparse
import re
import torch

cwd = os.getcwd()
cwd = "/".join(cwd.split("/")[:-1])
import sys
sys.path.append(cwd)
from src.utils import process_layers_to_process
from get_internal_info import prepare_internal_info
import random
random.seed(42)
os.environ["TOKENIZERS_PARALLELISM"] = "true"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dim", type=int, default=4096)
    parser.add_argument("--hidden_dim", type=int, default=11008)
    parser.add_argument("--model_type", type=str, default="LlamaMLP")
    parser.add_argument("--p_num_latents_list", nargs='*', type=int) 
    parser.add_argument("--splits", nargs='*', type=str)
    parser.add_argument("--share_perceiver", action="store_true")
    parser.add_argument("--info_type", type=str, choices=["last", "mean", 'each', 'only_question_each', 'only_question_last'], default="each")
    parser.add_argument("--cache_info", action="store_true")
    parser.add_argument("--layers_to_process", nargs='*', type=str)
    parser.add_argument("--hidden_state_dims", type=str)
    parser.add_argument("--select_hidden_state_dims_method", type=str)
    parser.add_argument("--uncertainty", type=str, default="")
    parser.add_argument("--annealing_step", type=int, default=10) #?????
    parser.add_argument("--lrs", nargs='*', type=float)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--training_epoch", type=int, default=10)
    parser.add_argument("--eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--logging_strategy", type=str, default="steps")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--report_to",  type=str, default="wandb")
    parser.add_argument("--load_best_model_at_end", type=bool, default=True)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--metric_for_best_model", type=str, default="accuracy")
    parser.add_argument("--greater_is_better", type=bool, default=True)
    parser.add_argument("--metric_name", type=str, default="accuracy")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--hidden_batch_size", type=int, default=64)
    parser.add_argument("--source_dirs", nargs='*', type=str) 
    parser.add_argument("--data_paths", nargs='*', type=str)
    parser.add_argument("--label", type=str, default='label')
    parser.add_argument("--internal_model_name", type=str, default="")
    parser.add_argument("--save_cache", type=str, default="")
    parser.add_argument("--save_dir_root", type=str)
    parser.add_argument("--clean_checkpoints", action="store_true")
    parser.add_argument("--only_predict", action="store_true")
    parser.add_argument("--ignore_missing_info", action="store_true")
    parser.add_argument("--save_hidden_state", action="store_true")
    parser.add_argument("--save_max_activation_ratio", action="store_true")
    parser.add_argument("--save_sparsity", action="store_true")
    parser.add_argument("--save_activation_correlation", action="store_true")
    parser.add_argument("--save_logits", action="store_true")
    parser.add_argument("--save_attention", action="store_true")
    parser.add_argument("--save_attention_lookback", action="store_true")
    parser.add_argument("--remove_question", action="store_true")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--select_vocab", type=str)
    parser.add_argument("--input_x", default="")
    parser.add_argument("--predict_result_file", type=str, default="test_results.json")
    parser.add_argument("--max_seq_length", type=int, default=512)
    args = parser.parse_args()

    batch_size = args.batch_size
    # args.per_device_train_batch_size = args.per_device_batch_size
    # args.per_device_eval_batch_size = args.per_device_batch_size
    # args.gradient_accumulation_steps = batch_size//args.per_device_batch_size

    model_type = args.model_type
    if_resnet = "ResNet" in model_type
    info_type = args.info_type
    o_layers_to_process = args.layers_to_process
    args.layers_to_process = process_layers_to_process(args.layers_to_process)
    args.save_dir_root = re.sub("\/+", "/", args.save_dir_root)
    if args.save_dir_root[-1] == "/":
        args.save_dir_root = args.save_dir_root[:-1]
    save_dir_root = args.save_dir_root
    if args.input_x:
        INPUT = args.input_x
    elif args.save_hidden_state:
        INPUT = "hidden_states"
    elif args.save_max_activation_ratio and args.save_sparsity and args.save_activation_correlation:
        INPUT = "max_activation_ratio_sparsity_correlation"
    elif args.save_max_activation_ratio:
        # assert "LSTM" in args.classifer_type
        INPUT = "max_activation_ratio"
    elif args.save_sparsity:
        # assert "LSTM" in args.classifer_type
        INPUT = "sparsity"
    elif args.save_activation_correlation:
        INPUT = "activation_correlation"
    elif args.save_logits:
        INPUT = "logits"
    elif args.save_attention:
        INPUT = "attentions"
    elif args.save_attention_lookback:
        INPUT = "lookback_ratios"
    else:
        assert False
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    internal_model_name = args.internal_model_name
    save_attention = False
    print('device:', args.device)
    if not args.only_predict:
        all_internal_info = prepare_internal_info(args.source_dirs, args.layers_to_process, args.internal_model_name, args.hidden_batch_size, args.device, 
                            info_type, args.save_hidden_state, args.save_max_activation_ratio, args.save_sparsity, args.save_activation_correlation, args.save_logits, args.save_attention, args.save_attention_lookback,
                            max_length=args.max_seq_length, select_vocab=args.select_vocab, remove_question=args.remove_question, save_cache=args.save_cache,
                            splits=args.splits)
                            