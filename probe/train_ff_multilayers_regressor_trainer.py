import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['OMP_NUM_THREADS'] = "64"
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

import torch
available_gpus = torch.cuda.device_count()
print("available_gpus", available_gpus)

from dataclasses import dataclass, field
from typing import List, Optional
from sklearn.metrics import accuracy_score, f1_score

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, has_length
from transformers.utils import check_min_version, send_example_telemetry

import evaluate
import datasets
import numpy as np
import sys
cwd = "/private/home/ziweiji/Hallu_Det/"
if not os.path.exists(cwd):
    cwd = "/home/ziweiji/Hallu_Det/"
sys.path.append(cwd)

from src.utils import process_layers_to_process
from src.dataset import HiddenLayersDataset

import pandas as pd
import json

from  src.train_utils import load_regressor_model

import random
random.seed(42)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import logging
logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    input_dim: int = field(
        default=4096, metadata={"help": "The input dimension."}
    )
    hidden_dim: int = field(
        default=11008, metadata={"help": "The hidden dimension."}
    )
    model_type: str = field(
        default="LlamaMLP", metadata={"help": "The type of classifier to use."}
    )
    model_path: str = field(
        default="", metadata={"help": "The model path." }
    )
    p_num_latents_list: Optional[List[int]] = field(
        default=None, metadata={"help": "The number of latents to use for the perceiver."}
    )
    alpha: float = field(default=0.1, metadata={"help": "for multi task loss"})
    penalize_only_hallucinated: int = field(default=True, metadata={"help": "Whether to penalize only hallucinated."})
    regularization: bool = field(default=False, metadata={"help": "Whether to use regularization."})
    use_val_to_train: bool = field(default=False, metadata={"help": "Whether to use val to train."})
    label_name: str = field(default="", metadata={"help": "The type of uncertainty to use."})
    share_perceiver: bool = field(
        default=True, metadata={"help": "Whether to share the perceiver across layers."}
    )
    info_type: str = field(
        default="last", metadata={"help": "The type of internal information to use."}
    )
    cache_info: bool = field(
        default=False, metadata={"help": "Whether to cache the internal information."}
    )
    layers_to_process: List[str] = field(
        default=None, metadata={"help": "The layers to process."}
    )
    pair_differ: bool = field(
        default=False, metadata={"help": "Whether to pair differ the hidden states."}
    )
    predict_split: str = field(default='test', metadata={"help": "predict test or val."})
    hidden_state_dims: Optional[str] = field(
        default=None, metadata={"help": "The dimensions of the hidden states."}
    )
    select_hidden_state_dims_method: Optional[str] = field(
        default=None, metadata={"help": "The method to select the hidden state dimensions."}
    )
    uncertainty: Optional[str] = field(
        default="", metadata={"help": "The type of uncertainty to use."}
    )
    annealing_step: int = field(
        default=10, metadata={"help": "The step to anneal the uncertainty."}
    )
    hidden_batch_size: int = field(
        default=64, metadata={"help": "The batch size to use for hidden states."}
    )
    source_dirs: List[str] = field(
        default=None, metadata={"help": "The directories to load the internal information from."}
    )
    data_paths: List[str] = field(
        default=None, metadata={"help": "The directories to load the data from."}
    )
    label: str = field(
        default='label', metadata={"help": "The label column."}
    )
    internal_model_name: str = field(
        default='Meta-Llama-3.1-8B-Instruct', metadata={"help": "The internal model name."}
    )
    save_cache: str = field(
        default="", metadata={"help": "The directory to save the cache to."}
    )
    only_predict: bool = field(
        default=False, metadata={"help": "Whether to only predict."}
    )
    ignore_missing_info: bool = field(
        default=False, metadata={"help": "Whether to ignore missing information."}
    )
    save_hidden_state: bool = field(
        default=False, metadata={"help": "Whether to save the hidden states."}
    )
    save_max_activation_ratio: bool = field(
        default=False, metadata={"help": "Whether to save the max activation ratio."}
    )
    save_sparsity: bool = field(
        default=False, metadata={"help": "Whether to save the sparsity."}
    )
    save_activation_correlation: bool = field(
        default=False, metadata={"help": "Whether to save the activation correlation."}
    )
    save_logits: bool = field(
        default=False, metadata={"help": "Whether to save the logits."}
    )
    save_attention: bool = field(
        default=False, metadata={"help": "Whether to save the attention."}
    )
    save_attention_lookback: bool = field(
        default=False, metadata={"help": "Whether to save the attention lookback."}
    )
    remove_question: bool = field(
        default=False, metadata={"help": "Whether to remove the question."}
    )
    select_vocab: Optional[str] = field(
        default=None, metadata={"help": "The vocabulary to select."}
    )
    input_x: Optional[str] = field(
        default="", metadata={"help": "The input x."}
    )
    predict_result_file: str = field(
        default="test_results.json", metadata={"help": "The prediction result file."}
    )
    max_seq_length: int = field(
        default=512, metadata={"help": "The maximum sequence length."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    lora: bool = field(default=False, metadata={"help": "Whether to use LoRA."})

def data_collator_with_padding(features):
    first = features[0]
    batch = {}
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)
    
    for k, v in first.items():
        if k not in ("label", "label_ids", 'labels_reg') and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                # x f[k] = [num_layers, seq_len, hidden_dim]
                max_length = max([f[k].size(1) for f in features])
                padded_v = []
                for f in features:
                    pad_length = max_length - f[k].size(1)
                    padded_v.append(torch.nn.functional.pad(f[k], (0, 0, 0, pad_length, 0, 0)))
                batch[k] = torch.stack(padded_v, dim=0)
            elif isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch
    

    
if __name__ == '__main__':
    parser = HfArgumentParser((DataTrainingArguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    send_example_telemetry("run_classification", args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    model_type = args.model_type
    # if_resnet = "ResNet" in model_type
    if args.model_type.startswith("P_"):
        assert 'each' in args.info_type

    o_layers_to_process = args.layers_to_process
    pair_differ = args.pair_differ
    args.layers_to_process = process_layers_to_process(args.layers_to_process)
    
    INPUT = "hidden_states"
    args.device = torch.device(training_args.device)
    internal_model_name = args.internal_model_name
    save_attention = False
    print('device:', args.device)
    
    args.task = "regression"
    if training_args.do_predict:
        training_args.do_eval = False  # why args doesn't work????????????
    train_dataset, val_dataset, predict_dataset = None, None, None
    if training_args.do_train:
        if args.use_val_to_train:
            train_split = 'val'
        else:
            train_split = 'train'
        train_dataset = HiddenLayersDataset(args.data_paths, train_split, args.layers_to_process, 
                                            args.save_cache, 
                                            args.info_type, args.label_name,
                                            args.ignore_missing_info, pair_differ,
                                            model_name=internal_model_name)
        print("train_dataset prepared", len(train_dataset))
    if training_args.do_eval: # why args doesn't work????????????
        print('eval dataset')
        val_dataset = HiddenLayersDataset(args.data_paths, 'val', args.layers_to_process, 
                                          args.save_cache, 
                                          args.info_type, args.label_name,
                                          args.ignore_missing_info, pair_differ,
                                          model_name=internal_model_name)
        print("val_dataset prepared", len(val_dataset))
    if training_args.do_predict:
        torch.backends.cudnn.enabled = False #RuntimeError: cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.
        predict_dataset = HiddenLayersDataset(args.data_paths, args.predict_split, args.layers_to_process, 
                                              args.save_cache, 
                                              args.info_type, args.label_name,
                                              args.ignore_missing_info, pair_differ,
                                              model_name=internal_model_name)
        print("predict_dataset prepared", len(predict_dataset))
    torch.cuda.empty_cache()

    if not args.p_num_latents_list:
        args.p_num_latents_list = [-1]
    for p_num_latents in args.p_num_latents_list:
        args.p_num_latents = p_num_latents
        if "LlamaMLP" in args.model_type or "Linear" in args.model_type: # concatenate layers hidden states MLP
            if "Qwen" in args.internal_model_name:
                args.input_dim = len(args.layers_to_process) * abs(args.p_num_latents) * 3584
            else:
                args.input_dim = len(args.layers_to_process) * abs(args.p_num_latents) * 4096

        
        label_list = None
        num_labels = 1
        label_to_id = None
        
        ########################################################
        metric = evaluate.load("mse")
        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.squeeze(preds)
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        ########################################################
        model = load_regressor_model(args)
        model = model.to(training_args.device)
        print(f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}")
        
        if training_args.fsdp:
            training_args.fsdp_config={
                "sharding_strategy": "FULL_SHARD",  # Full sharding strategy
                "cpu_offload": False,  # Offload parameters to CPU to save GPU memory
                "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",  # Auto wrap policy based on transformer layers
                "limit_all_gathers": True,  # Limit all-gather operations to reduce memory usage
                "xla": False,  # Disable XLA for FSDP
                # "activation_checkpointing": True
        }  
        training_args.dataloader_pin_memory = True
        training_args.dataloader_shuffle = True
        if 'each' in args.info_type:
            data_collator=data_collator_with_padding
        else:
            data_collator=default_data_collator
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=val_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )
            # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=val_dataset)
        metrics["eval_samples"] = len(val_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
        predictions = np.squeeze(predictions)
        logger.info("***** Predict results *****")
        labels = [f["label"].item() for f in predict_dataset]
        metric = evaluate.load("mse")
        mse = metric.compute(predictions=predictions, references=labels)
        logger.info("mse loss: {}".format(mse))

        current_num = 0
        for file in args.data_paths:
            dataset_name = file.split("/")[-3]
            test_data = pd.read_csv(file.format(split=args.predict_split))
            predictions_per_file = predictions[current_num:current_num+len(test_data)]
            # predictions_per_file to list
            labels_per_file = labels[current_num:current_num+len(test_data)]
            mse = metric.compute(predictions=predictions_per_file, references=labels_per_file)
            logger.info("mse loss: {}".format(mse))
            output_data_perfile = {
                "mse": mse,
                "predictions": [float(p) for p in predictions_per_file], 
                "labels": [float(l) for l in labels_per_file]
            }
            current_num += len(test_data)

            if args.predict_split == 'val':
                output_predict_file = os.path.join(training_args.output_dir, f"{dataset_name}_predict_val_results.json")
            else:
                output_predict_file = os.path.join(training_args.output_dir, f"{dataset_name}_predict_results.json")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as fout:
                    json.dump(output_data_perfile, fout)
        assert current_num == len(labels)
        logger.info("Predict results saved at {}".format(output_predict_file))
    
    kwargs = {"tasks": "text-classification"}
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)
