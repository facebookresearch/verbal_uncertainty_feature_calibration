import os
import torch
cwd = os.getcwd()
cwd = "/".join(cwd.split("/")[:-1])
import sys
sys.path.append(cwd)
sys.path.append(f"{cwd}/internal_information")
try:
    from peft import get_peft_model, LoraConfig, TaskType
except:
    print("import peft failed")
import random
random.seed(42)

from detector.classifier_models import (
    LinearClassifierConfig,
    LinearClassifier,
    LlamaMLPClassifier2Config,
    LlamaMLPClassifier2,
    LlamaMLP_Reg2_Config,
    LlamaMLP_Reg2,
    LSTMClassifier2Config,
    LSTMClassifier2,
)

from detector.multi_task_models import (
    LlamaMLP_MT_Config,
    LlamaMLP_MT,
)

from probe.regression_models import (
    LinearRegressorConfig,
    LinearRegressor,
    LlamaMLPRegressorConfig,
    LlamaMLPRegressor
)


from detector.perceiver import (
    PerceiverLlamaMLPClassifier2,
    PerceiverLlamaMLPClassifier2Config,
    Perceiver_LSTMClassifier2,
    PerceiverLSTMClassifier2Config
)
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP 


MODEL_MAP = {
            "Linear2": LinearClassifier,
            "LlamaMLP2": LlamaMLPClassifier2,
             "LlamaMLP_Reg2": LlamaMLP_Reg2,
             "P_LlamaMLP2": PerceiverLlamaMLPClassifier2,
             "LSTM2": LSTMClassifier2,
             "P_LSTM2": Perceiver_LSTMClassifier2,
             'LinearRegressor': LinearRegressor,
             'LlamaMLPRegressor': LlamaMLPRegressor,
             "LlamaMLP_MT": LlamaMLP_MT}

def load_classifier_model(args):
    if args.pair_differ:
        NL = len(args.layers_to_process) - 1
    else:
        NL = len(args.layers_to_process)

    if args.model_type == "Linear2":
        config = LinearClassifierConfig(
            input_dim=args.input_dim,
        )

    elif args.model_type == "LlamaMLP2":
        config = LlamaMLPClassifier2Config(
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim)
        target_modules=['gate_proj', 'up_proj', 'down_proj']
        
    elif args.model_type == "LlamaMLP_Reg2":
        config = LlamaMLP_Reg2_Config(
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                alpha=args.alpha,
                penalize_only_hallucinated=args.penalize_only_hallucinated,
                regularization=args.regularization,
                info_type=args.info_type)
     
    elif args.model_type == "LlamaMLP_MT":
        config = LlamaMLP_MT_Config(
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                alpha=args.alpha,
                info_type=args.info_type,
            )
    elif args.model_type == "P_LlamaMLP2":
        assert args.info_type == "each"
        config = PerceiverLlamaMLPClassifier2Config(
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                num_layers=NL,
                num_latents=args.p_num_latents,
                share_perceiver=args.share_perceiver)

    elif args.model_type == "LSTM2":
        config = LSTMClassifier2Config(
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,)
        target_modules = ['lstm']
        
        
    elif args.model_type == "P_LSTM2":
        assert args.info_type == "each"
        config = PerceiverLSTMClassifier2Config(
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                num_layers=NL,
                num_latents=args.p_num_latents,
                share_perceiver=args.share_perceiver
            )
    ######################################################
    MDOEL_CLASS = MODEL_MAP[args.model_type]
    if args.model_path:
        model = MDOEL_CLASS.from_pretrained(args.model_path,
                                        config=config,
                                        torch_dtype=torch.bfloat16)
    else:
        model = MDOEL_CLASS(config)

    if args.lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=target_modules,
            bias='none'
        )
        model = get_peft_model(model, lora_config)
    return model



def load_regressor_model(args):
    if args.model_type == "LinearRegressor":
        config = LinearRegressorConfig(
                    input_dim=args.input_dim,)
    elif args.model_type == "LlamaMLPRegressor":
        config = LlamaMLPRegressorConfig(
                    input_dim=args.input_dim,
                    hidden_dim=args.hidden_dim)

    MDOEL_CLASS = MODEL_MAP[args.model_type]
    if args.model_path:
        model = MDOEL_CLASS.from_pretrained(args.model_path,
                                        config=config,
                                        torch_dtype=torch.bfloat16)
    else:
        model = MDOEL_CLASS(config)

    return model
