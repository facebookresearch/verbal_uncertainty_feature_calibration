import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.transforms.functional import resize
from transformers import LlamaModel, LlamaConfig, LlamaPreTrainedModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import SequenceClassifierOutputWithPast, BaseModelOutputWithPast
from typing import Optional, Tuple, Union, List
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers import PreTrainedModel, PretrainedConfig

import random
random.seed(42)

from transformers import PreTrainedModel, PretrainedConfig
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, log_loss


class LinearRegressorConfig(PretrainedConfig):
    def __init__(self, input_dim=4096, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.problem_type = "regression"
        self.label2id = None
        

class LinearRegressor(PreTrainedModel):
    def __init__(self, config):
        super(LinearRegressor, self).__init__(config)
        self.config = config
        self.score = nn.Linear(config.input_dim, 1, bias=True)

    def forward(self, x, labels=None):
        # x.shape # [b, num_layers, hidden_dim] 
        x = x.view(x.size(0), -1).contiguous()
        logits = self.score(x)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
        )


class LlamaMLPRegressorConfig(PretrainedConfig):
    def __init__(self, 
                 input_dim=4096,
                 hidden_dim=11008, 
                 **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.problem_type = "regression"
        self.label2id = None


class LlamaMLPRegressor(PreTrainedModel):
    def __init__(self, config):
        super(LlamaMLPRegressor, self).__init__(config)
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.gate_proj = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.down_proj = nn.Linear(self.hidden_dim, self.input_dim, bias=False)
        self.act_fn = nn.SiLU()
        self.score = nn.Linear(self.input_dim, 1)  # single output for regression
        
    def forward(self, x, labels=None):
        # x.shape # [b, num_layers, seq_len, hidden_dim] 
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1).contiguous()  #[b, num_layers*seq_len*hidden_dim] 
            
        act = self.act_fn(self.gate_proj(x)) #[b, hidden_dim]
        up_proj = self.up_proj(x)  #[b, hidden_dim]
        down_proj = self.down_proj(act * up_proj).contiguous()
        logits = self.score(down_proj).contiguous()

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
            
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
        )
