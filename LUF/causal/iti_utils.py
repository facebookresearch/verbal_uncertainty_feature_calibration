import torch
import numpy as np
from tqdm.auto import tqdm
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

#################
hedge_resource_dir = "/data/home/ncan/data/hedging-resources"
hedge_word_file = "hedge_words.txt"
discourse_marker_file = "discourse_markers.txt"
booster_words_file = "booster_words.txt"

esu_filename = "expression_subjective_uncertainty.llama3.1.405B.txt"
euu_filename = "expression_universal_uncertainty.llama3.1.405B.txt"
#################

def remove_all_hooks(model):
    """Remove all forward/backward hooks from the model."""
    for module in model.modules():
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()
        module._backward_hooks.clear()
        

def register_feature_ablation_hook(model, Hs_feature, layer_idx, alpha=1.0):
    for l in layer_idx:
        device_idx_l = model.hf_device_map[f"model.layers.{l}"]
        h_feature_l = Hs_feature[l].to(f"cuda:{device_idx_l}")  # (h_dim)
        h_feature_l = h_feature_l / torch.sqrt(h_feature_l.pow(2).sum(-1))

        def make_feature_ablation_hook(h_feature_l, alpha):
            def feature_ablation_hook(module, inputs, outputs):
                if isinstance(outputs, tuple):
                    outputs_0 = outputs[0]   # (B, seq_len, h_dim)
                    if outputs_0.shape[1] > 1:
                        outputs_0 += h_feature_l * alpha
                        
                    return (outputs_0,) + outputs[1:]
                else:
                    if outputs.shape[1] > 1:
                        outputs += h_feature_l * alpha

                    return outputs
            return feature_ablation_hook

        model.model.layers[l].register_forward_hook(
            make_feature_ablation_hook(h_feature_l, alpha)
        )


def eval_ans_hedgeness_emb(emb_model, iti_results_df):

    ##### load and encode esu/euu hedging expressions #####
    with open(f"{hedge_resource_dir}/{esu_filename}") as f:
        esu_strings = f.readlines()
    with open(f"{hedge_resource_dir}/{euu_filename}") as f:
        euu_strings = f.readlines()

    eu_embeddings = F.normalize(emb_model.encode(esu_strings + euu_strings, instruction="", max_length=512), p=2, dim=1)
    #############################

    ##### get mean cosine sim scores to pre-defined hedging expressions #####
    lu_scores_emb = []
    for i in tqdm(range(iti_results_df.shape[0])):
        with torch.no_grad():
            model_answers = iti_results_df['model answer with iti'][i]
            model_answer_embs = F.normalize(emb_model.encode(
                model_answers, instruction = "", max_length=512
            ), p=2, dim=1)

            eu_cos_sim = (model_answer_embs @ eu_embeddings.T).mean().cpu().item()
        lu_scores_emb.append(eu_cos_sim)
    ##########################################

    return lu_scores_emb

