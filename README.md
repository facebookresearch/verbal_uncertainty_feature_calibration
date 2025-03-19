# Installation and Requirements
```
git clone git@github.com:fairinternal/mechanistic_uncertainty_calibration.git
cd mechanistic_uncertainty_calibration
pip install -r requirements.txt
```

# Dataset Preparation
Download and process datasets using datasets/build_data.ipynb

# Semantic Uncertainty
1. Sample multiple answers 
```
bash sem_uncertainty/scripts/generate.sh
```
2. Calcualte semantic entropy
```
bash sem_uncertainty/scripts/run_compute_uncertainty.sh
```

# Verbal Uncertainty Feature

## Verbal Uncertainty Calculation
1. Sample multiple answers
```
bash verbal_uncertainty/scripts/generate.sh
```
2. Set up [vLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#quickstart-online) sever to infer LLM
```
python src/vllm-all.py
```

3. Use LLM to judge the verbal uncertainty level
```
bash verbal_uncertainty/scripts/judge.sh
```
4. Merge the generated answers and calcualted uncertaities using datasets/merge.ipynb

## Feature Extraction
```
bash calibration/scripts/universal_luf.sh
```

## Causal Validation 
```
bash calibration/scripts/ablation.sh
calibration/hedging-causal-validate.ipynb
```

# Hallucination Detector
Train LogisticRegression(LR)-based detector based on uncertainties
```
bash detector/scripts/LR.sh
```

#  Uncertainty Probe
1. Obtain the hidden states of questions
```
bash probe/scripts/get_hidden_state.sh
```
2. Train the regressor probe given hidden state to predict the uncertainties (verbal uncertainty and semantic entropy)
```
bash probe/scripts/trainer.sh
```

# Uncertainty Calibration for Hallucination Mitigation
## Generation with inference-time intervention
calibrate verbal uncertainty with semantic entropy to mitigate hallucinations
```
bash calibration/scripts/semantic_control.sh
```
## Evaluation
```
bash calibration/scripts/run_eval.sh
```


# License
The majority of mechanistic_uncertainty_calibration is licensed under CC-BY-NC, however portions of the project are available under separate license terms: OATML is licensed under the MIT license.


# Citation
```
@article{ji2025calibrating,
  title={Calibrating Verbal Uncertainty as a Linear Feature to Reduce Hallucinations},
  author={Ziwei Ji, Lei Yu, Yeskendir Koishekenov, Yejin Bang, Anthony Hartshorn, Alan Schelten, Cheng Zhang, Pascale Fung, Nicola Cancedda},
  journal={arXiv preprint arXiv:2503.14477},
  year={2025}
}
```