# Calibrating Verbal Uncertainty as a Linear Feature to Reduce Hallucinations
Ziwei Ji*, Lei Yu*, Yeskendir Koishekenov, Yejin Bang, Anthony Hartshorn, Alan Schelten, Cheng Zhang, Pascale Fung, Nicola Cancedda

[![arXiv](https://img.shields.io/badge/arXiv-2406.15927-b31b1b.svg)](https://arxiv.org/pdf/2503.14477)

## Installation
```
git clone git@github.com:fairinternal/mechanistic_uncertainty_calibration.git
cd mechanistic_uncertainty_calibration

conda create --name vuf python==3.8.17
conda activate vuf
pip install -r requirements.txt
```

## Dataset Preparation
Download and process datasets: TriviaQA, NQ Open, and PopQA.
```
bash datasets/scripts/download_dataset.sh
```

## Set up vLLM Server
Set up [vLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#quickstart-online) sever to infer LLM. 
The LLM is used for labelling and evaluation in the following steps.
```
python src/vllm-all.py
```

## Semantic Uncertainty Calculation
1. Sample multiple answers 
```
bash sem_uncertainty/scripts/generate.sh
```
2. Calcualte semantic entropy
```
bash sem_uncertainty/scripts/run_compute_uncertainty.sh
```

## Verbal Uncertainty Calculation
1. Sample multiple answers
```
bash verbal_uncertainty/scripts/generate.sh
```
2. Calcualte verbal uncertainty via LLM-as-a-Judge
```
bash verbal_uncertainty/scripts/judge.sh
```


## Hallucination Labeling: 
1. Generate most likely answer
```
bash hallu_labeling/scripts/generate.sh
```
2. Label ACC and Refusal to get the hallucination label
```
bash hallu_labeling/scripts/run_labeling.sh
```

3. Merge the generated answers, hallucination labels and calcualted uncertaities
```
bash datasets/scripts/merge.sh
```


## Verbal Uncertainty Feature
### Feature Extraction
```
bash calibration/scripts/universal_vuf.sh
bash calibration/scripts/merge_vuf.sh
```

### Causal Validation 
1. Generation with different intensities of inference-time intervention (&alpha;).
```
bash calibration/scripts/causal.sh
```
2. Evaluate verbal uncertainty
```
bash calibration/scripts/causal_eval.sh
```

## Hallucination Detector
Train a LogisticRegression-based detector based on uncertainties
```
bash detection/scripts/detection.sh
```

## Uncertainty Calibration for Hallucination Mitigation
1. Calibrate verbal uncertainty with semantic uncertainty to mitigate hallucinations
```
bash calibration/scripts/semantic_control.sh
```
2. Evaluate verbal uncertainty, ACC, and refusal
```
bash calibration/scripts/semantic_control_eval.sh
```


## Ablation Study: Uncertainty Probe
Train uncrtainty probes to predict uncertainties (verbal uncertainty and semantic entropy) without multi-sampling.
1. Obtain the hidden states of questions
```
bash probe/scripts/get_hidden_state.sh
```
2. Train the regressor probe given hidden state to predict the uncertainties
```
bash probe/scripts/trainer.sh
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
