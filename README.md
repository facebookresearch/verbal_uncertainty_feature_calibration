# datasets
download and process datasets using build_data.ipynb

merge the generated answers and calcualted uncertaities using merge.ipynb

# internal_information
save the hidden states

# detector
our main hallucination detector which is a binary classifier

### LogisticRegression.ipynb
Determining the appropriate uncertainty for hallucination detection


# sem_uncertainty
modified from SEP paper https://github.com/OATML/semantic-entropy-probes

### semantic_entropy
1. sample multi answers 
2. calcualte semantic entropy

### eigen
calcualte eigen based on sampled multi answers 

# verbal_uncertainty
1. generate answer 
2. use LLM to judage the verbal uncertainty level.


# probe
uncertainty probe (regressor)
use question's hidden state to predict the uncertainties (ling_uncertainty and semantic entropy and eigen)

# calibration
calibrate verbal uncertainty with semantic entropy to mitigate hallucinations