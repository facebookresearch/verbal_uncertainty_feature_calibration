bash
conda activate detect
cd ~/Hallu_Det/sem_uncertainty

for D in 'pop_qa'
do
for SPLIT in train
do
for MODEL in Mistral-7B-Instruct-v0.3
do
python ~/Hallu_Det/sem_uncertainty/semantic_entropy/compute_semantic_entropy.py \
--dataset $D \
--split $SPLIT \
--model_name $MODEL \
--port 'http://learnfair6036:8000/v1' &

done
done
done



'trivia_qa'
Qwen2.5-7B-Instruct
test val tmux 0  6008
train tmux 1 6020

Mistral-7B-Instruct-v0.3
test val tmux 2 6021
train tmux 3 6023


'nq_open' 
Qwen2.5-7B-Instruct
test val tmux 4 6024
train tmux 5 6026

Mistral-7B-Instruct-v0.3
test val tmux 6 6039
train tmux 7 6031


'pop_qa'
Qwen2.5-7B-Instruct
test val tmux 8 6038
train tmux 9 6033

Mistral-7B-Instruct-v0.3
test val tmux 10 6035
train tmux 11 6036
 
