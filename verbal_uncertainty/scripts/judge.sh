
for D in 'trivia_qa'
do
for SPLIT in 'train' 'val' 'test'
do
for MODEL in 'Mistral-7B-Instruct-v0.3'
do
python verbal_uncertainty/vu_llm_judge.py \
--results_dir 'outputs' \
--dataset $D \
--split $SPLIT \
--model_name $MODEL \
--port 'http://learnfair6025:8000/v1' &
done
done
done