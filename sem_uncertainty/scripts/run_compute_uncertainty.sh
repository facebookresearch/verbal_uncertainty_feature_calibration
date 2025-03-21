for D in 'pop_qa'
do
for SPLIT in train
do
for MODEL in Mistral-7B-Instruct-v0.3
do
python semantic_entropy/compute_semantic_entropy.py \
--dataset $D \
--split $SPLIT \
--model_name $MODEL \
--port 'http://learnfair6036:8000/v1' &

done
done
done
