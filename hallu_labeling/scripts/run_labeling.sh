for D in 'trivia_qa'
do
for SPLIT in val
do
for MODEL in "Mistral-7B-Instruct-v0.3"
do
python hallu_labeling/get_refusal_rate.py \
--dataset $D \
--split $SPLIT \
--model_name $MODEL \
--port 'http://learnfair6036:8000/v1' &

done
done
done



for D in 'trivia_qa'
do
for SPLIT in val
do
for MODEL in "Mistral-7B-Instruct-v0.3"
do
python hallu_labeling/eval_acc.py \
--dataset $D \
--split $SPLIT \
--model_name $MODEL \
--port 'http://learnfair6035:8000/v1' &
done
done
done
