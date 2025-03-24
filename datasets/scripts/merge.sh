for D in trivia_qa nq_open pop_qa
do
for MODEL in 'Meta-Llama-3.1-8B-Instruct' 'Mistral-7B-Instruct-v0.3' 'Qwen2.5-7B-Instruct'
do
python  datasets/merge.py \
    --dataset $D \
    --model $MODEL
done
done