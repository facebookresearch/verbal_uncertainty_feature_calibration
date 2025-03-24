
for D in 'nq_open' 'pop_qa' 'trivia_qa'
do
for MODEL in 'Meta-Llama-3.1-8B-Instruct' 'Mistral-7B-Instruct-v0.3' 'Qwen2.5-7B-Instruct'
do
for TYPE in 'uncertainty' 'sentence'
do
for SPLIT in train test val
do
python calibration/scripts/submit_job.py \
--task "universal_vuf" \
--dataset $D \
--split $SPLIT \
--model_name $MODEL \
--prompt_type $TYPE

done
done
done
done