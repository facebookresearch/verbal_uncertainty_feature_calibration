for MODEL in 'Meta-Llama-3.1-8B-Instruct' 'Mistral-7B-Instruct-v0.3' 'Qwen2.5-7B-Instruct'
do
for TYPE in 'uncertainty' 'sentence'
do
python calibration/merge_vuf.py \
--model_name $MODEL \
--prompt_type $TYPE

done
done