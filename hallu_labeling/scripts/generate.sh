for D in 'trivia_qa' 'nq_open' 'pop_qa' 
do
for SPLIT in 'train' 'val' 'test'
do
for MODEL in 'Meta-Llama-3.1-8B-Instruct' 'Mistral-7B-Instruct-v0.3' 'Qwen2.5-7B-Instruct'
do
for T in 0.1
do
python sem_uncertainty/scripts/submit_job_generate.py \
--dataset $D \
--prompt_type "sentence" \
--split $SPLIT \
--temperature $T \
--model_name $MODEL &

done
done
done
done