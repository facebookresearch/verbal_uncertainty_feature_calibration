## QUESTION

for D in trivia_qa
do
for S in test train val
do
for MODEL in "Meta-Llama-3.1-8B-Instruct"
do
python probe/scripts/submit_job_get_hidden_state.py \
--source_dir $D \
--splits $S \
--datasplit $MODEL \
--internal_model_name $MODEL \
--info_type only_question_last &

done
done
done