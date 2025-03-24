for D in 'trivia_qa'
do
for MODEL in 'Meta-Llama-3.1-8B-Instruct'
do
for A in 1.0
do
python calibration/scripts/submit_job.py \
--task "semantic_control" \
--dataset $D \
--split "test" \
--iti_method 2 \
--max_alpha $A \
--use_predicted 0 \
--model_name $MODEL \
--str_process_layers "range(15,32)" \
--prompt_type uncertainty &

done
done
done