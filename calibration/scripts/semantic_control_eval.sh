# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

for D in 'trivia_qa'
do
for MODEL in "Meta-Llama-3.1-8B-Instruct" 
do
for ITI in 2
do
for LAYER in "range(15,32)"
do
for A in 1.0
do
python calibration/eval/eval_acc.py \
--dataset $D \
--model_name $MODEL \
--prompt_type "uncertainty" \
--split "test" \
--max_alpha $A \
--batch_size 40 \
--iti_method $ITI \
--str_process_layers $LAYER \
--entailment_model '0' &


python calibration/eval/eval_refusal.py \
--dataset $D \
--model_name $MODEL \
--prompt_type "uncertainty" \
--split "test" \
--max_alpha $A \
--iti_method $ITI \
--str_process_layers $LAYER \
--port '0' &


python calibration/eval/compute_semantic_entropy.py \
--dataset $D \
--model_name $MODEL \
--prompt_type "uncertainty" \
--split "test" \
--max_alpha $A \
--batch_size 1 \
--iti_method $ITI \
--str_process_layers $LAYER \
--entailment_model '0' &


python calibration/eval/eval_vu_most_likely.py \
--dataset $D \
--model_name $MODEL \
--prompt_type "uncertainty" \
--split "test" \
--max_alpha $A \
--batch_size 8 \
--iti_method $ITI \
--str_process_layers $LAYER \
--entailment_model '0' &


python calibration/eval/eval_vu.py \
--dataset $D \
--model_name $MODEL \
--prompt_type "uncertainty" \
--split "test" \
--max_alpha $A \
--batch_size 8 \
--iti_method $ITI \
--str_process_layers $LAYER \
--entailment_model '0' &

done
done
done
done
done

