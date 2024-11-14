# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#### causal ####
for D in 'trivia_qa'
do
for A in $(seq 0.25 0.25 2.0)
do
for MODEL in "Meta-Llama-3.1-8B-Instruct"
do
python calibration/eval/eval_vu.py \
--dataset $D \
--model_name $MODEL \
--prompt_type "sentence" \
--str_process_layers "range(15,32)" \
--split "val" \
--question_type 'uncertain' \
--max_alpha -${A} \
--batch_size 8 \
--iti_method 2 \
--entailment_model '0' &
done
done
done

for D in 'trivia_qa'
do
for A in $(seq 0.25 0.25 2.0)
do
for MODEL in "Meta-Llama-3.1-8B-Instruct"
do
python calibration/eval/eval_vu.py  \
--dataset $D \
--model_name $MODEL \
--prompt_type sentence \
--str_process_layers "range(15,32)" \
--split val \
--question_type certain \
--max_alpha $A \
--batch_size 8 \
--iti_method 2 \
--entailment_model '0' &
done
done
done
