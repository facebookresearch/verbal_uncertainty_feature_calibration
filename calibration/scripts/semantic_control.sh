# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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