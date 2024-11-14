# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# run_certain
for D in 'trivia_qa'
do
for A in $(seq 0.25 0.25 2.0)
do
for MODEL in "Meta-Llama-3.1-8B-Instruct"
do
python calibration/scripts/submit_job.py \
--task "causal" \
--dataset $D \
--split "val" \
--model_name $MODEL \
--prompt_type sentence \
--str_process_layers "range(15,32)" \
--run_certain 1 \
--iti_method 2 \
--alpha $A &
done
done
done


# run_uncertain
for D in 'trivia_qa'
do
for A in $(seq 0.25 0.25 2.0)
do
for MODEL in "Meta-Llama-3.1-8B-Instruct"
do
python calibration/scripts/submit_job.py \
--task "causal" \
--dataset $D \
--split "val" \
--model_name $MODEL \
--prompt_type sentence \
--str_process_layers "range(15,32)" \
--run_uncertain 1 \
--iti_method 2 \
--alpha -${A}

done
done
done
