# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

for D in 'trivia_qa'
do
for L in "range(10,20)"
do
for U in verbal_uncertainty
do
for MODEL in "Meta-Llama-3.1-8B-Instruct"
do
for LR in 1e-5
do
python probe/scripts/submit_job_trainer.py \
--dataset $D \
--model_type "LinearRegressor" \
--layers_to_process $L \
--label_name $U \
--learning_rate $LR \
--internal_model_name $MODEL 

done
done
done
done
done
