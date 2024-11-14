# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

for D in 'trivia_qa'
do
for SPLIT in 'train' 'val' 'test'
do
for MODEL in 'Mistral-7B-Instruct-v0.3'
do
python verbal_uncertainty/vu_llm_judge.py \
--results_dir 'outputs' \
--dataset $D \
--split $SPLIT \
--model_name $MODEL \
--port '0' &
done
done
done