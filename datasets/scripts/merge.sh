# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

for D in trivia_qa nq_open pop_qa
do
for MODEL in 'Meta-Llama-3.1-8B-Instruct' 'Mistral-7B-Instruct-v0.3' 'Qwen2.5-7B-Instruct'
do
python  datasets/merge.py \
    --dataset $D \
    --model $MODEL
done
done