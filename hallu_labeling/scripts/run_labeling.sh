# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

for D in 'trivia_qa'
do
for SPLIT in val
do
for MODEL in "Mistral-7B-Instruct-v0.3"
do
python hallu_labeling/get_refusal_rate.py \
--dataset $D \
--split $SPLIT \
--model_name $MODEL \
--port '0' &

done
done
done



for D in 'trivia_qa'
do
for SPLIT in val
do
for MODEL in "Mistral-7B-Instruct-v0.3"
do
python hallu_labeling/eval_acc.py \
--dataset $D \
--split $SPLIT \
--model_name $MODEL \
--port '0' &
done
done
done
