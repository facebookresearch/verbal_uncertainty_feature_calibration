# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


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