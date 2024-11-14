# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

for D in trivia_qa nq_open pop_qa
do
python  datasets/download_dataset.py \
    --dataset_name $D
done