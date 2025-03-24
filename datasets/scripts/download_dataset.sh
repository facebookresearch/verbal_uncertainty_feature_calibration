for D in trivia_qa nq_open pop_qa
do
python  datasets/download_dataset.py \
    --dataset_name $D
done