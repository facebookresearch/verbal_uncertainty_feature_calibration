import argparse
from datasets import load_dataset, Dataset
import random
import csv
import tqdm.auto as tqdm
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_dir)


def trivia_qa_write_file(data, out_dir, split):
    all_ids = set()
    with open(f"{out_dir}/{split}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "answer"])
        for i in range(len(data)):
            id = data[i]['question_id']
            assert id not in all_ids
            all_ids.add(id)
            writer.writerow([id, data[i]['question'], data[i]['answer']['aliases']])



def nq_open_write_file(data, out_dir, split):
    all_questions = []
    with open(f"{out_dir}/{split}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "answer"])
        for i in range(len(data)):
            question =  data[i]['question']
            assert question not in all_questions
            all_questions.append(question)
            writer.writerow([f'{split}_{i}',question, data[i]['answer']])


def pop_qa_write_file(data, out_dir, split):
    all_ids = set()
    with open(f"{out_dir}/{split}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "answer"])
        for i in range(len(data)):
            id = data[i]['id']
            assert id not in all_ids
            all_ids.add(id)
            writer.writerow([id, data[i]['question'], data[i]['possible_answers']])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='trivia_qa')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    # make directory for dataset
    out_dir = f"{current_dir}/{dataset_name}/sampled"
    os.makedirs(out_dir, exist_ok=True)

    if dataset_name == 'trivia_qa':
        data = load_dataset("mandarjoshi/trivia_qa", 'rc')

        # There are some repated question_id ids in the dataset, so we need to remove them
        train_data = data['train']
        train_df = train_data.to_pandas()
        train_df = train_df.drop_duplicates(subset='question_id')
        train_data = Dataset.from_pandas(train_df)
        train_data = train_data.shuffle(seed=42).select(range(10000))

        validation_data = data['validation']
        validation_df = validation_data.to_pandas()
        validation_df = validation_df.drop_duplicates(subset='question_id')
        validation_data = Dataset.from_pandas(validation_df)
        test_val_data = validation_data.shuffle(seed=42).select(range(2000))
        test_data = test_val_data.select(range(1000))
        val_data = test_val_data.select(range(1000, 2000))

        trivia_qa_write_file(train_data, out_dir, "train")
        trivia_qa_write_file(val_data, out_dir, "val")
        trivia_qa_write_file(test_data, out_dir, "test")

    elif dataset_name == 'nq_open':
        # Download the NQ Open dataset
        data = load_dataset("google-research-datasets/nq_open")

        train_data = data['train'].shuffle(seed=42).select(range(10000))
        test_val_data = data['validation'].shuffle(seed=42).select(range(2000))
        # split test_val_data into test and val
        test_data = test_val_data.select(range(1000))
        val_data = test_val_data.select(range(1000, 2000))

        nq_open_write_file(train_data, out_dir, "train")
        nq_open_write_file(test_data, out_dir, "test")
        nq_open_write_file(val_data, out_dir, "val")

    elif dataset_name == 'pop_qa':
        # Download the PopQA dataset
        data = load_dataset("akariasai/PopQA")

        data = data['test'].shuffle(seed=42).select(range(12000))
        train_data = data.select(range(10000))
        test_data = data.select(range(10000, 11000))
        val_data = data.select(range(11000, 12000))

        pop_qa_write_file(train_data, out_dir, "train")
        pop_qa_write_file(test_data, out_dir, "test")
        pop_qa_write_file(val_data, out_dir, "val")


    