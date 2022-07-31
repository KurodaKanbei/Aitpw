from dataclasses import field, fields
from transformers import AutoTokenizer
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split
from torchtext.legacy.data import BucketIterator, TabularDataset, Field, RawField
from transformers import AutoTokenizer
from datasets import Dataset
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
import os

def tokenize_function(examples, pretrain_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(pretrain_name)
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

     

def create_data(pos_file_name, neg_file_name, dir="../data/", train_file="train.json", test_file="test.json"):


    pos_file, neg_file = open(dir+pos_file_name, encoding="utf8").read().split("\n"), open(dir+neg_file_name, encoding="utf8").read().split("\n")
    data = {
        "text":([line for line in pos_file]+[line for line in neg_file]),
        "label": ([1]*len(pos_file)+[0]*len(neg_file))
            }
    data_df = pd.DataFrame(data, columns=["text", "label"])

    train, test = train_test_split(data_df, test_size = 0.1)


    trainset, testset =  Dataset.from_pandas(train),  Dataset.from_pandas(test)
    datasets = DatasetDict({"train":trainset, "test":testset})
    tokenized_datasets = datasets.map(tokenize_function, batched=True)
    tokenized_datasets["train"].to_json("data/"+train_file)
    tokenized_datasets["test"].to_json("data/"+test_file)

def create_test_data(save_dir="data/"):
    dir="../data/"
    test_file = open(dir+"test_data_preprocessed.txt", encoding="utf8").read().split("\n")
    
    data = {
        "text":([line for line in test_file]),
        "label": ([-1]*len(test_file)),
            }
    data_df = pd.DataFrame(data, columns=["text", "label"])
    testset = Dataset.from_pandas(data_df)
    datasets = DatasetDict({"train":testset})
    tokenized_datasets = datasets.map(tokenize_function, batched=True)
    tokenized_datasets["train"].to_json(save_dir+"res_base_uncased.json")
    print(len(tokenized_datasets["train"]))

if __name__ == "__main__":
    if not os.path.exists('data'):
        os.makedirs('data')
    create_data("train_pos_full_augmented.txt", "train_neg_full_augmented.txt")
    create_test_data()

    
