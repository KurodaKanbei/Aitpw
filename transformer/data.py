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


def tokenize_function(examples, pretrain_name="bert-base-cased"):
    tokenizer = AutoTokenizer.from_pretrained(pretrain_name)
    return tokenizer(examples["text"], padding="max_length", truncation=True)

     

def create_data(args, pos_file_name, neg_file_name, dir="../data/"):
    fields={
    "text": ("text", Field(sequential=True, use_vocab=True, lower=True)), 
    "label": ("label", RawField()),
    }


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
    tokenized_datasets["train"].to_json("data/train.json")
    tokenized_datasets["test"].to_json("data/test.json")


    """
    train.to_json("data/train.json", orient="records", lines=True)
    test.to_json("data/test.json", orient="records", lines=True)

    train_data, test_data = TabularDataset.splits(
        path="data/",
        train="train.json",
        test="test.json", 
        format="json",
        fields=fields,
    )


    fields["text"][1].build_vocab(train_data, max_size=10000, min_freq=2)

    train_iterator, test_iterator = BucketIterator.splits(
        (train_data, test_data),
        batch_size = args.batch_size,
        sort=False,
        device = args.device,
    )

    return train_iterator, test_iterator
    """

    
