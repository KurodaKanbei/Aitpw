from data import create_data
from param import parse_args
from transformers import AutoModelForSequenceClassification
from datasets.dataset_dict import DatasetDict
from datasets import load_dataset
from datasets import load_metric
from transformers import TrainingArguments, Trainer

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

if __name__ == "__main__":
    args = parse_args()

    if args.create_data:
        create_data(args , "train_pos.txt", "train_neg.txt")
    
    print(load_dataset('json', data_files="data/test.json")["train"])


    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
    metric = load_metric("accuracy")
    training_args = TrainingArguments(output_dir="outputs", evaluation_strategy="epoch")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=load_dataset('json', data_files="data/train.json")["train"],
        eval_dataset=load_dataset('json', data_files="data/test.json")["train"],
        compute_metrics=compute_metrics,
        )
    
    trainer.train()