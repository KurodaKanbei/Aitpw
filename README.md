## Twitter Datasets
Download the tweet datasets from here:
http://www.da.inf.ethz.ch/teaching/2018/CIL/material/exercise/twitter-datasets.zip

The dataset should have the following files:
- sample_submission.csv
- train_neg.txt :  a subset of negative training samples
- train_pos.txt: a subset of positive training samples
- test_data.txt:
- train_neg_full.txt: the full negative training samples
- train_pos_full.txt: the full positive training samples


## Project Structure
- data/ - original data and preprocessed data in txt format
- bert/ - classification scripts using bert
    - data/ - folder holding tokenized data
    - data.py - generating tokenized data
    - run.ipynb - google colab code for model finetuing
    - inference.ipynb - google colab code for model inference


## Data Before Model Training (Necessary)
- Put all data under data/
- To preprocess data, run "python preprocess.py"
- To augment data, run "python diversity.py"


## Bert Based Prediction
- To enter the bert directory, "cd bert"
- To tokenize data, run "data.py"
- Put bert/data on Google Drive as data/
- Run "run.ipynb" from Google Colab for training
- Run "inference.ipynb" from Google Colab for inference

## Back-translation
- Put data/train_pos_full_preprocessed.txt, data/train_neg_full_preprocessed.txt under data/ on Google Drive
- Enter the bert directory, "cd bert"
- run "backtranslation.ipynb" from Google Colab


