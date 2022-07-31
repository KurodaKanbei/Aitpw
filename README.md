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

## Envrionment Set Up
### For Xgboost on Windows
	- "pip install -r requirements.txt" 
### For FastText on Unix
	- "unzip v0.9.2.zip
	   cd fastText-0.9.2
	   make
	   pip install ."

## Project Structure

- proprocess.py preprocess all the text files and merge them into a corpus txt file

- diversity.py augment the text after preprocessing

- build_tfidf.py build the sparse matrix based on TF-IDF score

- build_fasttext.py build the labeled dataset for FastText

- main.py train and inference for xgboost

- train_ft.py train and inference for FastText

- data/ - original data and preprocessed data in txt format
	- train_and_test_corpus.txt the preprocessed data (including test set)
    - train_and_test_corpus_aug_8_8.txt the augmented data (including test set)
    - full.svm.txt, train.svm.txt, val.svm.txt, test.svm.txt sparse matrix file for xgboost training and inference
    - ft_train.txt, ft_val.txt, ft_test.txt labeled text file for FastText

- models/ - saved models from xgboost and FastText

- predictions/ - generated predictions from xgboost and FastText

- bert/ - classification scripts using bert
    - data/ - folder holding tokenized data
    - data.py - generating tokenized data
    - run.ipynb - google colab code for model finetuing
    - inference.ipynb - google colab code for model inference


## Preprocessing (Necessary)
- Put all data under data/
- To preprocess data, run "python preprocess.py"
- To augment data, run "python diversity.py"

## Xgboost Based Prediction
- To build TF-IDF vectors, "python build_tfidf.py"
- To start trainning models, "python main.py"
- To continue trainning models from a checkpoint, "python main.py -snapshot [model_name]"
- To make prediction with a saved model, "python main.py -predict-only -snapshot [model_name]"
Further parameters please refer to the argparser.

## FastText Based Prediction
- To set up labeled datasets, "python build_fasttext.py"
- To start trainning and evaluating models, "python train_ft.py"

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


