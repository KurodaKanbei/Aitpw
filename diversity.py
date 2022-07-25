import os
import sys
import sklearn
import random
import numpy as np
from sklearn import feature_extraction
from sklearn.datasets import dump_svmlight_file
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import gensim.downloader
d_features = 25
glove_vectors = gensim.downloader.load('glove-twitter-' + str(d_features))

def compute_diverse(vectors, top_k=2):
    vectors = vectors[-top_k:]
    M = np.zeros((top_k, d_features))


def solve(source):
    
    corpus = []
    cnt = 0
    with open(os.path.join('data', source), 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            corpus.append(line)
            cnt += 1

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    word = vectorizer.get_feature_names_out()

    vocal = []
    for i in range(len(word)):
        if glove_vectors.__contains__(word[i]):
            vocal.append(glove_vectors[word[i]])
        else:
            vocal.append(np.zeros(d_features)) 
    vectors = np.zeros(shape=(len(corpus), d_features))

    diverse = []
    for i in range(len(corpus)):
        if i % 10000 == 0:
            print(i)
        cnt = 0
        vector = []
        row = tfidf.getrow(i)
        for j, idx in enumerate(row.indices):
            if glove_vectors.__contains__(word[idx]):
                vector.append((row.data[j], vocal[idx]))
                cnt += 1
        vector = sorted(vector, key=lambda x : x[0])
        if len(vector) >= 2:
            diverse.append(compute_diverse(vector))
    # with open(os.path.join('data', dest), 'wb') as f:
    #    pickle.dump(vectors, f)
    # np.save(os.path.join('data', dest), vectors)

if __name__ == '__main__':
    solve(source = 'train_and_test_corpus.txt')
