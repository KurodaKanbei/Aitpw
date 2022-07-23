import os
import sys
import sklearn
import pickle
import numpy as np
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import gensim.downloader
d_features = 200
glove_vectors = gensim.downloader.load('glove-twitter-' + str(d_features))

def solve(source, dest):
    
    corpus = []
    cnt = 0
    with open(os.path.join('data', source), 'r') as f:
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

    for i in range(len(corpus)):
        if i % 10000 == 0:
            print(i)
        cnt = 0
        vector = []
        row = tfidf.getrow(i)
        for j, idx in enumerate(row.indices):
            vector.append(row.data[j] * vocal[idx])
            if glove_vectors.__contains__(word[idx]):
                cnt += 1
        vec = np.sum(vector) / max(cnt, 1)
        vectors[i] = vec

    # with open(os.path.join('data', dest), 'wb') as f:
    #    pickle.dump(vectors, f)
    np.save(os.path.join('data', dest), vectors)

if __name__ == '__main__':
    solve(source = 'train_and_test_corpus.txt', dest = 'features.npy')
