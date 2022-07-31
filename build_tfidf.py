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
glove_vectors = None
# glove_vectors = gensim.downloader.load('glove-twitter-' + str(d_features))

def solve(source):
    
    corpus = []
    cnt = 0
    with open(os.path.join('data', source), 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            corpus.append(line)
            cnt += 1

    vectorizer = CountVectorizer(ngram_range=(1, 2))
    transformer = TfidfTransformer()
    
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    print('output to ftrain_svm and ftest_svm...')

    n_pos = int(125e4)
    n_neg = int(125e4)
    n_full = n_pos + n_neg
    n_test = int(1e4)
    test_labels = [0] * n_test
    full_labels = [1] * n_pos + [0] * n_neg

    ffull_svm = open(os.path.join('data', 'full.svm.txt'), 'wb')
    dump_svmlight_file(tfidf[:n_full], full_labels, ffull_svm)
    ffull_svm.close()

    ftest_svm = open(os.path.join('data', 'test.svm.txt'), 'wb')
    dump_svmlight_file(tfidf[-n_test:, :], test_labels, ftest_svm)
    ftest_svm.close()

    print('start split')
    with open(os.path.join('data', 'full.svm.txt'), 'rb') as f:
        f_train = open(os.path.join('data', 'train.svm.txt'), 'wb')
        f_val = open(os.path.join('data', 'val.svm.txt'), 'wb')
        for line in f.readlines():
            if random.randint(0, 9) == 0:
                f_val.write(line)
            else:
                f_train.write(line)
        f_train.close()
        f_val.close()
    exit(0)

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
    # np.save(os.path.join('data', dest), vectors)

if __name__ == '__main__':
    solve(source = 'train_and_test_corpus_8_8.txt')
