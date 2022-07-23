import os
import sys
import sklearn
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

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    print('output to ftrain_svm and ftest_svm...')

    n_pos = int(125e4)
    n_neg = int(125e4)
    n_train = n_pos + n_neg
    n_test = int(1e4)
    test_labels = [0] * n_test
    train_labels = [1] * n_pos + [0] * n_neg
    ftrain_svm = open(os.path.join('data', 'train.svm.txt'), 'wb')
    dump_svmlight_file(tfidf[0:n_train, :], train_labels, ftrain_svm)
    ftrain_svm.close()

    ftest_svm = open(os.path.join('data', 'test.svm.txt'), 'wb')
    dump_svmlight_file(tfidf[-n_test:, :], test_labels, ftest_svm)
    ftest_svm.close()
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
    solve(source = 'train_and_test_corpus.txt')
