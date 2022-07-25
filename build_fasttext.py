import random
import os

def solve(source):
    corpus = []
    with open(os.path.join('data', source), 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            corpus.append(line)
    n_pos = int(125e4)
    n_neg = int(125e4)
    train_list = []
    val_list = []
    ft_train = open(os.path.join('data', 'ft_train.txt'), 'w', encoding='UTF-8')
    ft_val = open(os.path.join('data', 'ft_val.txt'), 'w', encoding='UTF-8')
    for i in range(n_pos + n_neg):
        s = ''
        if i < n_pos:
            s = '__label__1 ' + corpus[i]
        else:
            s = '__label__0 ' + corpus[i]
        t = random.randint(0, 10)
        if t > 0:
            train_list.append(s)
        else:
            val_list.append(s)
    random.shuffle(train_list)
    random.shuffle(val_list)
    ft_train.writelines(train_list)
    ft_val.writelines(val_list)
    ft_train.close()
    ft_val.close()


if __name__ == '__main__':
    solve(source='train_and_test_corpus.txt')