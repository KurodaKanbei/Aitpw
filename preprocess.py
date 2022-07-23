import os
import wordninja
import re

document = []
cnt = 0

def solve(source, test=False):
    global document
    global cnt
    with open(os.path.join('data', source), 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            if test:
                pos = line.find(',')
                line = line[pos + 1:];
            sentence = line.split(' ')
            words = []
            for word in sentence:
                if word == '<user>' or word == '<url>':
                    continue
                _word = re.sub('[^a-zA-Z0-9]+', '', word)
                if _word == '':
                    continue
                if word[0] == '#':
                    for vocal in wordninja.split(_word):
                        words.append(vocal)
                else:
                    words.append(_word)
            sentence = ' '.join(str(word) for word in words) + '\n'
            cnt = cnt + 1
            document.append(sentence)
            if cnt % 10000 == 0:
                print(cnt)

if __name__ == '__main__':
    solve(source = 'train_pos_full.txt')
    solve(source = 'train_neg_full.txt')
    solve(source = 'test_data.txt', test=True)
    dest = 'train_and_test_corpus.txt'
    with open(os.path.join('data', dest), 'w') as f:
        f.writelines(document)
