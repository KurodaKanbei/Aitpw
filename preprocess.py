import os
import wordninja
import re
# import nltk
# from nltk.corpus import stopwords, wordnet
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
document = []
cnt = 0

def filter(word):
    return re.sub('[^a-zA-Z0-9]+', '', word)

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def solve(source, test=False):
    global document
    global cnt
    contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                           "could've": "could have", "couldn't": "could not", "didn't": "did not",
                           "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
                           "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
                           "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
                           "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",
                           "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have",
                           "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
                           "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                           "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                           "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                           "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                           "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                           "she'll've": "she will have", "she's": "she is", "should've": "should have",
                           "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                           "so's": "so as", "this's": "this is", "that'd": "that would", "that'd've": "that would have",
                           "that's": "that is", "there'd": "there would", "there'd've": "there would have",
                           "there's": "there is", "here's": "here is", "they'd": "they would",
                           "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
                           "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
                           "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
                           "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",
                           "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have",
                           "where'd": "where did", "where's": "where is", "where've": "where have",
                           "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
                           "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                           "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                           "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                           "you'll've": "you will have", "you're": "you are", "you've": "you have"}
    mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling',
                    'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',
                    'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize',
                    'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What',
                    'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can',
                    'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I',
                    'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation',
                    'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis',
                    'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017',
                    '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess',
                    "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization',
                    'demonitization': 'demonetization', 'demonetisation': 'demonetization'}
    # stop_words = stopwords.words('english')
    # lemmatizer = WordNetLemmatizer()
    with open(os.path.join('data', source), 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip()
            if test:
                pos = line.find(',')
                line = line[pos + 1:]
            sentence = line.split(' ')
            words = []
            for word in sentence:
                if word == '<user>' or word == '<url>' or word == '':
                    continue
                if word[0] == '#':
                    if word == '#':
                        continue
                    #_word = re.sub('[^a-zA-Z0-9]+', '', word[1:])
                    for vocal in wordninja.split(word[1:]):
                        words.append(vocal)
                    continue
                # _word = re.sub('[^a-zA-Z0-9]+', '', word)
                # if word == '' or word in stop_words:
                #     continue
                if word in contraction_mapping.keys():
                    for vocal in contraction_mapping[word].split(' '):
                        words.append(vocal)
                elif word in mispell_dict.keys():
                    for vocal in mispell_dict[word].split(' '):
                        words.append(vocal)
                else:
                    words.append(word)
            # sentence = ' '.join(str(re.sub('[^a-zA-Z0-9]+', '', word)) for word in words) + '\n'
            sentence = []

            i = 0
            while i < len(words):
                _word = re.sub('[^a-zA-Z0-9]+', '', words[i])
                if _word == '' or _word == ' ':
                    i += 1
                    continue
                if len(_word) >= 2:
                    sentence.append(_word)
                    i += 1
                    continue
                if len(_word) == 1:
                    if _word == 'u':
                        sentence.append(_word)
                        i += 1
                        continue
                    else:
                        j = i
                        s = ''
                        while j < len(words):
                            _word = re.sub('[^a-zA-Z0-9]+', '', words[j])
                            if len(_word) <= 1:
                                s = s + _word
                                j += 1
                            else:
                                break
                        i = j
                        for vocal in wordninja.split(s):
                            sentence.append(vocal)
            if len(sentence) == 0:
                sentence = [filter(word) for word in line.split(' ')]
            sentence = ' '.join(word for word in sentence) + '\n'

            ''' 
            words = word_tokenize(sentence)
            sentence = []
            for word in words:
                _word = re.sub('[^a-zA-Z0-9]+', '', word)
                if _word == '' or _word in stop_words:
                    continue
                if len(_word) >= 3:
                    _word = lemmatizer.lemmatize(word, get_wordnet_pos(word))
                _word = re.sub('[^a-zA-Z0-9]+', '', word)
                if _word == '' or _word in stop_words:
                    continue
                sentence.append(_word)
            s = ' '.join(str(word) for word in sentence) + '\n'
            '''
            cnt = cnt + 1
            document.append(sentence)
            if cnt % 10000 == 0:
                print(cnt)

if __name__ == '__main__':
    solve(source = 'train_pos_full.txt')
    solve(source = 'train_neg_full.txt')
    solve(source = 'test_data.txt', test=True)
    dest = 'train_and_test_corpus.txt'
    with open(os.path.join('data', dest), 'w', encoding='UTF-8') as f:
        f.writelines(document)
