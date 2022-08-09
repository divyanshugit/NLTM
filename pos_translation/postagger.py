
from lib2to3.pgen2 import token
import nltk
from nltk.corpus import indian
from nltk.tag import tnt
import string
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def train_pos():
    tagged_set = 'hindi.pos'
    word_set = indian.sents(tagged_set)
    count = 0
    for sen in word_set:
        count += 1
        sen = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in sen]).strip()

    train_perc = 0.9
    train_rows = int(train_perc*count)
    test_rows = train_rows+1

    data = indian.tagged_sents(tagged_set)
    train_data = data[:train_rows]
    test_data = data[test_rows:]

    pos_tagger = tnt.TnT()
    pos_tagger.train(train_data)
    pos_tagger.accuracy(test_data)

    return pos_tagger

def pos_tagger_fn(sentence, pos_tagger):
    tokenized = nltk.word_tokenize(sentence)
    return pos_tagger.tag(tokenized)

def one_hot_encoder_fn(sentence):
    pos_tagger = train_pos()
    data = pos_tagger_fn(sentence, pos_tagger)
    df = pd.DataFrame(data, columns=['text','pos'])
    onehotencoder = OneHotEncoder()
    X = onehotencoder.fit_transform(df.pos.values.reshape(-1,1)).toarray()
    return X

test_sentence = "३९ गेंदों में दो चौकों और एक छक्के की मदद से ३४ रन बनाने वाले परोरे अंत तक आउट नहीं हुए ।"

print(one_hot_encoder_fn(test_sentence))
