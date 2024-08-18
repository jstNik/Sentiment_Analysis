import os
import re
import string
from enum import Enum
from pathlib import Path

import contractions
import pandas as pd
from nltk import WordNetLemmatizer, word_tokenize, pos_tag
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


class Model(Enum):
    NAIVE_BAYES = MultinomialNB()
    RANDOM_FOREST = RandomForestClassifier()
    PERCEPTRON = Perceptron()


def cleaning(text):
    lemmatizer = WordNetLemmatizer()

    # converting to lowercase, removing URL links, special characters, punctuations...
    text = text.lower()  # converting to lowercase
    text = contractions.fix(text)
    text = re.sub(r'http(s?)://(\S*)|www.(\S*)', '', text)  # removing URL links
    text = re.sub(r'\b\d+\b', '', text)  # removing number
    text = re.sub('<.*?>+', '', text)  # removing special characters
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # punctuations
    text = re.sub('\n', '', text)
    text = re.sub('[’“”…]', '', text)

    # removing emoji:
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    tokenized_text = word_tokenize(text)
    pos_tuple = pos_tag(tokenized_text)
    text, pos_text, adj, position = '', '', '', ''
    for idx, t in enumerate(pos_tuple):
        if t[0] in stopwords.words('english'):
            continue
        lem = lemmatizer.lemmatize(t[0])
        text += ' ' + lem
        pos_text += ' ' + lem + '_' + t[1]
        if t[1] == 'JJ' or t[1] == 'JJR' or t[1] == 'JJS':
            adj += ' ' + lem
        if idx < len(pos_tuple) // 4:
            position += ' ' + lem + '_' + 'F'
        elif idx > len(pos_tuple) * 3 // 4:
            position += ' ' + lem + '_' + 'L'
        else:
            position += ' ' + lem + '_' + 'H'
    text = text[1:]
    pos_text = pos_text[1:]
    adj = adj[1:]
    return text, pos_text, adj, position


def run(cv: CountVectorizer, k_best: int, x, y):
    # reviews = cv.fit_transform(x)
    selector = SelectKBest(chi2, k=k_best)
    pipeline = make_pipeline(cv) if k_best == 0 else make_pipeline(cv, selector)
    reviews = pipeline.fit_transform(x, y)
    features = reviews.shape[1]
    x_train, x_test, y_train, y_test = train_test_split(reviews, y, test_size=0.3, shuffle=True)
    accuracy = []
    for model in Model:
        model.value.fit(x_train, y_train)
        predicted = model.value.predict(x_test)
        accuracy.append(round(accuracy_score(predicted, y_test) * 100, 1))
    return features, accuracy


def main():
    directories = ['dataset/tokens/neg/', 'dataset/tokens/pos/']
    data = []

    for i in range(len(directories)):
        for path in os.listdir(directories[i]):
            file = open(directories[i] + path)
            review, pos, adj, position = cleaning(file.read())
            data.append([i, review, pos, adj, position])
            file.close()

    df = pd.DataFrame(data, columns=['sentiment', 'review', 'review + pos', 'adjectives', 'positions'])

    test_cases = [
        [(1, 1), False, df['review'], 0, 'Unigram', 'Freq.'],
        [(1, 1), True, df['review'], 0, 'Unigram', 'Pres.'],
        [(1, 2), True, df['review'], 0, 'Unigram + Bigrams', 'Pres.'],
        [(2, 2), True, df['review'], 0, 'Bigrams', 'Pres.'],
        [(1, 1), True, df['review + pos'], 0, 'Unigram + POS', 'Pres.'],
        [(1, 1), True, df['adjectives'], 0, 'Adjectives', 'Pres.'],
        [(1, 1), True, df['review'], 2633, 'Top 2633 unigrams', 'Pres.'],
        [(1, 1), True, df['positions'], 0, 'Unigrams + positions', 'Pres.']
    ]

    results = []
    for n, b, x, k, m, f in test_cases:
        features, accuracy = run(CountVectorizer(ngram_range=n, binary=b), k, x, df['sentiment'])
        results.append([m, features, f] + accuracy)

    pd_res = pd.DataFrame(results, columns=['Feature', '# of features', 'Frequency or Presence', 'Naive Bayes', 'Random Forest', 'Perceptron'])
    pd_res.to_csv(Path('result.csv'))

    pass


if __name__ == "__main__":
    main()
